XEON_PHI/    This file is part of ELPA.
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
#include <stdio.h>
#include <stdlib.h>

#define __forceinline __attribute__((always_inline))

#ifdef DOUBLE_PRECISION_COMPLEX
#define offset 8

#define __AVX512_DATATYPE __m512d
#define _AVX512_LOAD _mm512_load_pd
#define _AVX512_STORE _mm512_store_pd
#define _AVX512_SET1 _mm512_set1_pd
#define _AVX512_SET _mm512_set_pd
#define _AVX512_MUL _mm512_mul_pd
#define _AVX512_ADD _mm512_add_pd
#define _AVX512_MASK_STOREU _mm512_mask_storeu_pd
#define _AVX512_SHUFFLE _mm512_shuffle_pd
#ifdef HAVE_AVX512_XEON
#define _AVX512_XOR _mm512_xor_pd
#endif
#define _SHUFFLE 0x55

#ifdef HAVE_AVX512

#define __ELPA_USE_FMA__
#define _mm512_FMADDSUB_pd(a,b,c) _mm512_fmaddsub_pd(a,b,c)
#define _mm512_FMSUBADD_pd(a,b,c) _mm512_fmsubadd_pd(a,b,c)
#endif

#define _AVX512_FMADDSUB _mm512_FMADDSUB_pd
#define _AVX512_FMSUBADD _mm512_FMSUBADD_pd
#endif /* DOUBLE_PRECISION_COMPLEX */

#ifdef SINGLE_PRECISION_COMPLEX
#define offset 16

#define __AVX512_DATATYPE __m512
#define _AVX512_LOAD _mm512_load_ps
#define _AVX512_STORE _mm512_store_ps
#define _AVX512_SET1 _mm512_set1_ps
#define _AVX512_SET _mm512_set_ps
#define _AVX512_MUL _mm512_mul_ps
#define _AVX512_ADD _mm512_add_ps
#define _AVX512_MASK_STOREU _mm512_mask_storeu_ps
#define _AVX512_SHUFFLE _mm512_shuffle_ps
#ifdef HAVE_AVX512_XEON
#define _AVX512_XOR _mm512_xor_ps
#endif
#define _SHUFFLE 0xb1

#ifdef HAVE_AVX512

#define __ELPA_USE_FMA__
#define _mm512_FMADDSUB_ps(a,b,c) _mm512_fmaddsub_ps(a,b,c)
#define _mm512_FMSUBADD_ps(a,b,c) _mm512_fmsubadd_ps(a,b,c)
#endif

#define _AVX512_FMADDSUB _mm512_FMADDSUB_ps
#define _AVX512_FMSUBADD _mm512_FMSUBADD_ps
#endif /* SINGLE_PRECISION_COMPLEX */

//Forward declaration
#ifdef DOUBLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_16_AVX512_2hv_double(double complex* q, double complex* hh, int nb, int ldq, int ldh, double complex s);
static __forceinline void hh_trafo_complex_kernel_12_AVX512_2hv_double(double complex* q, double complex* hh, int nb, int ldq, int ldh, double complex s);
static __forceinline void hh_trafo_complex_kernel_8_AVX512_2hv_double(double complex* q, double complex* hh, int nb, int ldq, int ldh, double complex s);
static __forceinline void hh_trafo_complex_kernel_4_AVX512_2hv_double(double complex* q, double complex* hh, int nb, int ldq, int ldh, double complex s);
#endif

#ifdef SINGLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_32_AVX512_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s);
static __forceinline void hh_trafo_complex_kernel_24_AVX512_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s);
static __forceinline void hh_trafo_complex_kernel_16_AVX512_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s);
static __forceinline void hh_trafo_complex_kernel_8_AVX512_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s);
#endif

/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine double_hh_trafo_complex_avx512_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="double_hh_trafo_complex_avx512_2hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     ! complex(kind=c_double_complex)     :: q(*)
!f>     type(c_ptr), value                   :: q
!f>     complex(kind=c_double_complex)     :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine double_hh_trafo_complex_avx512_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="double_hh_trafo_complex_avx512_2hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     ! complex(kind=c_float_complex)     :: q(*)
!f>     type(c_ptr), value                  :: q
!f>     complex(kind=c_float_complex)     :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/

#ifdef DOUBLE_PRECISION_COMPLEX
void double_hh_trafo_complex_avx512_2hv_double(double complex* q, double complex* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
#ifdef SINGLE_PRECISION_COMPLEX
void double_hh_trafo_complex_avx512_2hv_single(float complex* q, float complex* hh, int* pnb, int* pnq, int* pldq, int* pldh)
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
	for (i = 0; i < nq-12; i+=16)
	{
		hh_trafo_complex_kernel_16_AVX512_2hv_double(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 16;
	}
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	for (i = 0; i < nq-24; i+=32)
	{
		hh_trafo_complex_kernel_32_AVX512_2hv_single(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 32;

	}
#endif
	if (nq-i == 0) {
		return;
	}
#ifdef DOUBLE_PRECISION_COMPLEX
        if (nq-i == 12 )
	{
		hh_trafo_complex_kernel_12_AVX512_2hv_double(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 12;
	}
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        if (nq-i == 24 )
	{
		hh_trafo_complex_kernel_24_AVX512_2hv_single(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 24;
	}
#endif
#ifdef DOUBLE_PRECISION_COMPLEX
        if (nq-i == 8 )
	{
		hh_trafo_complex_kernel_8_AVX512_2hv_double(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 8;
	}
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        if (nq-i == 16 )
	{
		hh_trafo_complex_kernel_16_AVX512_2hv_single(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 16;
	}
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
        if (nq-i == 4 ) {

		hh_trafo_complex_kernel_4_AVX512_2hv_double(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 4;
	}
#endif

#ifdef SINGLE_PRECISION_COMPLEX
        if (nq-i == 8 ) {

		hh_trafo_complex_kernel_8_AVX512_2hv_single(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 8;
	}
#endif
#ifdef WITH_DEBUG
	if (worked_on != nq)
	{
	     printf("Error in complex AVX512 BLOCK 2 kernel \n");
	     abort();
	}
#endif
}

#ifdef DOUBLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_16_AVX512_2hv_double(double complex* q, double complex* hh, int nb, int ldq, int ldh, double complex s)
#endif
#ifdef SINGLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_32_AVX512_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s)
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
	__AVX512_DATATYPE x1, x2, x3, x4;
	__AVX512_DATATYPE y1, y2, y3, y4;
	__AVX512_DATATYPE q1, q2, q3, q4;
	__AVX512_DATATYPE h1_real, h1_imag, h2_real, h2_imag;
	__AVX512_DATATYPE tmp1, tmp2, tmp3, tmp4;
	int i=0;

#ifdef DOUBLE_PRECISION_COMPLEX
       __AVX512_DATATYPE sign = (__AVX512_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	__AVX512_DATATYPE sign = (__AVX512_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif

	x1 = _AVX512_LOAD(&q_dbl[(2*ldq)+0]);  // q1, q2, q3, q4
	x2 = _AVX512_LOAD(&q_dbl[(2*ldq)+offset]);  // q5, q6, q7, q8
	x3 = _AVX512_LOAD(&q_dbl[(2*ldq)+2*offset]); // q9, q10, q11, q12
	x4 = _AVX512_LOAD(&q_dbl[(2*ldq)+3*offset]); // q13, q14, q15, q16

	h2_real = _AVX512_SET1(hh_dbl[(ldh+1)*2]);
	h2_imag = _AVX512_SET1(hh_dbl[((ldh+1)*2)+1]);

	y1 = _AVX512_LOAD(&q_dbl[0]);
	y2 = _AVX512_LOAD(&q_dbl[offset]);
	y3 = _AVX512_LOAD(&q_dbl[2*offset]);
	y4 = _AVX512_LOAD(&q_dbl[3*offset]);

	tmp1 = _AVX512_MUL(h2_imag, x1);

	y1 = _AVX512_ADD(y1, _AVX512_FMSUBADD(h2_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	tmp2 = _AVX512_MUL(h2_imag, x2);

	y2 = _AVX512_ADD(y2, _AVX512_FMSUBADD(h2_real, x2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

	tmp3 = _AVX512_MUL(h2_imag, x3);

	y3 = _AVX512_ADD(y3, _AVX512_FMSUBADD(h2_real, x3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

	tmp4 = _AVX512_MUL(h2_imag, x4);

	y4 = _AVX512_ADD(y4, _AVX512_FMSUBADD(h2_real, x4, _AVX512_SHUFFLE(tmp4, tmp4, _SHUFFLE)));

	for (i = 2; i < nb; i++)
	{
		q1 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+0]);
		q2 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+offset]);
		q3 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+2*offset]);
		q4 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+3*offset]);

		h1_real = _AVX512_SET1(hh_dbl[(i-1)*2]);
		h1_imag = _AVX512_SET1(hh_dbl[((i-1)*2)+1]);

		tmp1 = _AVX512_MUL(h1_imag, q1);

		x1 = _AVX512_ADD(x1, _AVX512_FMSUBADD(h1_real, q1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

		tmp2 = _AVX512_MUL(h1_imag, q2);

		x2 = _AVX512_ADD(x2, _AVX512_FMSUBADD(h1_real, q2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

		tmp3 = _AVX512_MUL(h1_imag, q3);

		x3 = _AVX512_ADD(x3, _AVX512_FMSUBADD(h1_real, q3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

		tmp4 = _AVX512_MUL(h1_imag, q4);

		x4 = _AVX512_ADD(x4, _AVX512_FMSUBADD(h1_real, q4, _AVX512_SHUFFLE(tmp4, tmp4, _SHUFFLE)));

		h2_real = _AVX512_SET1(hh_dbl[(ldh+i)*2]);
		h2_imag = _AVX512_SET1(hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _AVX512_MUL(h2_imag, q1);

		y1 = _AVX512_ADD(y1, _AVX512_FMSUBADD(h2_real, q1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

		tmp2 = _AVX512_MUL(h2_imag, q2);

		y2 = _AVX512_ADD(y2, _AVX512_FMSUBADD(h2_real, q2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

		tmp3 = _AVX512_MUL(h2_imag, q3);

		y3 = _AVX512_ADD(y3, _AVX512_FMSUBADD(h2_real, q3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

		tmp4 = _AVX512_MUL(h2_imag, q4);

		y4 = _AVX512_ADD(y4, _AVX512_FMSUBADD(h2_real, q4, _AVX512_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
	}

	h1_real = _AVX512_SET1(hh_dbl[(nb-1)*2]);
	h1_imag = _AVX512_SET1(hh_dbl[((nb-1)*2)+1]);

	q1 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+0]);
	q2 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+offset]);
	q3 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+2*offset]);
	q4 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+3*offset]);

	tmp1 = _AVX512_MUL(h1_imag, q1);

	x1 = _AVX512_ADD(x1, _AVX512_FMSUBADD(h1_real, q1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	tmp2 = _AVX512_MUL(h1_imag, q2);

	x2 = _AVX512_ADD(x2, _AVX512_FMSUBADD(h1_real, q2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

	tmp3 = _AVX512_MUL(h1_imag, q3);

	x3 = _AVX512_ADD(x3, _AVX512_FMSUBADD(h1_real, q3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

	tmp4 = _AVX512_MUL(h1_imag, q4);

	x4 = _AVX512_ADD(x4, _AVX512_FMSUBADD(h1_real, q4, _AVX512_SHUFFLE(tmp4, tmp4, _SHUFFLE)));

	h1_real = _AVX512_SET1(hh_dbl[0]);
	h1_imag = _AVX512_SET1(hh_dbl[1]);

#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);
#endif
#endif
#ifdef HAVE_AVX512_XEON
#if defined(DOUBLE_PRECISION_COMPLEX) || defined(SINGLE_PRECISION_COMPLEX)
        h1_real = _AVX512_XOR(h1_real, sign);
        h1_imag = _AVX512_XOR(h1_imag, sign);
#endif
#endif
	tmp1 = _AVX512_MUL(h1_imag, x1);

	x1 = _AVX512_FMADDSUB(h1_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE));

	tmp2 = _AVX512_MUL(h1_imag, x2);

	x2 = _AVX512_FMADDSUB(h1_real, x2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE));

	tmp3 = _AVX512_MUL(h1_imag, x3);

	x3 = _AVX512_FMADDSUB(h1_real, x3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE));

	tmp4 = _AVX512_MUL(h1_imag, x4);

	x4 = _AVX512_FMADDSUB(h1_real, x4, _AVX512_SHUFFLE(tmp4, tmp4, _SHUFFLE));

	h1_real = _AVX512_SET1(hh_dbl[ldh*2]);
	h1_imag = _AVX512_SET1(hh_dbl[(ldh*2)+1]);
	h2_real = _AVX512_SET1(hh_dbl[ldh*2]);
	h2_imag = _AVX512_SET1(hh_dbl[(ldh*2)+1]);

#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
        h2_real = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h2_real, (__m512i) sign);
        h2_imag = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h2_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h2_real = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h2_real, (__m512i) sign);
        h2_imag = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h2_imag, (__m512i) sign);
#endif
#endif
#ifdef HAVE_AVX512_XEON
#if defined(DOUBLE_PRECISION_COMPLEX) || defined(SINGLE_PRECISION_COMPLEX)
        h1_real = _AVX512_XOR(h1_real, sign);
        h1_imag = _AVX512_XOR(h1_imag, sign);
        h2_real = _AVX512_XOR(h2_real, sign);
        h2_imag = _AVX512_XOR(h2_imag, sign);
#endif
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
	tmp2 = _AVX512_SET(s_dbl[1], s_dbl[0],
			     s_dbl[1], s_dbl[0],
			     s_dbl[1], s_dbl[0],
			     s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	tmp2 = (__m512) _mm512_set1_pd(*(double*)(&s_dbl[0]));
#endif
	tmp1 = _AVX512_MUL(h2_imag, tmp2);

	tmp2 = _AVX512_FMADDSUB(h2_real, tmp2, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE));

        _AVX512_MASK_STOREU(s_dbl, 0x01 + 0x02, tmp2);

	h2_real = _AVX512_SET1(s_dbl[0]);
	h2_imag = _AVX512_SET1(s_dbl[1]);

	tmp1 = _AVX512_MUL(h1_imag, y1);

	y1 = _AVX512_FMADDSUB(h1_real, y1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE));

	tmp2 = _AVX512_MUL(h1_imag, y2);

	y2 = _AVX512_FMADDSUB(h1_real, y2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE));

	tmp3 = _AVX512_MUL(h1_imag, y3);

	y3 = _AVX512_FMADDSUB(h1_real, y3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE));

	tmp4 = _AVX512_MUL(h1_imag, y4);

	y4 = _AVX512_FMADDSUB(h1_real, y4, _AVX512_SHUFFLE(tmp4, tmp4, _SHUFFLE));

	tmp1 = _AVX512_MUL(h2_imag, x1);

	y1 = _AVX512_ADD(y1, _AVX512_FMADDSUB(h2_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	tmp2 = _AVX512_MUL(h2_imag, x2);

	y2 = _AVX512_ADD(y2, _AVX512_FMADDSUB(h2_real, x2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

	tmp3 = _AVX512_MUL(h2_imag, x3);

	y3 = _AVX512_ADD(y3, _AVX512_FMADDSUB(h2_real, x3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

	tmp4 = _AVX512_MUL(h2_imag, x4);

	y4 = _AVX512_ADD(y4, _AVX512_FMADDSUB(h2_real, x4, _AVX512_SHUFFLE(tmp4, tmp4, _SHUFFLE)));

	q1 = _AVX512_LOAD(&q_dbl[0]);
	q2 = _AVX512_LOAD(&q_dbl[offset]);
	q3 = _AVX512_LOAD(&q_dbl[2*offset]);
	q4 = _AVX512_LOAD(&q_dbl[3*offset]);

	q1 = _AVX512_ADD(q1, y1);
	q2 = _AVX512_ADD(q2, y2);
	q3 = _AVX512_ADD(q3, y3);
	q4 = _AVX512_ADD(q4, y4);

	_AVX512_STORE(&q_dbl[0], q1);
	_AVX512_STORE(&q_dbl[offset], q2);
	_AVX512_STORE(&q_dbl[2*offset], q3);
	_AVX512_STORE(&q_dbl[3*offset], q4);

	h2_real = _AVX512_SET1(hh_dbl[(ldh+1)*2]);
	h2_imag = _AVX512_SET1(hh_dbl[((ldh+1)*2)+1]);

	q1 = _AVX512_LOAD(&q_dbl[(ldq*2)+0]);
	q2 = _AVX512_LOAD(&q_dbl[(ldq*2)+offset]);
	q3 = _AVX512_LOAD(&q_dbl[(ldq*2)+2*offset]);
	q4 = _AVX512_LOAD(&q_dbl[(ldq*2)+3*offset]);

	q1 = _AVX512_ADD(q1, x1);
	q2 = _AVX512_ADD(q2, x2);
	q3 = _AVX512_ADD(q3, x3);
	q4 = _AVX512_ADD(q4, x4);

	tmp1 = _AVX512_MUL(h2_imag, y1);

	q1 = _AVX512_ADD(q1, _AVX512_FMADDSUB(h2_real, y1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	tmp2 = _AVX512_MUL(h2_imag, y2);

	q2 = _AVX512_ADD(q2, _AVX512_FMADDSUB(h2_real, y2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

	tmp3 = _AVX512_MUL(h2_imag, y3);

	q3 = _AVX512_ADD(q3, _AVX512_FMADDSUB(h2_real, y3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

	tmp4 = _AVX512_MUL(h2_imag, y4);

	q4 = _AVX512_ADD(q4, _AVX512_FMADDSUB(h2_real, y4, _AVX512_SHUFFLE(tmp4, tmp4, _SHUFFLE)));

	_AVX512_STORE(&q_dbl[(ldq*2)+0], q1);
	_AVX512_STORE(&q_dbl[(ldq*2)+offset], q2);
	_AVX512_STORE(&q_dbl[(ldq*2)+2*offset], q3);
	_AVX512_STORE(&q_dbl[(ldq*2)+3*offset], q4);

	for (i = 2; i < nb; i++)
	{
		q1 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+0]);
		q2 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+offset]);
		q3 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+2*offset]);
		q4 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+3*offset]);

		h1_real = _AVX512_SET1(hh_dbl[(i-1)*2]);
		h1_imag = _AVX512_SET1(hh_dbl[((i-1)*2)+1]);

		tmp1 = _AVX512_MUL(h1_imag, x1);

		q1 = _AVX512_ADD(q1, _AVX512_FMADDSUB(h1_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

		tmp2 = _AVX512_MUL(h1_imag, x2);

		q2 = _AVX512_ADD(q2, _AVX512_FMADDSUB(h1_real, x2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

		tmp3 = _AVX512_MUL(h1_imag, x3);

		q3 = _AVX512_ADD(q3, _AVX512_FMADDSUB(h1_real, x3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

		tmp4 = _AVX512_MUL(h1_imag, x4);

		q4 = _AVX512_ADD(q4, _AVX512_FMADDSUB(h1_real, x4, _AVX512_SHUFFLE(tmp4, tmp4, _SHUFFLE)));

		h2_real = _AVX512_SET1(hh_dbl[(ldh+i)*2]);
		h2_imag = _AVX512_SET1(hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _AVX512_MUL(h2_imag, y1);

		q1 = _AVX512_ADD(q1, _AVX512_FMADDSUB(h2_real, y1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

		tmp2 = _AVX512_MUL(h2_imag, y2);

		q2 = _AVX512_ADD(q2, _AVX512_FMADDSUB(h2_real, y2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

		tmp3 = _AVX512_MUL(h2_imag, y3);

		q3 = _AVX512_ADD(q3, _AVX512_FMADDSUB(h2_real, y3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

		tmp4 = _AVX512_MUL(h2_imag, y4);

		q4 = _AVX512_ADD(q4, _AVX512_FMADDSUB(h2_real, y4, _AVX512_SHUFFLE(tmp4, tmp4, _SHUFFLE)));

		_AVX512_STORE(&q_dbl[(2*i*ldq)+0], q1);
		_AVX512_STORE(&q_dbl[(2*i*ldq)+offset], q2);
		_AVX512_STORE(&q_dbl[(2*i*ldq)+2*offset], q3);
		_AVX512_STORE(&q_dbl[(2*i*ldq)+3*offset], q4);
	}

	h1_real = _AVX512_SET1(hh_dbl[(nb-1)*2]);
	h1_imag = _AVX512_SET1(hh_dbl[((nb-1)*2)+1]);

	q1 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+0]);
	q2 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+offset]);
	q3 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+2*offset]);
	q4 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+3*offset]);

	tmp1 = _AVX512_MUL(h1_imag, x1);

	q1 = _AVX512_ADD(q1, _AVX512_FMADDSUB(h1_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	tmp2 = _AVX512_MUL(h1_imag, x2);

	q2 = _AVX512_ADD(q2, _AVX512_FMADDSUB(h1_real, x2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

	tmp3 = _AVX512_MUL(h1_imag, x3);

	q3 = _AVX512_ADD(q3, _AVX512_FMADDSUB(h1_real, x3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

	tmp4 = _AVX512_MUL(h1_imag, x4);

	q4 = _AVX512_ADD(q4, _AVX512_FMADDSUB(h1_real, x4, _AVX512_SHUFFLE(tmp4, tmp4, _SHUFFLE)));

	_AVX512_STORE(&q_dbl[(2*nb*ldq)+0], q1);
	_AVX512_STORE(&q_dbl[(2*nb*ldq)+offset], q2);
	_AVX512_STORE(&q_dbl[(2*nb*ldq)+2*offset], q3);
	_AVX512_STORE(&q_dbl[(2*nb*ldq)+3*offset], q4);
}


#ifdef DOUBLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_12_AVX512_2hv_double(double complex* q, double complex* hh, int nb, int ldq, int ldh, double complex s)
#endif
#ifdef SINGLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_24_AVX512_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s)
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
	__AVX512_DATATYPE x1, x2, x3, x4;
	__AVX512_DATATYPE y1, y2, y3, y4;
	__AVX512_DATATYPE q1, q2, q3, q4;
	__AVX512_DATATYPE h1_real, h1_imag, h2_real, h2_imag;
	__AVX512_DATATYPE tmp1, tmp2, tmp3, tmp4;
	int i=0;

#ifdef DOUBLE_PRECISION_COMPLEX
       __AVX512_DATATYPE sign = (__AVX512_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	__AVX512_DATATYPE sign = (__AVX512_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif

	x1 = _AVX512_LOAD(&q_dbl[(2*ldq)+0]);  // q1, q2, q3, q4
	x2 = _AVX512_LOAD(&q_dbl[(2*ldq)+offset]);  // q5, q6, q7, q8
	x3 = _AVX512_LOAD(&q_dbl[(2*ldq)+2*offset]); // q9, q10, q11, q12

	h2_real = _AVX512_SET1(hh_dbl[(ldh+1)*2]);
	h2_imag = _AVX512_SET1(hh_dbl[((ldh+1)*2)+1]);

	y1 = _AVX512_LOAD(&q_dbl[0]);
	y2 = _AVX512_LOAD(&q_dbl[offset]);
	y3 = _AVX512_LOAD(&q_dbl[2*offset]);

	tmp1 = _AVX512_MUL(h2_imag, x1);

	y1 = _AVX512_ADD(y1, _AVX512_FMSUBADD(h2_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	tmp2 = _AVX512_MUL(h2_imag, x2);

	y2 = _AVX512_ADD(y2, _AVX512_FMSUBADD(h2_real, x2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

	tmp3 = _AVX512_MUL(h2_imag, x3);

	y3 = _AVX512_ADD(y3, _AVX512_FMSUBADD(h2_real, x3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

	for (i = 2; i < nb; i++)
	{
		q1 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+0]);
		q2 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+offset]);
		q3 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+2*offset]);

		h1_real = _AVX512_SET1(hh_dbl[(i-1)*2]);
		h1_imag = _AVX512_SET1(hh_dbl[((i-1)*2)+1]);

		tmp1 = _AVX512_MUL(h1_imag, q1);

		x1 = _AVX512_ADD(x1, _AVX512_FMSUBADD(h1_real, q1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

		tmp2 = _AVX512_MUL(h1_imag, q2);

		x2 = _AVX512_ADD(x2, _AVX512_FMSUBADD(h1_real, q2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

		tmp3 = _AVX512_MUL(h1_imag, q3);

		x3 = _AVX512_ADD(x3, _AVX512_FMSUBADD(h1_real, q3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

		h2_real = _AVX512_SET1(hh_dbl[(ldh+i)*2]);
		h2_imag = _AVX512_SET1(hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _AVX512_MUL(h2_imag, q1);

		y1 = _AVX512_ADD(y1, _AVX512_FMSUBADD(h2_real, q1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

		tmp2 = _AVX512_MUL(h2_imag, q2);

		y2 = _AVX512_ADD(y2, _AVX512_FMSUBADD(h2_real, q2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

		tmp3 = _AVX512_MUL(h2_imag, q3);

		y3 = _AVX512_ADD(y3, _AVX512_FMSUBADD(h2_real, q3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

	}

	h1_real = _AVX512_SET1(hh_dbl[(nb-1)*2]);
	h1_imag = _AVX512_SET1(hh_dbl[((nb-1)*2)+1]);

	q1 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+0]);
	q2 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+offset]);
	q3 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+2*offset]);

	tmp1 = _AVX512_MUL(h1_imag, q1);

	x1 = _AVX512_ADD(x1, _AVX512_FMSUBADD(h1_real, q1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	tmp2 = _AVX512_MUL(h1_imag, q2);

	x2 = _AVX512_ADD(x2, _AVX512_FMSUBADD(h1_real, q2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

	tmp3 = _AVX512_MUL(h1_imag, q3);

	x3 = _AVX512_ADD(x3, _AVX512_FMSUBADD(h1_real, q3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

	h1_real = _AVX512_SET1(hh_dbl[0]);
	h1_imag = _AVX512_SET1(hh_dbl[1]);

#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);
#endif
#endif
#ifdef HAVE_AVX512_XEON
#if defined(DOUBLE_PRECISION_COMPLEX) || defined(SINGLE_PRECISION_COMPLEX)
        h1_real = _AVX512_XOR(h1_real, sign);
        h1_imag = _AVX512_XOR(h1_imag, sign);
#endif
#endif
	tmp1 = _AVX512_MUL(h1_imag, x1);

	x1 = _AVX512_FMADDSUB(h1_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE));

	tmp2 = _AVX512_MUL(h1_imag, x2);

	x2 = _AVX512_FMADDSUB(h1_real, x2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE));

	tmp3 = _AVX512_MUL(h1_imag, x3);

	x3 = _AVX512_FMADDSUB(h1_real, x3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE));

	h1_real = _AVX512_SET1(hh_dbl[ldh*2]);
	h1_imag = _AVX512_SET1(hh_dbl[(ldh*2)+1]);
	h2_real = _AVX512_SET1(hh_dbl[ldh*2]);
	h2_imag = _AVX512_SET1(hh_dbl[(ldh*2)+1]);

#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
        h2_real = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h2_real, (__m512i) sign);
        h2_imag = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h2_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h2_real = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h2_real, (__m512i) sign);
        h2_imag = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h2_imag, (__m512i) sign);
#endif
#endif
#ifdef HAVE_AVX512_XEON
#if defined(DOUBLE_PRECISION_COMPLEX) || defined(SINGLE_PRECISION_COMPLEX)
        h1_real = _AVX512_XOR(h1_real, sign);
        h1_imag = _AVX512_XOR(h1_imag, sign);
        h2_real = _AVX512_XOR(h2_real, sign);
        h2_imag = _AVX512_XOR(h2_imag, sign);
#endif
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
	tmp2 = _AVX512_SET(s_dbl[1], s_dbl[0],
			     s_dbl[1], s_dbl[0],
			     s_dbl[1], s_dbl[0],
			     s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	tmp2 = (__m512) _mm512_set1_pd(*(double*)(&s_dbl[0]));
#endif
	tmp1 = _AVX512_MUL(h2_imag, tmp2);

	tmp2 = _AVX512_FMADDSUB(h2_real, tmp2, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE));

        _AVX512_MASK_STOREU(s_dbl, 0x01 + 0x02, tmp2);

	h2_real = _AVX512_SET1(s_dbl[0]);
	h2_imag = _AVX512_SET1(s_dbl[1]);

	tmp1 = _AVX512_MUL(h1_imag, y1);

	y1 = _AVX512_FMADDSUB(h1_real, y1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE));

	tmp2 = _AVX512_MUL(h1_imag, y2);

	y2 = _AVX512_FMADDSUB(h1_real, y2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE));

	tmp3 = _AVX512_MUL(h1_imag, y3);

	y3 = _AVX512_FMADDSUB(h1_real, y3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE));

	tmp1 = _AVX512_MUL(h2_imag, x1);

	y1 = _AVX512_ADD(y1, _AVX512_FMADDSUB(h2_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	tmp2 = _AVX512_MUL(h2_imag, x2);

	y2 = _AVX512_ADD(y2, _AVX512_FMADDSUB(h2_real, x2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

	tmp3 = _AVX512_MUL(h2_imag, x3);

	y3 = _AVX512_ADD(y3, _AVX512_FMADDSUB(h2_real, x3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

	q1 = _AVX512_LOAD(&q_dbl[0]);
	q2 = _AVX512_LOAD(&q_dbl[offset]);
	q3 = _AVX512_LOAD(&q_dbl[2*offset]);

	q1 = _AVX512_ADD(q1, y1);
	q2 = _AVX512_ADD(q2, y2);
	q3 = _AVX512_ADD(q3, y3);

	_AVX512_STORE(&q_dbl[0], q1);
	_AVX512_STORE(&q_dbl[offset], q2);
	_AVX512_STORE(&q_dbl[2*offset], q3);

	h2_real = _AVX512_SET1(hh_dbl[(ldh+1)*2]);
	h2_imag = _AVX512_SET1(hh_dbl[((ldh+1)*2)+1]);

	q1 = _AVX512_LOAD(&q_dbl[(ldq*2)+0]);
	q2 = _AVX512_LOAD(&q_dbl[(ldq*2)+offset]);
	q3 = _AVX512_LOAD(&q_dbl[(ldq*2)+2*offset]);

	q1 = _AVX512_ADD(q1, x1);
	q2 = _AVX512_ADD(q2, x2);
	q3 = _AVX512_ADD(q3, x3);

	tmp1 = _AVX512_MUL(h2_imag, y1);

	q1 = _AVX512_ADD(q1, _AVX512_FMADDSUB(h2_real, y1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	tmp2 = _AVX512_MUL(h2_imag, y2);

	q2 = _AVX512_ADD(q2, _AVX512_FMADDSUB(h2_real, y2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

	tmp3 = _AVX512_MUL(h2_imag, y3);

	q3 = _AVX512_ADD(q3, _AVX512_FMADDSUB(h2_real, y3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

	_AVX512_STORE(&q_dbl[(ldq*2)+0], q1);
	_AVX512_STORE(&q_dbl[(ldq*2)+offset], q2);
	_AVX512_STORE(&q_dbl[(ldq*2)+2*offset], q3);

	for (i = 2; i < nb; i++)
	{
		q1 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+0]);
		q2 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+offset]);
		q3 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+2*offset]);

		h1_real = _AVX512_SET1(hh_dbl[(i-1)*2]);
		h1_imag = _AVX512_SET1(hh_dbl[((i-1)*2)+1]);

		tmp1 = _AVX512_MUL(h1_imag, x1);

		q1 = _AVX512_ADD(q1, _AVX512_FMADDSUB(h1_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

		tmp2 = _AVX512_MUL(h1_imag, x2);

		q2 = _AVX512_ADD(q2, _AVX512_FMADDSUB(h1_real, x2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

		tmp3 = _AVX512_MUL(h1_imag, x3);

		q3 = _AVX512_ADD(q3, _AVX512_FMADDSUB(h1_real, x3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

		h2_real = _AVX512_SET1(hh_dbl[(ldh+i)*2]);
		h2_imag = _AVX512_SET1(hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _AVX512_MUL(h2_imag, y1);

		q1 = _AVX512_ADD(q1, _AVX512_FMADDSUB(h2_real, y1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

		tmp2 = _AVX512_MUL(h2_imag, y2);

		q2 = _AVX512_ADD(q2, _AVX512_FMADDSUB(h2_real, y2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

		tmp3 = _AVX512_MUL(h2_imag, y3);

		q3 = _AVX512_ADD(q3, _AVX512_FMADDSUB(h2_real, y3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

		_AVX512_STORE(&q_dbl[(2*i*ldq)+0], q1);
		_AVX512_STORE(&q_dbl[(2*i*ldq)+offset], q2);
		_AVX512_STORE(&q_dbl[(2*i*ldq)+2*offset], q3);
	}

	h1_real = _AVX512_SET1(hh_dbl[(nb-1)*2]);
	h1_imag = _AVX512_SET1(hh_dbl[((nb-1)*2)+1]);

	q1 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+0]);
	q2 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+offset]);
	q3 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+2*offset]);

	tmp1 = _AVX512_MUL(h1_imag, x1);

	q1 = _AVX512_ADD(q1, _AVX512_FMADDSUB(h1_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	tmp2 = _AVX512_MUL(h1_imag, x2);

	q2 = _AVX512_ADD(q2, _AVX512_FMADDSUB(h1_real, x2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

	tmp3 = _AVX512_MUL(h1_imag, x3);

	q3 = _AVX512_ADD(q3, _AVX512_FMADDSUB(h1_real, x3, _AVX512_SHUFFLE(tmp3, tmp3, _SHUFFLE)));

	_AVX512_STORE(&q_dbl[(2*nb*ldq)+0], q1);
	_AVX512_STORE(&q_dbl[(2*nb*ldq)+offset], q2);
	_AVX512_STORE(&q_dbl[(2*nb*ldq)+2*offset], q3);
}


#ifdef DOUBLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_8_AVX512_2hv_double(double complex* q, double complex* hh, int nb, int ldq, int ldh, double complex s)
#endif
#ifdef SINGLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_16_AVX512_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s)
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

	__AVX512_DATATYPE x1, x2;
	__AVX512_DATATYPE y1, y2;
	__AVX512_DATATYPE q1, q2;
	__AVX512_DATATYPE h1_real, h1_imag, h2_real, h2_imag;
	__AVX512_DATATYPE tmp1, tmp2;
	int i=0;

#ifdef DOUBLE_PRECISION_COMPLEX
       __AVX512_DATATYPE sign = (__AVX512_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	__AVX512_DATATYPE sign = (__AVX512_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif

        x1 = _AVX512_LOAD(&q_dbl[(2*ldq)+0]);
	x2 = _AVX512_LOAD(&q_dbl[(2*ldq)+offset]);

	h2_real = _AVX512_SET1(hh_dbl[(ldh+1)*2]);
	h2_imag = _AVX512_SET1(hh_dbl[((ldh+1)*2)+1]);

	y1 = _AVX512_LOAD(&q_dbl[0]);
	y2 = _AVX512_LOAD(&q_dbl[offset]);

	tmp1 = _AVX512_MUL(h2_imag, x1);

	y1 = _AVX512_ADD(y1, _AVX512_FMSUBADD(h2_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	tmp2 = _AVX512_MUL(h2_imag, x2);

	y2 = _AVX512_ADD(y2, _AVX512_FMSUBADD(h2_real, x2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

	for (i = 2; i < nb; i++)
	{
		q1 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+0]);
		q2 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+offset]);

		h1_real = _AVX512_SET1(hh_dbl[(i-1)*2]);
		h1_imag = _AVX512_SET1(hh_dbl[((i-1)*2)+1]);

		tmp1 = _AVX512_MUL(h1_imag, q1);

		x1 = _AVX512_ADD(x1, _AVX512_FMSUBADD(h1_real, q1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

		tmp2 = _AVX512_MUL(h1_imag, q2);

		x2 = _AVX512_ADD(x2, _AVX512_FMSUBADD(h1_real, q2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

		h2_real = _AVX512_SET1(hh_dbl[(ldh+i)*2]);
		h2_imag = _AVX512_SET1(hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _AVX512_MUL(h2_imag, q1);

		y1 = _AVX512_ADD(y1, _AVX512_FMSUBADD(h2_real, q1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

		tmp2 = _AVX512_MUL(h2_imag, q2);

		y2 = _AVX512_ADD(y2, _AVX512_FMSUBADD(h2_real, q2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
	}

	h1_real = _AVX512_SET1(hh_dbl[(nb-1)*2]);
	h1_imag = _AVX512_SET1(hh_dbl[((nb-1)*2)+1]);

	q1 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+0]);
	q2 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+offset]);

	tmp1 = _AVX512_MUL(h1_imag, q1);

	x1 = _AVX512_ADD(x1, _AVX512_FMSUBADD(h1_real, q1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	tmp2 = _AVX512_MUL(h1_imag, q2);

	x2 = _AVX512_ADD(x2, _AVX512_FMSUBADD(h1_real, q2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

	h1_real = _AVX512_SET1(hh_dbl[0]);
	h1_imag = _AVX512_SET1(hh_dbl[1]);

#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);
#endif
#endif
#ifdef HAVE_AVX512_XEON
#if defined(DOUBLE_PRECISION_COMPLEX) || defined(SINGLE_PRECISION_COMPLEX)
        h1_real = _AVX512_XOR(h1_real, sign);
        h1_imag = _AVX512_XOR(h1_imag, sign);
#endif
#endif

	tmp1 = _AVX512_MUL(h1_imag, x1);

	x1 = _AVX512_FMADDSUB(h1_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE));

	tmp2 = _AVX512_MUL(h1_imag, x2);

	x2 = _AVX512_FMADDSUB(h1_real, x2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE));

	h1_real = _AVX512_SET1(hh_dbl[ldh*2]);
	h1_imag = _AVX512_SET1(hh_dbl[(ldh*2)+1]);
	h2_real = _AVX512_SET1(hh_dbl[ldh*2]);
	h2_imag = _AVX512_SET1(hh_dbl[(ldh*2)+1]);

#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
        h2_real = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h2_real, (__m512i) sign);
        h2_imag = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h2_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h2_real = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h2_real, (__m512i) sign);
        h2_imag = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h2_imag, (__m512i) sign);
#endif
#endif
#ifdef HAVE_AVX512_XEON
#if defined(DOUBLE_PRECISION_COMPLEX) || defined(SINGLE_PRECISION_COMPLEX)
        h1_real = _AVX512_XOR(h1_real, sign);
        h1_imag = _AVX512_XOR(h1_imag, sign);
        h2_real = _AVX512_XOR(h2_real, sign);
        h2_imag = _AVX512_XOR(h2_imag, sign);
#endif
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
	tmp2 = _AVX512_SET(s_dbl[1], s_dbl[0],
			     s_dbl[1], s_dbl[0],
			     s_dbl[1], s_dbl[0],
			     s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	tmp2 = (__m512) _mm512_set1_pd(*(double*)(&s_dbl[0]));
#endif

	tmp1 = _AVX512_MUL(h2_imag, tmp2);

	tmp2 = _AVX512_FMADDSUB(h2_real, tmp2, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE));

        _AVX512_MASK_STOREU(s_dbl, 0x01 + 0x02, tmp2);

	h2_real = _AVX512_SET1(s_dbl[0]);
	h2_imag = _AVX512_SET1(s_dbl[1]);

	tmp1 = _AVX512_MUL(h1_imag, y1);

	y1 = _AVX512_FMADDSUB(h1_real, y1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE));

	tmp2 = _AVX512_MUL(h1_imag, y2);

	y2 = _AVX512_FMADDSUB(h1_real, y2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE));

	tmp1 = _AVX512_MUL(h2_imag, x1);

	y1 = _AVX512_ADD(y1, _AVX512_FMADDSUB(h2_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	tmp2 = _AVX512_MUL(h2_imag, x2);

	y2 = _AVX512_ADD(y2, _AVX512_FMADDSUB(h2_real, x2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

	q1 = _AVX512_LOAD(&q_dbl[0]);
	q2 = _AVX512_LOAD(&q_dbl[offset]);

	q1 = _AVX512_ADD(q1, y1);
	q2 = _AVX512_ADD(q2, y2);

	_AVX512_STORE(&q_dbl[0], q1);
	_AVX512_STORE(&q_dbl[offset], q2);

	h2_real = _AVX512_SET1(hh_dbl[(ldh+1)*2]);
	h2_imag = _AVX512_SET1(hh_dbl[((ldh+1)*2)+1]);

	q1 = _AVX512_LOAD(&q_dbl[(ldq*2)+0]);
	q2 = _AVX512_LOAD(&q_dbl[(ldq*2)+offset]);

	q1 = _AVX512_ADD(q1, x1);
	q2 = _AVX512_ADD(q2, x2);

	tmp1 = _AVX512_MUL(h2_imag, y1);

	q1 = _AVX512_ADD(q1, _AVX512_FMADDSUB(h2_real, y1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	tmp2 = _AVX512_MUL(h2_imag, y2);

	q2 = _AVX512_ADD(q2, _AVX512_FMADDSUB(h2_real, y2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

	_AVX512_STORE(&q_dbl[(ldq*2)+0], q1);
	_AVX512_STORE(&q_dbl[(ldq*2)+offset], q2);

	for (i = 2; i < nb; i++)
	{
		q1 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+0]);
		q2 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+offset]);

		h1_real = _AVX512_SET1(hh_dbl[(i-1)*2]);
		h1_imag = _AVX512_SET1(hh_dbl[((i-1)*2)+1]);

		tmp1 = _AVX512_MUL(h1_imag, x1);

		q1 = _AVX512_ADD(q1, _AVX512_FMADDSUB(h1_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

		tmp2 = _AVX512_MUL(h1_imag, x2);

		q2 = _AVX512_ADD(q2, _AVX512_FMADDSUB(h1_real, x2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

		h2_real = _AVX512_SET1(hh_dbl[(ldh+i)*2]);
		h2_imag = _AVX512_SET1(hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _AVX512_MUL(h2_imag, y1);

		q1 = _AVX512_ADD(q1, _AVX512_FMADDSUB(h2_real, y1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

		tmp2 = _AVX512_MUL(h2_imag, y2);

		q2 = _AVX512_ADD(q2, _AVX512_FMADDSUB(h2_real, y2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

		_AVX512_STORE(&q_dbl[(2*i*ldq)+0], q1);
		_AVX512_STORE(&q_dbl[(2*i*ldq)+offset], q2);
	}
	h1_real = _AVX512_SET1(hh_dbl[(nb-1)*2]);
	h1_imag = _AVX512_SET1(hh_dbl[((nb-1)*2)+1]);

	q1 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+0]);
	q2 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+offset]);

	tmp1 = _AVX512_MUL(h1_imag, x1);

	q1 = _AVX512_ADD(q1, _AVX512_FMADDSUB(h1_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	tmp2 = _AVX512_MUL(h1_imag, x2);

	q2 = _AVX512_ADD(q2, _AVX512_FMADDSUB(h1_real, x2, _AVX512_SHUFFLE(tmp2, tmp2, _SHUFFLE)));

	_AVX512_STORE(&q_dbl[(2*nb*ldq)+0], q1);
	_AVX512_STORE(&q_dbl[(2*nb*ldq)+offset], q2);
}

#ifdef DOUBLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_4_AVX512_2hv_double(double complex* q, double complex* hh, int nb, int ldq, int ldh, double complex s)
#endif
#ifdef SINGLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_8_AVX512_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s)
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

	__AVX512_DATATYPE x1, x2;
	__AVX512_DATATYPE y1, y2;
	__AVX512_DATATYPE q1, q2;
	__AVX512_DATATYPE h1_real, h1_imag, h2_real, h2_imag;
	__AVX512_DATATYPE tmp1, tmp2;
	int i=0;

#ifdef DOUBLE_PRECISION_COMPLEX
       __AVX512_DATATYPE sign = (__AVX512_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	__AVX512_DATATYPE sign = (__AVX512_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif

        x1 = _AVX512_LOAD(&q_dbl[(2*ldq)+0]);

	h2_real = _AVX512_SET1(hh_dbl[(ldh+1)*2]);
	h2_imag = _AVX512_SET1(hh_dbl[((ldh+1)*2)+1]);

	y1 = _AVX512_LOAD(&q_dbl[0]);

	tmp1 = _AVX512_MUL(h2_imag, x1);

	y1 = _AVX512_ADD(y1, _AVX512_FMSUBADD(h2_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	for (i = 2; i < nb; i++)
	{
		q1 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+0]);
		q2 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+offset]);

		h1_real = _AVX512_SET1(hh_dbl[(i-1)*2]);
		h1_imag = _AVX512_SET1(hh_dbl[((i-1)*2)+1]);

		tmp1 = _AVX512_MUL(h1_imag, q1);

		x1 = _AVX512_ADD(x1, _AVX512_FMSUBADD(h1_real, q1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

		h2_real = _AVX512_SET1(hh_dbl[(ldh+i)*2]);
		h2_imag = _AVX512_SET1(hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _AVX512_MUL(h2_imag, q1);

		y1 = _AVX512_ADD(y1, _AVX512_FMSUBADD(h2_real, q1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	}

	h1_real = _AVX512_SET1(hh_dbl[(nb-1)*2]);
	h1_imag = _AVX512_SET1(hh_dbl[((nb-1)*2)+1]);

	q1 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+0]);

	tmp1 = _AVX512_MUL(h1_imag, q1);

	x1 = _AVX512_ADD(x1, _AVX512_FMSUBADD(h1_real, q1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	h1_real = _AVX512_SET1(hh_dbl[0]);
	h1_imag = _AVX512_SET1(hh_dbl[1]);

#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);
#endif
#endif
#ifdef HAVE_AVX512_XEON
#if defined(DOUBLE_PRECISION_COMPLEX) || defined(SINGLE_PRECISION_COMPLEX)
        h1_real = _AVX512_XOR(h1_real, sign);
        h1_imag = _AVX512_XOR(h1_imag, sign);
#endif
#endif

	tmp1 = _AVX512_MUL(h1_imag, x1);

	x1 = _AVX512_FMADDSUB(h1_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE));

	h1_real = _AVX512_SET1(hh_dbl[ldh*2]);
	h1_imag = _AVX512_SET1(hh_dbl[(ldh*2)+1]);
	h2_real = _AVX512_SET1(hh_dbl[ldh*2]);
	h2_imag = _AVX512_SET1(hh_dbl[(ldh*2)+1]);

#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
        h2_real = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h2_real, (__m512i) sign);
        h2_imag = (__AVX512_DATATYPE) _mm512_xor_epi64((__m512i) h2_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h2_real = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h2_real, (__m512i) sign);
        h2_imag = (__AVX512_DATATYPE) _mm512_xor_epi32((__m512i) h2_imag, (__m512i) sign);
#endif
#endif
#ifdef HAVE_AVX512_XEON
#if defined(DOUBLE_PRECISION_COMPLEX) || defined(SINGLE_PRECISION_COMPLEX)
        h1_real = _AVX512_XOR(h1_real, sign);
        h1_imag = _AVX512_XOR(h1_imag, sign);
        h2_real = _AVX512_XOR(h2_real, sign);
        h2_imag = _AVX512_XOR(h2_imag, sign);
#endif
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
	tmp2 = _AVX512_SET(s_dbl[1], s_dbl[0],
			     s_dbl[1], s_dbl[0],
			     s_dbl[1], s_dbl[0],
			     s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	tmp2 = (__m512) _mm512_set1_pd(*(double*)(&s_dbl[0]));
#endif

	tmp1 = _AVX512_MUL(h2_imag, tmp2);

	tmp2 = _AVX512_FMADDSUB(h2_real, tmp2, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE));

        _AVX512_MASK_STOREU(s_dbl, 0x01 + 0x02, tmp2);

	h2_real = _AVX512_SET1(s_dbl[0]);
	h2_imag = _AVX512_SET1(s_dbl[1]);

	tmp1 = _AVX512_MUL(h1_imag, y1);

	y1 = _AVX512_FMADDSUB(h1_real, y1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE));

	tmp1 = _AVX512_MUL(h2_imag, x1);

	y1 = _AVX512_ADD(y1, _AVX512_FMADDSUB(h2_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	q1 = _AVX512_LOAD(&q_dbl[0]);

	q1 = _AVX512_ADD(q1, y1);

	_AVX512_STORE(&q_dbl[0], q1);

	h2_real = _AVX512_SET1(hh_dbl[(ldh+1)*2]);
	h2_imag = _AVX512_SET1(hh_dbl[((ldh+1)*2)+1]);

	q1 = _AVX512_LOAD(&q_dbl[(ldq*2)+0]);

	q1 = _AVX512_ADD(q1, x1);

	tmp1 = _AVX512_MUL(h2_imag, y1);

	q1 = _AVX512_ADD(q1, _AVX512_FMADDSUB(h2_real, y1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	_AVX512_STORE(&q_dbl[(ldq*2)+0], q1);

	for (i = 2; i < nb; i++)
	{
		q1 = _AVX512_LOAD(&q_dbl[(2*i*ldq)+0]);

		h1_real = _AVX512_SET1(hh_dbl[(i-1)*2]);
		h1_imag = _AVX512_SET1(hh_dbl[((i-1)*2)+1]);

		tmp1 = _AVX512_MUL(h1_imag, x1);

		q1 = _AVX512_ADD(q1, _AVX512_FMADDSUB(h1_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

		h2_real = _AVX512_SET1(hh_dbl[(ldh+i)*2]);
		h2_imag = _AVX512_SET1(hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _AVX512_MUL(h2_imag, y1);

		q1 = _AVX512_ADD(q1, _AVX512_FMADDSUB(h2_real, y1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

		_AVX512_STORE(&q_dbl[(2*i*ldq)+0], q1);
	}
	h1_real = _AVX512_SET1(hh_dbl[(nb-1)*2]);
	h1_imag = _AVX512_SET1(hh_dbl[((nb-1)*2)+1]);

	q1 = _AVX512_LOAD(&q_dbl[(2*nb*ldq)+0]);

	tmp1 = _AVX512_MUL(h1_imag, x1);

	q1 = _AVX512_ADD(q1, _AVX512_FMADDSUB(h1_real, x1, _AVX512_SHUFFLE(tmp1, tmp1, _SHUFFLE)));

	_AVX512_STORE(&q_dbl[(2*nb*ldq)+0], q1);
}

