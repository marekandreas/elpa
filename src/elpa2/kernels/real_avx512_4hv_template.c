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
#include <stdio.h>
#include <stdlib.h>

#define __forceinline __attribute__((always_inline)) static

#ifdef DOUBLE_PRECISION_REAL
#define offset 8
#define __AVX512_DATATYPE __m512d
#define _AVX512_LOAD _mm512_load_pd
#define _AVX512_STORE _mm512_store_pd
#define _AVX512_SET1 _mm512_set1_pd
#define _AVX512_MUL _mm512_mul_pd
#define _AVX512_SUB _mm512_sub_pd

#ifdef HAVE_AVX512
#define __ELPA_USE_FMA__
#define _mm512_FMA_pd(a,b,c) _mm512_fmadd_pd(a,b,c)
#define _mm512_NFMA_pd(a,b,c) _mm512_fnmadd_pd(a,b,c)
#define _mm512_FMSUB_pd(a,b,c) _mm512_fmsub_pd(a,b,c)
#endif

#define _AVX512_FMA _mm512_FMA_pd
#define _AVX512_NFMA _mm512_NFMA_pd
#define _AVX512_FMSUB _mm512_FMSUB_pd
#endif /* DOUBLE_PRECISION_REAL */

#ifdef SINGLE_PRECISION_REAL
#define offset 16
#define __AVX512_DATATYPE __m512
#define _AVX512_LOAD _mm512_load_ps
#define _AVX512_STORE _mm512_store_ps
#define _AVX512_SET1 _mm512_set1_ps
#define _AVX512_MUL _mm512_mul_ps
#define _AVX512_SUB _mm512_sub_ps

#ifdef HAVE_AVX512
#define __ELPA_USE_FMA__
#define _mm512_FMA_ps(a,b,c) _mm512_fmadd_ps(a,b,c)
#define _mm512_NFMA_ps(a,b,c) _mm512_fnmadd_ps(a,b,c)
#define _mm512_FMSUB_ps(a,b,c) _mm512_fmsub_ps(a,b,c)
#endif

#define _AVX512_FMA _mm512_FMA_ps
#define _AVX512_NFMA _mm512_NFMA_ps
#define _AVX512_FMSUB _mm512_FMSUB_ps
#endif /* SINGLE_PRECISION_REAL */

#ifdef DOUBLE_PRECISION_REAL
//Forward declaration
__forceinline void hh_trafo_kernel_8_AVX512_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4);
__forceinline void hh_trafo_kernel_16_AVX512_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4);
__forceinline void hh_trafo_kernel_24_AVX512_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4);
__forceinline void hh_trafo_kernel_32_AVX512_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4);


void quad_hh_trafo_real_avx512_4hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif
#ifdef SINGLE_PRECISION_REAL
//Forward declaration
__forceinline void hh_trafo_kernel_16_AVX512_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);
__forceinline void hh_trafo_kernel_32_AVX512_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);
__forceinline void hh_trafo_kernel_48_AVX512_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);
__forceinline void hh_trafo_kernel_64_AVX512_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);

void quad_hh_trafo_real_avx_avx2_4hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif


#ifdef DOUBLE_PRECISION_REAL
/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine quad_hh_trafo_real_avx512_4hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="quad_hh_trafo_real_avx512_4hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_double)     :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
#endif
/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine quad_hh_trafo_real_avx512_4hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="quad_hh_trafo_real_avx512_4hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)      :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
#ifdef SINGLE_PRECISION_REAL

#endif

#ifdef DOUBLE_PRECISION_REAL
void quad_hh_trafo_real_avx512_4hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
#ifdef SINGLE_PRECISION_REAL
void quad_hh_trafo_real_avx512_4hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;
	int worked_on;

	worked_on = 0;

	// calculating scalar products to compute
	// 4 householder vectors simultaneously
#ifdef DOUBLE_PRECISION_REAL
	double s_1_2 = hh[(ldh)+1];
	double s_1_3 = hh[(ldh*2)+2];
	double s_2_3 = hh[(ldh*2)+1];
	double s_1_4 = hh[(ldh*3)+3];
	double s_2_4 = hh[(ldh*3)+2];
	double s_3_4 = hh[(ldh*3)+1];
#endif
#ifdef SINGLE_PRECISION_REAL
	float s_1_2 = hh[(ldh)+1];
	float s_1_3 = hh[(ldh*2)+2];
	float s_2_3 = hh[(ldh*2)+1];
	float s_1_4 = hh[(ldh*3)+3];
	float s_2_4 = hh[(ldh*3)+2];
	float s_3_4 = hh[(ldh*3)+1];
#endif


	// calculate scalar product of first and fourth householder Vector
	// loop counter = 2
	s_1_2 += hh[2-1] * hh[(2+ldh)];
	s_2_3 += hh[(ldh)+2-1] * hh[2+(ldh*2)];
	s_3_4 += hh[(ldh*2)+2-1] * hh[2+(ldh*3)];

	// loop counter = 3
	s_1_2 += hh[3-1] * hh[(3+ldh)];
	s_2_3 += hh[(ldh)+3-1] * hh[3+(ldh*2)];
	s_3_4 += hh[(ldh*2)+3-1] * hh[3+(ldh*3)];

	s_1_3 += hh[3-2] * hh[3+(ldh*2)];
	s_2_4 += hh[(ldh*1)+3-2] * hh[3+(ldh*3)];

	#pragma ivdep
	for (i = 4; i < nb; i++)
	{
		s_1_2 += hh[i-1] * hh[(i+ldh)];
		s_2_3 += hh[(ldh)+i-1] * hh[i+(ldh*2)];
		s_3_4 += hh[(ldh*2)+i-1] * hh[i+(ldh*3)];

		s_1_3 += hh[i-2] * hh[i+(ldh*2)];
		s_2_4 += hh[(ldh*1)+i-2] * hh[i+(ldh*3)];

		s_1_4 += hh[i-3] * hh[i+(ldh*3)];
	}

	// Production level kernel calls with padding
#ifdef DOUBLE_PRECISION_REAL
	for (i = 0; i < nq-24; i+=32)
	{
		hh_trafo_kernel_32_AVX512_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
		worked_on += 32;
	}
#endif
#ifdef SINGLE_PRECISION_REAL
	for (i = 0; i < nq-48; i+=64)
	{
		hh_trafo_kernel_64_AVX512_4hv_single(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
		worked_on += 64;
	}
#endif
	if (nq == i)
	{
		return;
	}
#ifdef DOUBLE_PRECISION_REAL
	if (nq-i == 24)
	{
		hh_trafo_kernel_24_AVX512_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
		worked_on += 24;
	}
#endif

#ifdef SINGLE_PRECISION_REAL
	if (nq-i == 48)
	{
		hh_trafo_kernel_48_AVX512_4hv_single(&q[i], hh, nb, ldq, ldh,  s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
		worked_on += 48;
	}
#endif

#ifdef DOUBLE_PRECISION_REAL
        if (nq-i == 16)
	{
		hh_trafo_kernel_16_AVX512_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
		worked_on += 16;
	}
#endif

#ifdef SINGLE_PRECISION_REAL
        if (nq-i == 32)
	{
		hh_trafo_kernel_32_AVX512_4hv_single(&q[i], hh, nb, ldq, ldh,  s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
		worked_on += 32;
        }
#endif

#ifdef DOUBLE_PRECISION_REAL
        if (nq-i == 8)
	{
		hh_trafo_kernel_8_AVX512_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
		worked_on += 8;
	}
#endif

#ifdef SINGLE_PRECISION_REAL
        if (nq-i == 16)
	{
		hh_trafo_kernel_16_AVX512_4hv_single(&q[i], hh, nb, ldq, ldh,  s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
		worked_on += 16;
        }
#endif

        if (worked_on != nq)
	{
		 printf("Error in AVX512 real BLOCK 2 kernel \n");
		 abort();
	}
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
 * vectors + a rank 1 update is performed
 */

#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_32_AVX512_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_64_AVX512_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
#endif

{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__AVX512_DATATYPE a1_1 = _AVX512_LOAD(&q[ldq*3]);
	__AVX512_DATATYPE a2_1 = _AVX512_LOAD(&q[ldq*2]);
	__AVX512_DATATYPE a3_1 = _AVX512_LOAD(&q[ldq]);
	__AVX512_DATATYPE a4_1 = _AVX512_LOAD(&q[0]);

	__AVX512_DATATYPE a1_2 = _AVX512_LOAD(&q[(ldq*3)+offset]);
	__AVX512_DATATYPE a2_2 = _AVX512_LOAD(&q[(ldq*2)+offset]);
	__AVX512_DATATYPE a3_2 = _AVX512_LOAD(&q[ldq+offset]);
	__AVX512_DATATYPE a4_2 = _AVX512_LOAD(&q[0+offset]);

	__AVX512_DATATYPE a1_3 = _AVX512_LOAD(&q[(ldq*3)+2*offset]);
	__AVX512_DATATYPE a2_3 = _AVX512_LOAD(&q[(ldq*2)+2*offset]);
	__AVX512_DATATYPE a3_3 = _AVX512_LOAD(&q[ldq+2*offset]);
	__AVX512_DATATYPE a4_3 = _AVX512_LOAD(&q[0+2*offset]);

	__AVX512_DATATYPE a1_4 = _AVX512_LOAD(&q[(ldq*3)+3*offset]);
	__AVX512_DATATYPE a2_4 = _AVX512_LOAD(&q[(ldq*2)+3*offset]);
	__AVX512_DATATYPE a3_4 = _AVX512_LOAD(&q[ldq+3*offset]);
	__AVX512_DATATYPE a4_4 = _AVX512_LOAD(&q[0+3*offset]);


	__AVX512_DATATYPE h_2_1 = _AVX512_SET1(hh[ldh+1]);
	__AVX512_DATATYPE h_3_2 = _AVX512_SET1(hh[(ldh*2)+1]);
	__AVX512_DATATYPE h_3_1 = _AVX512_SET1(hh[(ldh*2)+2]);
	__AVX512_DATATYPE h_4_3 = _AVX512_SET1(hh[(ldh*3)+1]);
	__AVX512_DATATYPE h_4_2 = _AVX512_SET1(hh[(ldh*3)+2]);
	__AVX512_DATATYPE h_4_1 = _AVX512_SET1(hh[(ldh*3)+3]);

	__AVX512_DATATYPE w1 = _AVX512_FMA(a3_1, h_4_3, a4_1);
	w1 = _AVX512_FMA(a2_1, h_4_2, w1);
	w1 = _AVX512_FMA(a1_1, h_4_1, w1);
	__AVX512_DATATYPE z1 = _AVX512_FMA(a2_1, h_3_2, a3_1);
	z1 = _AVX512_FMA(a1_1, h_3_1, z1);
	__AVX512_DATATYPE y1 = _AVX512_FMA(a1_1, h_2_1, a2_1);
	__AVX512_DATATYPE x1 = a1_1;

	__AVX512_DATATYPE w2 = _AVX512_FMA(a3_2, h_4_3, a4_2);
	w2 = _AVX512_FMA(a2_2, h_4_2, w2);
	w2 = _AVX512_FMA(a1_2, h_4_1, w2);
	__AVX512_DATATYPE z2 = _AVX512_FMA(a2_2, h_3_2, a3_2);
	z2 = _AVX512_FMA(a1_2, h_3_1, z2);
	__AVX512_DATATYPE y2 = _AVX512_FMA(a1_2, h_2_1, a2_2);
	__AVX512_DATATYPE x2 = a1_2;

	__AVX512_DATATYPE w3 = _AVX512_FMA(a3_3, h_4_3, a4_3);
	w3 = _AVX512_FMA(a2_3, h_4_2, w3);
	w3 = _AVX512_FMA(a1_3, h_4_1, w3);
	__AVX512_DATATYPE z3 = _AVX512_FMA(a2_3, h_3_2, a3_3);
	z3 = _AVX512_FMA(a1_3, h_3_1, z3);
	__AVX512_DATATYPE y3 = _AVX512_FMA(a1_3, h_2_1, a2_3);
	__AVX512_DATATYPE x3 = a1_3;

	__AVX512_DATATYPE w4 = _AVX512_FMA(a3_4, h_4_3, a4_4);
	w4 = _AVX512_FMA(a2_4, h_4_2, w4);
	w4 = _AVX512_FMA(a1_4, h_4_1, w4);
	__AVX512_DATATYPE z4 = _AVX512_FMA(a2_4, h_3_2, a3_4);
	z4 = _AVX512_FMA(a1_4, h_3_1, z4);
	__AVX512_DATATYPE y4 = _AVX512_FMA(a1_4, h_2_1, a2_4);
	__AVX512_DATATYPE x4 = a1_4;


	__AVX512_DATATYPE q1;
	__AVX512_DATATYPE q2;
	__AVX512_DATATYPE q3;
	__AVX512_DATATYPE q4;

	__AVX512_DATATYPE h1;
	__AVX512_DATATYPE h2;
	__AVX512_DATATYPE h3;
	__AVX512_DATATYPE h4;

	for(i = 4; i < nb; i++)
	{
		h1 = _AVX512_SET1(hh[i-3]);
		h2 = _AVX512_SET1(hh[ldh+i-2]);
		h3 = _AVX512_SET1(hh[(ldh*2)+i-1]);
		h4 = _AVX512_SET1(hh[(ldh*3)+i]);

		q1 = _AVX512_LOAD(&q[i*ldq]);
		q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);
		q3 = _AVX512_LOAD(&q[(i*ldq)+2*offset]);
		q4 = _AVX512_LOAD(&q[(i*ldq)+3*offset]);

		x1 = _AVX512_FMA(q1, h1, x1);
		y1 = _AVX512_FMA(q1, h2, y1);
		z1 = _AVX512_FMA(q1, h3, z1);
		w1 = _AVX512_FMA(q1, h4, w1);

		x2 = _AVX512_FMA(q2, h1, x2);
		y2 = _AVX512_FMA(q2, h2, y2);
		z2 = _AVX512_FMA(q2, h3, z2);
		w2 = _AVX512_FMA(q2, h4, w2);

		x3 = _AVX512_FMA(q3, h1, x3);
		y3 = _AVX512_FMA(q3, h2, y3);
		z3 = _AVX512_FMA(q3, h3, z3);
		w3 = _AVX512_FMA(q3, h4, w3);

		x4 = _AVX512_FMA(q4, h1, x4);
		y4 = _AVX512_FMA(q4, h2, y4);
		z4 = _AVX512_FMA(q4, h3, z4);
		w4 = _AVX512_FMA(q4, h4, w4);

	}

	h1 = _AVX512_SET1(hh[nb-3]);
	h2 = _AVX512_SET1(hh[ldh+nb-2]);
	h3 = _AVX512_SET1(hh[(ldh*2)+nb-1]);

	q1 = _AVX512_LOAD(&q[nb*ldq]);
	q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);
	q3 = _AVX512_LOAD(&q[(nb*ldq)+2*offset]);
	q4 = _AVX512_LOAD(&q[(nb*ldq)+3*offset]);

	x1 = _AVX512_FMA(q1, h1, x1);
	y1 = _AVX512_FMA(q1, h2, y1);
	z1 = _AVX512_FMA(q1, h3, z1);

	x2 = _AVX512_FMA(q2, h1, x2);
	y2 = _AVX512_FMA(q2, h2, y2);
	z2 = _AVX512_FMA(q2, h3, z2);

	x3 = _AVX512_FMA(q3, h1, x3);
	y3 = _AVX512_FMA(q3, h2, y3);
	z3 = _AVX512_FMA(q3, h3, z3);

	x4 = _AVX512_FMA(q4, h1, x4);
	y4 = _AVX512_FMA(q4, h2, y4);
	z4 = _AVX512_FMA(q4, h3, z4);


	h1 = _AVX512_SET1(hh[nb-2]);
	h2 = _AVX512_SET1(hh[(ldh*1)+nb-1]);

	q1 = _AVX512_LOAD(&q[(nb+1)*ldq]);
	q2 = _AVX512_LOAD(&q[((nb+1)*ldq)+offset]);
	q3 = _AVX512_LOAD(&q[((nb+1)*ldq)+2*offset]);
	q4 = _AVX512_LOAD(&q[((nb+1)*ldq)+3*offset]);

	x1 = _AVX512_FMA(q1, h1, x1);
	y1 = _AVX512_FMA(q1, h2, y1);
	x2 = _AVX512_FMA(q2, h1, x2);
	y2 = _AVX512_FMA(q2, h2, y2);
	x3 = _AVX512_FMA(q3, h1, x3);
	y3 = _AVX512_FMA(q3, h2, y3);
	x4 = _AVX512_FMA(q4, h1, x4);
	y4 = _AVX512_FMA(q4, h2, y4);

	h1 = _AVX512_SET1(hh[nb-1]);

	q1 = _AVX512_LOAD(&q[(nb+2)*ldq]);
	q2 = _AVX512_LOAD(&q[((nb+2)*ldq)+offset]);
	q3 = _AVX512_LOAD(&q[((nb+2)*ldq)+2*offset]);
	q4 = _AVX512_LOAD(&q[((nb+2)*ldq)+3*offset]);

	x1 = _AVX512_FMA(q1, h1, x1);
	x2 = _AVX512_FMA(q2, h1, x2);
	x3 = _AVX512_FMA(q3, h1, x3);
	x4 = _AVX512_FMA(q4, h1, x4);

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [8 x nb+3]
	/////////////////////////////////////////////////////

	__AVX512_DATATYPE tau1 = _AVX512_SET1(hh[0]);
	__AVX512_DATATYPE tau2 = _AVX512_SET1(hh[ldh]);
	__AVX512_DATATYPE tau3 = _AVX512_SET1(hh[ldh*2]);
	__AVX512_DATATYPE tau4 = _AVX512_SET1(hh[ldh*3]);

	__AVX512_DATATYPE vs_1_2 = _AVX512_SET1(s_1_2);
	__AVX512_DATATYPE vs_1_3 = _AVX512_SET1(s_1_3);
	__AVX512_DATATYPE vs_2_3 = _AVX512_SET1(s_2_3);
	__AVX512_DATATYPE vs_1_4 = _AVX512_SET1(s_1_4);
	__AVX512_DATATYPE vs_2_4 = _AVX512_SET1(s_2_4);
	__AVX512_DATATYPE vs_3_4 = _AVX512_SET1(s_3_4);

	h1 = tau1;
	x1 = _AVX512_MUL(x1, h1);
	x2 = _AVX512_MUL(x2, h1);
	x3 = _AVX512_MUL(x3, h1);
	x4 = _AVX512_MUL(x4, h1);

	h1 = tau2;
	h2 = _AVX512_MUL(h1, vs_1_2);

	y1 = _AVX512_FMSUB(y1, h1, _AVX512_MUL(x1,h2));
	y2 = _AVX512_FMSUB(y2, h1, _AVX512_MUL(x2,h2));
	y3 = _AVX512_FMSUB(y3, h1, _AVX512_MUL(x3,h2));
	y4 = _AVX512_FMSUB(y4, h1, _AVX512_MUL(x4,h2));

	h1 = tau3;
	h2 = _AVX512_MUL(h1, vs_1_3);
	h3 = _AVX512_MUL(h1, vs_2_3);

	z1 = _AVX512_FMSUB(z1, h1, _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2)));
	z2 = _AVX512_FMSUB(z2, h1, _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2)));
	z3 = _AVX512_FMSUB(z3, h1, _AVX512_FMA(y3, h3, _AVX512_MUL(x3,h2)));
	z4 = _AVX512_FMSUB(z4, h1, _AVX512_FMA(y4, h3, _AVX512_MUL(x4,h2)));

	h1 = tau4;
	h2 = _AVX512_MUL(h1, vs_1_4);
	h3 = _AVX512_MUL(h1, vs_2_4);
	h4 = _AVX512_MUL(h1, vs_3_4);

	w1 = _AVX512_FMSUB(w1, h1, _AVX512_FMA(z1, h4, _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2))));
	w2 = _AVX512_FMSUB(w2, h1, _AVX512_FMA(z2, h4, _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2))));
	w3 = _AVX512_FMSUB(w3, h1, _AVX512_FMA(z3, h4, _AVX512_FMA(y3, h3, _AVX512_MUL(x3,h2))));
	w4 = _AVX512_FMSUB(w4, h1, _AVX512_FMA(z4, h4, _AVX512_FMA(y4, h3, _AVX512_MUL(x4,h2))));

	q1 = _AVX512_LOAD(&q[0]);
	q1 = _AVX512_SUB(q1, w1);
	_AVX512_STORE(&q[0],q1);

	q2 = _AVX512_LOAD(&q[0+offset]);
	q2 = _AVX512_SUB(q2, w2);
	_AVX512_STORE(&q[0+offset],q2);

	q3 = _AVX512_LOAD(&q[0+2*offset]);
	q3 = _AVX512_SUB(q3, w3);
	_AVX512_STORE(&q[0+2*offset],q3);

	q4 = _AVX512_LOAD(&q[0+3*offset]);
	q4 = _AVX512_SUB(q4, w4);
	_AVX512_STORE(&q[0+3*offset],q4);


	h4 = _AVX512_SET1(hh[(ldh*3)+1]);
	q1 = _AVX512_LOAD(&q[ldq]);
	q2 = _AVX512_LOAD(&q[ldq+offset]);
	q3 = _AVX512_LOAD(&q[ldq+2*offset]);
	q4 = _AVX512_LOAD(&q[ldq+3*offset]);

	q1 = _AVX512_SUB(q1, _AVX512_FMA(w1, h4, z1));
	_AVX512_STORE(&q[ldq],q1);
	q2 = _AVX512_SUB(q2, _AVX512_FMA(w2, h4, z2));
	_AVX512_STORE(&q[ldq+offset],q2);
	q3 = _AVX512_SUB(q3, _AVX512_FMA(w3, h4, z3));
	_AVX512_STORE(&q[ldq+2*offset],q3);
	q4 = _AVX512_SUB(q4, _AVX512_FMA(w4, h4, z4));
	_AVX512_STORE(&q[ldq+3*offset],q4);


	h3 = _AVX512_SET1(hh[(ldh*2)+1]);
	h4 = _AVX512_SET1(hh[(ldh*3)+2]);
	q1 = _AVX512_LOAD(&q[ldq*2]);
	q2 = _AVX512_LOAD(&q[(ldq*2)+offset]);
	q3 = _AVX512_LOAD(&q[(ldq*2)+2*offset]);
	q4 = _AVX512_LOAD(&q[(ldq*2)+3*offset]);

        q1 = _AVX512_SUB(q1, y1);
        q1 = _AVX512_NFMA(z1, h3, q1);
        q1 = _AVX512_NFMA(w1, h4, q1);
	_AVX512_STORE(&q[ldq*2],q1);

        q2 = _AVX512_SUB(q2, y2);
        q2 = _AVX512_NFMA(z2, h3, q2);
        q2 = _AVX512_NFMA(w2, h4, q2);
	_AVX512_STORE(&q[(ldq*2)+offset],q2);

        q3 = _AVX512_SUB(q3, y3);
        q3 = _AVX512_NFMA(z3, h3, q3);
        q3 = _AVX512_NFMA(w3, h4, q3);
	_AVX512_STORE(&q[(ldq*2)+2*offset],q3);

        q4 = _AVX512_SUB(q4, y4);
        q4 = _AVX512_NFMA(z4, h3, q4);
        q4 = _AVX512_NFMA(w4, h4, q4);
	_AVX512_STORE(&q[(ldq*2)+3*offset],q4);


	h2 = _AVX512_SET1(hh[ldh+1]);
	h3 = _AVX512_SET1(hh[(ldh*2)+2]);
	h4 = _AVX512_SET1(hh[(ldh*3)+3]);

	q1 = _AVX512_LOAD(&q[ldq*3]);

	q1 = _AVX512_SUB(q1, x1);
        q1 = _AVX512_NFMA(y1, h2, q1);
        q1 = _AVX512_NFMA(z1, h3, q1);
        q1 = _AVX512_NFMA(w1, h4, q1);
	_AVX512_STORE(&q[ldq*3], q1);

	q2 = _AVX512_LOAD(&q[(ldq*3)+offset]);

	q2 = _AVX512_SUB(q2, x2);
        q2 = _AVX512_NFMA(y2, h2, q2);
        q2 = _AVX512_NFMA(z2, h3, q2);
        q2 = _AVX512_NFMA(w2, h4, q2);
	_AVX512_STORE(&q[(ldq*3)+offset], q2);

	q3 = _AVX512_LOAD(&q[(ldq*3)+2*offset]);

	q3 = _AVX512_SUB(q3, x3);
        q3 = _AVX512_NFMA(y3, h2, q3);
        q3 = _AVX512_NFMA(z3, h3, q3);
        q3 = _AVX512_NFMA(w3, h4, q3);
	_AVX512_STORE(&q[(ldq*3)+2*offset], q3);

	q4 = _AVX512_LOAD(&q[(ldq*3)+3*offset]);

	q4 = _AVX512_SUB(q4, x4);
        q4 = _AVX512_NFMA(y4, h2, q4);
        q4 = _AVX512_NFMA(z4, h3, q4);
        q4 = _AVX512_NFMA(w4, h4, q4);
	_AVX512_STORE(&q[(ldq*3)+3*offset], q4);

	for (i = 4; i < nb; i++)
	{
		h1 = _AVX512_SET1(hh[i-3]);
		h2 = _AVX512_SET1(hh[ldh+i-2]);
		h3 = _AVX512_SET1(hh[(ldh*2)+i-1]);
		h4 = _AVX512_SET1(hh[(ldh*3)+i]);

		q1 = _AVX512_LOAD(&q[i*ldq]);
                q1 = _AVX512_NFMA(x1, h1, q1);
                q1 = _AVX512_NFMA(y1, h2, q1);
                q1 = _AVX512_NFMA(z1, h3, q1);
                q1 = _AVX512_NFMA(w1, h4, q1);
		_AVX512_STORE(&q[i*ldq],q1);

		q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);
                q2 = _AVX512_NFMA(x2, h1, q2);
                q2 = _AVX512_NFMA(y2, h2, q2);
                q2 = _AVX512_NFMA(z2, h3, q2);
                q2 = _AVX512_NFMA(w2, h4, q2);
		_AVX512_STORE(&q[(i*ldq)+offset],q2);

		q3 = _AVX512_LOAD(&q[(i*ldq)+2*offset]);
                q3 = _AVX512_NFMA(x3, h1, q3);
                q3 = _AVX512_NFMA(y3, h2, q3);
                q3 = _AVX512_NFMA(z3, h3, q3);
                q3 = _AVX512_NFMA(w3, h4, q3);
		_AVX512_STORE(&q[(i*ldq)+2*offset],q3);

		q4 = _AVX512_LOAD(&q[(i*ldq)+3*offset]);
                q4 = _AVX512_NFMA(x4, h1, q4);
                q4 = _AVX512_NFMA(y4, h2, q4);
                q4 = _AVX512_NFMA(z4, h3, q4);
                q4 = _AVX512_NFMA(w4, h4, q4);
		_AVX512_STORE(&q[(i*ldq)+3*offset],q4);

	}

	h1 = _AVX512_SET1(hh[nb-3]);
	h2 = _AVX512_SET1(hh[ldh+nb-2]);
	h3 = _AVX512_SET1(hh[(ldh*2)+nb-1]);
	q1 = _AVX512_LOAD(&q[nb*ldq]);

	q1 = _AVX512_NFMA(x1, h1, q1);
        q1 = _AVX512_NFMA(y1, h2, q1);
        q1 = _AVX512_NFMA(z1, h3, q1);
	_AVX512_STORE(&q[nb*ldq],q1);

	q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);

	q2 = _AVX512_NFMA(x2, h1, q2);
        q2 = _AVX512_NFMA(y2, h2, q2);
        q2 = _AVX512_NFMA(z2, h3, q2);
	_AVX512_STORE(&q[(nb*ldq)+offset],q2);

	q3 = _AVX512_LOAD(&q[(nb*ldq)+2*offset]);

	q3 = _AVX512_NFMA(x3, h1, q3);
        q3 = _AVX512_NFMA(y3, h2, q3);
        q3 = _AVX512_NFMA(z3, h3, q3);
	_AVX512_STORE(&q[(nb*ldq)+2*offset],q3);

	q4 = _AVX512_LOAD(&q[(nb*ldq)+3*offset]);

	q4 = _AVX512_NFMA(x4, h1, q4);
        q4 = _AVX512_NFMA(y4, h2, q4);
        q4 = _AVX512_NFMA(z4, h3, q4);
	_AVX512_STORE(&q[(nb*ldq)+3*offset],q4);

	h1 = _AVX512_SET1(hh[nb-2]);
	h2 = _AVX512_SET1(hh[ldh+nb-1]);
	q1 = _AVX512_LOAD(&q[(nb+1)*ldq]);

	q1 = _AVX512_NFMA(x1, h1, q1);
	q1 = _AVX512_NFMA(y1, h2, q1);
	_AVX512_STORE(&q[(nb+1)*ldq],q1);

	q2 = _AVX512_LOAD(&q[((nb+1)*ldq)+offset]);

	q2 = _AVX512_NFMA(x2, h1, q2);
	q2 = _AVX512_NFMA(y2, h2, q2);
	_AVX512_STORE(&q[((nb+1)*ldq)+offset],q2);

	q3 = _AVX512_LOAD(&q[((nb+1)*ldq)+2*offset]);

	q3 = _AVX512_NFMA(x3, h1, q3);
	q3 = _AVX512_NFMA(y3, h2, q3);
	_AVX512_STORE(&q[((nb+1)*ldq)+2*offset],q3);

	q4 = _AVX512_LOAD(&q[((nb+1)*ldq)+3*offset]);

	q4 = _AVX512_NFMA(x4, h1, q4);
	q4 = _AVX512_NFMA(y4, h2, q4);
	_AVX512_STORE(&q[((nb+1)*ldq)+3*offset],q4);


	h1 = _AVX512_SET1(hh[nb-1]);
	q1 = _AVX512_LOAD(&q[(nb+2)*ldq]);

	q1 = _AVX512_NFMA(x1, h1, q1);
	_AVX512_STORE(&q[(nb+2)*ldq],q1);

	q2 = _AVX512_LOAD(&q[((nb+2)*ldq)+offset]);

	q2 = _AVX512_NFMA(x2, h1, q2);
	_AVX512_STORE(&q[((nb+2)*ldq)+offset],q2);

	q3 = _AVX512_LOAD(&q[((nb+2)*ldq)+2*offset]);

	q3 = _AVX512_NFMA(x3, h1, q3);
	_AVX512_STORE(&q[((nb+2)*ldq)+2*offset],q3);

	q4 = _AVX512_LOAD(&q[((nb+2)*ldq)+3*offset]);

	q4 = _AVX512_NFMA(x4, h1, q4);
	_AVX512_STORE(&q[((nb+2)*ldq)+3*offset],q4);

}

/**
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 24 rows of Q simultaneously, a
#endif
#ifdef DOUBLE_PRECISION_REAL
 * 48 rows of Q simultaneously, a
#endif
 * matrix Vector product with two householder
 * vectors + a rank 1 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_24_AVX512_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_48_AVX512_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__AVX512_DATATYPE a1_1 = _AVX512_LOAD(&q[ldq*3]);
	__AVX512_DATATYPE a2_1 = _AVX512_LOAD(&q[ldq*2]);
	__AVX512_DATATYPE a3_1 = _AVX512_LOAD(&q[ldq]);
	__AVX512_DATATYPE a4_1 = _AVX512_LOAD(&q[0]);


	__AVX512_DATATYPE a1_2 = _AVX512_LOAD(&q[(ldq*3)+offset]);
	__AVX512_DATATYPE a2_2 = _AVX512_LOAD(&q[(ldq*2)+offset]);
	__AVX512_DATATYPE a3_2 = _AVX512_LOAD(&q[ldq+offset]);
	__AVX512_DATATYPE a4_2 = _AVX512_LOAD(&q[0+offset]);

	__AVX512_DATATYPE a1_3 = _AVX512_LOAD(&q[(ldq*3)+2*offset]);
	__AVX512_DATATYPE a2_3 = _AVX512_LOAD(&q[(ldq*2)+2*offset]);
	__AVX512_DATATYPE a3_3 = _AVX512_LOAD(&q[ldq+2*offset]);
	__AVX512_DATATYPE a4_3 = _AVX512_LOAD(&q[0+2*offset]);

	__AVX512_DATATYPE h_2_1 = _AVX512_SET1(hh[ldh+1]);
	__AVX512_DATATYPE h_3_2 = _AVX512_SET1(hh[(ldh*2)+1]);
	__AVX512_DATATYPE h_3_1 = _AVX512_SET1(hh[(ldh*2)+2]);
	__AVX512_DATATYPE h_4_3 = _AVX512_SET1(hh[(ldh*3)+1]);
	__AVX512_DATATYPE h_4_2 = _AVX512_SET1(hh[(ldh*3)+2]);
	__AVX512_DATATYPE h_4_1 = _AVX512_SET1(hh[(ldh*3)+3]);

	__AVX512_DATATYPE w1 = _AVX512_FMA(a3_1, h_4_3, a4_1);
	w1 = _AVX512_FMA(a2_1, h_4_2, w1);
	w1 = _AVX512_FMA(a1_1, h_4_1, w1);
	__AVX512_DATATYPE z1 = _AVX512_FMA(a2_1, h_3_2, a3_1);
	z1 = _AVX512_FMA(a1_1, h_3_1, z1);
	__AVX512_DATATYPE y1 = _AVX512_FMA(a1_1, h_2_1, a2_1);
	__AVX512_DATATYPE x1 = a1_1;

	__AVX512_DATATYPE w2 = _AVX512_FMA(a3_2, h_4_3, a4_2);
	w2 = _AVX512_FMA(a2_2, h_4_2, w2);
	w2 = _AVX512_FMA(a1_2, h_4_1, w2);
	__AVX512_DATATYPE z2 = _AVX512_FMA(a2_2, h_3_2, a3_2);
	z2 = _AVX512_FMA(a1_2, h_3_1, z2);
	__AVX512_DATATYPE y2 = _AVX512_FMA(a1_2, h_2_1, a2_2);
	__AVX512_DATATYPE x2 = a1_2;

	__AVX512_DATATYPE w3 = _AVX512_FMA(a3_3, h_4_3, a4_3);
	w3 = _AVX512_FMA(a2_3, h_4_2, w3);
	w3 = _AVX512_FMA(a1_3, h_4_1, w3);
	__AVX512_DATATYPE z3 = _AVX512_FMA(a2_3, h_3_2, a3_3);
	z3 = _AVX512_FMA(a1_3, h_3_1, z3);
	__AVX512_DATATYPE y3 = _AVX512_FMA(a1_3, h_2_1, a2_3);
	__AVX512_DATATYPE x3 = a1_3;

	__AVX512_DATATYPE q1;
	__AVX512_DATATYPE q2;
	__AVX512_DATATYPE q3;

	__AVX512_DATATYPE h1;
	__AVX512_DATATYPE h2;
	__AVX512_DATATYPE h3;
	__AVX512_DATATYPE h4;

	for(i = 4; i < nb; i++)
	{
		h1 = _AVX512_SET1(hh[i-3]);
		h2 = _AVX512_SET1(hh[ldh+i-2]);
		h3 = _AVX512_SET1(hh[(ldh*2)+i-1]);
		h4 = _AVX512_SET1(hh[(ldh*3)+i]);

		q1 = _AVX512_LOAD(&q[i*ldq]);
		q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);
		q3 = _AVX512_LOAD(&q[(i*ldq)+2*offset]);

		x1 = _AVX512_FMA(q1, h1, x1);
		y1 = _AVX512_FMA(q1, h2, y1);
		z1 = _AVX512_FMA(q1, h3, z1);
		w1 = _AVX512_FMA(q1, h4, w1);

		x2 = _AVX512_FMA(q2, h1, x2);
		y2 = _AVX512_FMA(q2, h2, y2);
		z2 = _AVX512_FMA(q2, h3, z2);
		w2 = _AVX512_FMA(q2, h4, w2);

		x3 = _AVX512_FMA(q3, h1, x3);
		y3 = _AVX512_FMA(q3, h2, y3);
		z3 = _AVX512_FMA(q3, h3, z3);
		w3 = _AVX512_FMA(q3, h4, w3);

	}

	h1 = _AVX512_SET1(hh[nb-3]);
	h2 = _AVX512_SET1(hh[ldh+nb-2]);
	h3 = _AVX512_SET1(hh[(ldh*2)+nb-1]);

	q1 = _AVX512_LOAD(&q[nb*ldq]);
	q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);
	q3 = _AVX512_LOAD(&q[(nb*ldq)+2*offset]);

	x1 = _AVX512_FMA(q1, h1, x1);
	y1 = _AVX512_FMA(q1, h2, y1);
	z1 = _AVX512_FMA(q1, h3, z1);

	x2 = _AVX512_FMA(q2, h1, x2);
	y2 = _AVX512_FMA(q2, h2, y2);
	z2 = _AVX512_FMA(q2, h3, z2);

	x3 = _AVX512_FMA(q3, h1, x3);
	y3 = _AVX512_FMA(q3, h2, y3);
	z3 = _AVX512_FMA(q3, h3, z3);

	h1 = _AVX512_SET1(hh[nb-2]);
	h2 = _AVX512_SET1(hh[(ldh*1)+nb-1]);

	q1 = _AVX512_LOAD(&q[(nb+1)*ldq]);
	q2 = _AVX512_LOAD(&q[((nb+1)*ldq)+offset]);
	q3 = _AVX512_LOAD(&q[((nb+1)*ldq)+2*offset]);

	x1 = _AVX512_FMA(q1, h1, x1);
	y1 = _AVX512_FMA(q1, h2, y1);
	x2 = _AVX512_FMA(q2, h1, x2);
	y2 = _AVX512_FMA(q2, h2, y2);
	x3 = _AVX512_FMA(q3, h1, x3);
	y3 = _AVX512_FMA(q3, h2, y3);

	h1 = _AVX512_SET1(hh[nb-1]);

	q1 = _AVX512_LOAD(&q[(nb+2)*ldq]);
	q2 = _AVX512_LOAD(&q[((nb+2)*ldq)+offset]);
	q3 = _AVX512_LOAD(&q[((nb+2)*ldq)+2*offset]);

	x1 = _AVX512_FMA(q1, h1, x1);
	x2 = _AVX512_FMA(q2, h1, x2);
	x3 = _AVX512_FMA(q3, h1, x3);

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [8 x nb+3]
	/////////////////////////////////////////////////////

	__AVX512_DATATYPE tau1 = _AVX512_SET1(hh[0]);
	__AVX512_DATATYPE tau2 = _AVX512_SET1(hh[ldh]);
	__AVX512_DATATYPE tau3 = _AVX512_SET1(hh[ldh*2]);
	__AVX512_DATATYPE tau4 = _AVX512_SET1(hh[ldh*3]);

	__AVX512_DATATYPE vs_1_2 = _AVX512_SET1(s_1_2);
	__AVX512_DATATYPE vs_1_3 = _AVX512_SET1(s_1_3);
	__AVX512_DATATYPE vs_2_3 = _AVX512_SET1(s_2_3);
	__AVX512_DATATYPE vs_1_4 = _AVX512_SET1(s_1_4);
	__AVX512_DATATYPE vs_2_4 = _AVX512_SET1(s_2_4);
	__AVX512_DATATYPE vs_3_4 = _AVX512_SET1(s_3_4);

	h1 = tau1;
	x1 = _AVX512_MUL(x1, h1);
	x2 = _AVX512_MUL(x2, h1);
	x3 = _AVX512_MUL(x3, h1);

	h1 = tau2;
	h2 = _AVX512_MUL(h1, vs_1_2);

	y1 = _AVX512_FMSUB(y1, h1, _AVX512_MUL(x1,h2));
	y2 = _AVX512_FMSUB(y2, h1, _AVX512_MUL(x2,h2));
	y3 = _AVX512_FMSUB(y3, h1, _AVX512_MUL(x3,h2));

	h1 = tau3;
	h2 = _AVX512_MUL(h1, vs_1_3);
	h3 = _AVX512_MUL(h1, vs_2_3);

	z1 = _AVX512_FMSUB(z1, h1, _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2)));
	z2 = _AVX512_FMSUB(z2, h1, _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2)));
	z3 = _AVX512_FMSUB(z3, h1, _AVX512_FMA(y3, h3, _AVX512_MUL(x3,h2)));

	h1 = tau4;
	h2 = _AVX512_MUL(h1, vs_1_4);
	h3 = _AVX512_MUL(h1, vs_2_4);
	h4 = _AVX512_MUL(h1, vs_3_4);

	w1 = _AVX512_FMSUB(w1, h1, _AVX512_FMA(z1, h4, _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2))));
	w2 = _AVX512_FMSUB(w2, h1, _AVX512_FMA(z2, h4, _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2))));
	w3 = _AVX512_FMSUB(w3, h1, _AVX512_FMA(z3, h4, _AVX512_FMA(y3, h3, _AVX512_MUL(x3,h2))));

	q1 = _AVX512_LOAD(&q[0]);
	q1 = _AVX512_SUB(q1, w1);
	_AVX512_STORE(&q[0],q1);

	q2 = _AVX512_LOAD(&q[0+offset]);
	q2 = _AVX512_SUB(q2, w2);
	_AVX512_STORE(&q[0+offset],q2);

	q3 = _AVX512_LOAD(&q[0+2*offset]);
	q3 = _AVX512_SUB(q3, w3);
	_AVX512_STORE(&q[0+2*offset],q3);


	h4 = _AVX512_SET1(hh[(ldh*3)+1]);
	q1 = _AVX512_LOAD(&q[ldq]);
	q2 = _AVX512_LOAD(&q[ldq+offset]);
	q3 = _AVX512_LOAD(&q[ldq+2*offset]);

	q1 = _AVX512_SUB(q1, _AVX512_FMA(w1, h4, z1));
	_AVX512_STORE(&q[ldq],q1);
	q2 = _AVX512_SUB(q2, _AVX512_FMA(w2, h4, z2));
	_AVX512_STORE(&q[ldq+offset],q2);
	q3 = _AVX512_SUB(q3, _AVX512_FMA(w3, h4, z3));
	_AVX512_STORE(&q[ldq+2*offset],q3);


	h3 = _AVX512_SET1(hh[(ldh*2)+1]);
	h4 = _AVX512_SET1(hh[(ldh*3)+2]);
	q1 = _AVX512_LOAD(&q[ldq*2]);
	q2 = _AVX512_LOAD(&q[(ldq*2)+offset]);
	q3 = _AVX512_LOAD(&q[(ldq*2)+2*offset]);

        q1 = _AVX512_SUB(q1, y1);
        q1 = _AVX512_NFMA(z1, h3, q1);
        q1 = _AVX512_NFMA(w1, h4, q1);

	_AVX512_STORE(&q[ldq*2],q1);

        q2 = _AVX512_SUB(q2, y2);
        q2 = _AVX512_NFMA(z2, h3, q2);
        q2 = _AVX512_NFMA(w2, h4, q2);

	_AVX512_STORE(&q[(ldq*2)+offset],q2);

        q3 = _AVX512_SUB(q3, y3);
        q3 = _AVX512_NFMA(z3, h3, q3);
        q3 = _AVX512_NFMA(w3, h4, q3);

	_AVX512_STORE(&q[(ldq*2)+2*offset],q3);

	h2 = _AVX512_SET1(hh[ldh+1]);
	h3 = _AVX512_SET1(hh[(ldh*2)+2]);
	h4 = _AVX512_SET1(hh[(ldh*3)+3]);

	q1 = _AVX512_LOAD(&q[ldq*3]);

	q1 = _AVX512_SUB(q1, x1);
        q1 = _AVX512_NFMA(y1, h2, q1);
        q1 = _AVX512_NFMA(z1, h3, q1);
        q1 = _AVX512_NFMA(w1, h4, q1);

	_AVX512_STORE(&q[ldq*3], q1);

	q2 = _AVX512_LOAD(&q[(ldq*3)+offset]);

	q2 = _AVX512_SUB(q2, x2);
        q2 = _AVX512_NFMA(y2, h2, q2);
        q2 = _AVX512_NFMA(z2, h3, q2);
        q2 = _AVX512_NFMA(w2, h4, q2);

	_AVX512_STORE(&q[(ldq*3)+offset], q2);

	q3 = _AVX512_LOAD(&q[(ldq*3)+2*offset]);

	q3 = _AVX512_SUB(q3, x3);
        q3 = _AVX512_NFMA(y3, h2, q3);
        q3 = _AVX512_NFMA(z3, h3, q3);
        q3 = _AVX512_NFMA(w3, h4, q3);

	_AVX512_STORE(&q[(ldq*3)+2*offset], q3);

	for (i = 4; i < nb; i++)
	{
		h1 = _AVX512_SET1(hh[i-3]);
		h2 = _AVX512_SET1(hh[ldh+i-2]);
		h3 = _AVX512_SET1(hh[(ldh*2)+i-1]);
		h4 = _AVX512_SET1(hh[(ldh*3)+i]);

		q1 = _AVX512_LOAD(&q[i*ldq]);
                q1 = _AVX512_NFMA(x1, h1, q1);
                q1 = _AVX512_NFMA(y1, h2, q1);
                q1 = _AVX512_NFMA(z1, h3, q1);
                q1 = _AVX512_NFMA(w1, h4, q1);
		_AVX512_STORE(&q[i*ldq],q1);

		q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);
                q2 = _AVX512_NFMA(x2, h1, q2);
                q2 = _AVX512_NFMA(y2, h2, q2);
                q2 = _AVX512_NFMA(z2, h3, q2);
                q2 = _AVX512_NFMA(w2, h4, q2);
		_AVX512_STORE(&q[(i*ldq)+offset],q2);

		q3 = _AVX512_LOAD(&q[(i*ldq)+2*offset]);
                q3 = _AVX512_NFMA(x3, h1, q3);
                q3 = _AVX512_NFMA(y3, h2, q3);
                q3 = _AVX512_NFMA(z3, h3, q3);
                q3 = _AVX512_NFMA(w3, h4, q3);
		_AVX512_STORE(&q[(i*ldq)+2*offset],q3);

	}

	h1 = _AVX512_SET1(hh[nb-3]);
	h2 = _AVX512_SET1(hh[ldh+nb-2]);
	h3 = _AVX512_SET1(hh[(ldh*2)+nb-1]);
	q1 = _AVX512_LOAD(&q[nb*ldq]);

	q1 = _AVX512_NFMA(x1, h1, q1);
        q1 = _AVX512_NFMA(y1, h2, q1);
        q1 = _AVX512_NFMA(z1, h3, q1);

	_AVX512_STORE(&q[nb*ldq],q1);

	q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);

	q2 = _AVX512_NFMA(x2, h1, q2);
        q2 = _AVX512_NFMA(y2, h2, q2);
        q2 = _AVX512_NFMA(z2, h3, q2);

	_AVX512_STORE(&q[(nb*ldq)+offset],q2);

	q3 = _AVX512_LOAD(&q[(nb*ldq)+2*offset]);

	q3 = _AVX512_NFMA(x3, h1, q3);
        q3 = _AVX512_NFMA(y3, h2, q3);
        q3 = _AVX512_NFMA(z3, h3, q3);

	_AVX512_STORE(&q[(nb*ldq)+2*offset],q3);


	h1 = _AVX512_SET1(hh[nb-2]);
	h2 = _AVX512_SET1(hh[ldh+nb-1]);
	q1 = _AVX512_LOAD(&q[(nb+1)*ldq]);

	q1 = _AVX512_NFMA(x1, h1, q1);
	q1 = _AVX512_NFMA(y1, h2, q1);

	_AVX512_STORE(&q[(nb+1)*ldq],q1);

	q2 = _AVX512_LOAD(&q[((nb+1)*ldq)+offset]);

	q2 = _AVX512_NFMA(x2, h1, q2);
	q2 = _AVX512_NFMA(y2, h2, q2);

	_AVX512_STORE(&q[((nb+1)*ldq)+offset],q2);

	q3 = _AVX512_LOAD(&q[((nb+1)*ldq)+2*offset]);

	q3 = _AVX512_NFMA(x3, h1, q3);
	q3 = _AVX512_NFMA(y3, h2, q3);

	_AVX512_STORE(&q[((nb+1)*ldq)+2*offset],q3);


	h1 = _AVX512_SET1(hh[nb-1]);
	q1 = _AVX512_LOAD(&q[(nb+2)*ldq]);

	q1 = _AVX512_NFMA(x1, h1, q1);

	_AVX512_STORE(&q[(nb+2)*ldq],q1);

	q2 = _AVX512_LOAD(&q[((nb+2)*ldq)+offset]);

	q2 = _AVX512_NFMA(x2, h1, q2);

	_AVX512_STORE(&q[((nb+2)*ldq)+offset],q2);

	q3 = _AVX512_LOAD(&q[((nb+2)*ldq)+2*offset]);

	q3 = _AVX512_NFMA(x3, h1, q3);

	_AVX512_STORE(&q[((nb+2)*ldq)+2*offset],q3);

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
 * vectors + a rank 1 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_16_AVX512_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_32_AVX512_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__AVX512_DATATYPE a1_1 = _AVX512_LOAD(&q[ldq*3]);
	__AVX512_DATATYPE a2_1 = _AVX512_LOAD(&q[ldq*2]);
	__AVX512_DATATYPE a3_1 = _AVX512_LOAD(&q[ldq]);
	__AVX512_DATATYPE a4_1 = _AVX512_LOAD(&q[0]);

	__AVX512_DATATYPE a1_2 = _AVX512_LOAD(&q[(ldq*3)+offset]);
	__AVX512_DATATYPE a2_2 = _AVX512_LOAD(&q[(ldq*2)+offset]);
	__AVX512_DATATYPE a3_2 = _AVX512_LOAD(&q[ldq+offset]);
	__AVX512_DATATYPE a4_2 = _AVX512_LOAD(&q[0+offset]);

	__AVX512_DATATYPE h_2_1 = _AVX512_SET1(hh[ldh+1]);
	__AVX512_DATATYPE h_3_2 = _AVX512_SET1(hh[(ldh*2)+1]);
	__AVX512_DATATYPE h_3_1 = _AVX512_SET1(hh[(ldh*2)+2]);
	__AVX512_DATATYPE h_4_3 = _AVX512_SET1(hh[(ldh*3)+1]);
	__AVX512_DATATYPE h_4_2 = _AVX512_SET1(hh[(ldh*3)+2]);
	__AVX512_DATATYPE h_4_1 = _AVX512_SET1(hh[(ldh*3)+3]);

	__AVX512_DATATYPE w1 = _AVX512_FMA(a3_1, h_4_3, a4_1);
	w1 = _AVX512_FMA(a2_1, h_4_2, w1);
	w1 = _AVX512_FMA(a1_1, h_4_1, w1);
	__AVX512_DATATYPE z1 = _AVX512_FMA(a2_1, h_3_2, a3_1);
	z1 = _AVX512_FMA(a1_1, h_3_1, z1);
	__AVX512_DATATYPE y1 = _AVX512_FMA(a1_1, h_2_1, a2_1);
	__AVX512_DATATYPE x1 = a1_1;

	__AVX512_DATATYPE q1;
	__AVX512_DATATYPE q2;

	__AVX512_DATATYPE w2 = _AVX512_FMA(a3_2, h_4_3, a4_2);
	w2 = _AVX512_FMA(a2_2, h_4_2, w2);
	w2 = _AVX512_FMA(a1_2, h_4_1, w2);
	__AVX512_DATATYPE z2 = _AVX512_FMA(a2_2, h_3_2, a3_2);
	z2 = _AVX512_FMA(a1_2, h_3_1, z2);
	__AVX512_DATATYPE y2 = _AVX512_FMA(a1_2, h_2_1, a2_2);
	__AVX512_DATATYPE x2 = a1_2;

	__AVX512_DATATYPE h1;
	__AVX512_DATATYPE h2;
	__AVX512_DATATYPE h3;
	__AVX512_DATATYPE h4;

	for(i = 4; i < nb; i++)
	{
		h1 = _AVX512_SET1(hh[i-3]);
		h2 = _AVX512_SET1(hh[ldh+i-2]);
		h3 = _AVX512_SET1(hh[(ldh*2)+i-1]);
		h4 = _AVX512_SET1(hh[(ldh*3)+i]);

		q1 = _AVX512_LOAD(&q[i*ldq]);
		q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);

		x1 = _AVX512_FMA(q1, h1, x1);
		y1 = _AVX512_FMA(q1, h2, y1);
		z1 = _AVX512_FMA(q1, h3, z1);
		w1 = _AVX512_FMA(q1, h4, w1);

		x2 = _AVX512_FMA(q2, h1, x2);
		y2 = _AVX512_FMA(q2, h2, y2);
		z2 = _AVX512_FMA(q2, h3, z2);
		w2 = _AVX512_FMA(q2, h4, w2);
	}

	h1 = _AVX512_SET1(hh[nb-3]);
	h2 = _AVX512_SET1(hh[ldh+nb-2]);
	h3 = _AVX512_SET1(hh[(ldh*2)+nb-1]);

	q1 = _AVX512_LOAD(&q[nb*ldq]);
	q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);

	x1 = _AVX512_FMA(q1, h1, x1);
	y1 = _AVX512_FMA(q1, h2, y1);
	z1 = _AVX512_FMA(q1, h3, z1);

	x2 = _AVX512_FMA(q2, h1, x2);
	y2 = _AVX512_FMA(q2, h2, y2);
	z2 = _AVX512_FMA(q2, h3, z2);

	h1 = _AVX512_SET1(hh[nb-2]);
	h2 = _AVX512_SET1(hh[(ldh*1)+nb-1]);

	q1 = _AVX512_LOAD(&q[(nb+1)*ldq]);
	q2 = _AVX512_LOAD(&q[((nb+1)*ldq)+offset]);

	x1 = _AVX512_FMA(q1, h1, x1);
	y1 = _AVX512_FMA(q1, h2, y1);
	x2 = _AVX512_FMA(q2, h1, x2);
	y2 = _AVX512_FMA(q2, h2, y2);

	h1 = _AVX512_SET1(hh[nb-1]);

	q1 = _AVX512_LOAD(&q[(nb+2)*ldq]);
	q2 = _AVX512_LOAD(&q[((nb+2)*ldq)+offset]);

	x1 = _AVX512_FMA(q1, h1, x1);
	x2 = _AVX512_FMA(q2, h1, x2);

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [8 x nb+3]
	/////////////////////////////////////////////////////

	__AVX512_DATATYPE tau1 = _AVX512_SET1(hh[0]);
	__AVX512_DATATYPE tau2 = _AVX512_SET1(hh[ldh]);
	__AVX512_DATATYPE tau3 = _AVX512_SET1(hh[ldh*2]);
	__AVX512_DATATYPE tau4 = _AVX512_SET1(hh[ldh*3]);

	__AVX512_DATATYPE vs_1_2 = _AVX512_SET1(s_1_2);
	__AVX512_DATATYPE vs_1_3 = _AVX512_SET1(s_1_3);
	__AVX512_DATATYPE vs_2_3 = _AVX512_SET1(s_2_3);
	__AVX512_DATATYPE vs_1_4 = _AVX512_SET1(s_1_4);
	__AVX512_DATATYPE vs_2_4 = _AVX512_SET1(s_2_4);
	__AVX512_DATATYPE vs_3_4 = _AVX512_SET1(s_3_4);

	h1 = tau1;
	x1 = _AVX512_MUL(x1, h1);
	x2 = _AVX512_MUL(x2, h1);

	h1 = tau2;
	h2 = _AVX512_MUL(h1, vs_1_2);

	y1 = _AVX512_FMSUB(y1, h1, _AVX512_MUL(x1,h2));
	y2 = _AVX512_FMSUB(y2, h1, _AVX512_MUL(x2,h2));

	h1 = tau3;
	h2 = _AVX512_MUL(h1, vs_1_3);
	h3 = _AVX512_MUL(h1, vs_2_3);

	z1 = _AVX512_FMSUB(z1, h1, _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2)));
	z2 = _AVX512_FMSUB(z2, h1, _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2)));

	h1 = tau4;
	h2 = _AVX512_MUL(h1, vs_1_4);
	h3 = _AVX512_MUL(h1, vs_2_4);
	h4 = _AVX512_MUL(h1, vs_3_4);

	w1 = _AVX512_FMSUB(w1, h1, _AVX512_FMA(z1, h4, _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2))));
	w2 = _AVX512_FMSUB(w2, h1, _AVX512_FMA(z2, h4, _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2))));

	q1 = _AVX512_LOAD(&q[0]);
	q1 = _AVX512_SUB(q1, w1);
	_AVX512_STORE(&q[0],q1);

	q2 = _AVX512_LOAD(&q[0+offset]);
	q2 = _AVX512_SUB(q2, w2);
	_AVX512_STORE(&q[0+offset],q2);


	h4 = _AVX512_SET1(hh[(ldh*3)+1]);
	q1 = _AVX512_LOAD(&q[ldq]);
	q2 = _AVX512_LOAD(&q[ldq+offset]);

	q1 = _AVX512_SUB(q1, _AVX512_FMA(w1, h4, z1));
	_AVX512_STORE(&q[ldq],q1);
	q2 = _AVX512_SUB(q2, _AVX512_FMA(w2, h4, z2));
	_AVX512_STORE(&q[ldq+offset],q2);

	h3 = _AVX512_SET1(hh[(ldh*2)+1]);
	h4 = _AVX512_SET1(hh[(ldh*3)+2]);
	q1 = _AVX512_LOAD(&q[ldq*2]);
	q2 = _AVX512_LOAD(&q[(ldq*2)+offset]);

        q1 = _AVX512_SUB(q1, y1);
        q1 = _AVX512_NFMA(z1, h3, q1);
        q1 = _AVX512_NFMA(w1, h4, q1);

	_AVX512_STORE(&q[ldq*2],q1);

        q2 = _AVX512_SUB(q2, y2);
        q2 = _AVX512_NFMA(z2, h3, q2);
        q2 = _AVX512_NFMA(w2, h4, q2);

	_AVX512_STORE(&q[(ldq*2)+offset],q2);

	h2 = _AVX512_SET1(hh[ldh+1]);
	h3 = _AVX512_SET1(hh[(ldh*2)+2]);
	h4 = _AVX512_SET1(hh[(ldh*3)+3]);

	q1 = _AVX512_LOAD(&q[ldq*3]);

	q1 = _AVX512_SUB(q1, x1);
        q1 = _AVX512_NFMA(y1, h2, q1);
        q1 = _AVX512_NFMA(z1, h3, q1);
        q1 = _AVX512_NFMA(w1, h4, q1);

	_AVX512_STORE(&q[ldq*3], q1);

	q2 = _AVX512_LOAD(&q[(ldq*3)+offset]);

	q2 = _AVX512_SUB(q2, x2);
        q2 = _AVX512_NFMA(y2, h2, q2);
        q2 = _AVX512_NFMA(z2, h3, q2);
        q2 = _AVX512_NFMA(w2, h4, q2);

	_AVX512_STORE(&q[(ldq*3)+offset], q2);

	for (i = 4; i < nb; i++)
	{
		h1 = _AVX512_SET1(hh[i-3]);
		h2 = _AVX512_SET1(hh[ldh+i-2]);
		h3 = _AVX512_SET1(hh[(ldh*2)+i-1]);
		h4 = _AVX512_SET1(hh[(ldh*3)+i]);

		q1 = _AVX512_LOAD(&q[i*ldq]);
                q1 = _AVX512_NFMA(x1, h1, q1);
                q1 = _AVX512_NFMA(y1, h2, q1);
                q1 = _AVX512_NFMA(z1, h3, q1);
                q1 = _AVX512_NFMA(w1, h4, q1);
		_AVX512_STORE(&q[i*ldq],q1);

		q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);
                q2 = _AVX512_NFMA(x2, h1, q2);
                q2 = _AVX512_NFMA(y2, h2, q2);
                q2 = _AVX512_NFMA(z2, h3, q2);
                q2 = _AVX512_NFMA(w2, h4, q2);
		_AVX512_STORE(&q[(i*ldq)+offset],q2);

	}

	h1 = _AVX512_SET1(hh[nb-3]);
	h2 = _AVX512_SET1(hh[ldh+nb-2]);
	h3 = _AVX512_SET1(hh[(ldh*2)+nb-1]);
	q1 = _AVX512_LOAD(&q[nb*ldq]);

	q1 = _AVX512_NFMA(x1, h1, q1);
        q1 = _AVX512_NFMA(y1, h2, q1);
        q1 = _AVX512_NFMA(z1, h3, q1);

	_AVX512_STORE(&q[nb*ldq],q1);

	q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);

	q2 = _AVX512_NFMA(x2, h1, q2);
        q2 = _AVX512_NFMA(y2, h2, q2);
        q2 = _AVX512_NFMA(z2, h3, q2);

	_AVX512_STORE(&q[(nb*ldq)+offset],q2);

	h1 = _AVX512_SET1(hh[nb-2]);
	h2 = _AVX512_SET1(hh[ldh+nb-1]);
	q1 = _AVX512_LOAD(&q[(nb+1)*ldq]);

	q1 = _AVX512_NFMA(x1, h1, q1);
	q1 = _AVX512_NFMA(y1, h2, q1);

	_AVX512_STORE(&q[(nb+1)*ldq],q1);

	q2 = _AVX512_LOAD(&q[((nb+1)*ldq)+offset]);

	q2 = _AVX512_NFMA(x2, h1, q2);
	q2 = _AVX512_NFMA(y2, h2, q2);

	_AVX512_STORE(&q[((nb+1)*ldq)+offset],q2);

	h1 = _AVX512_SET1(hh[nb-1]);
	q1 = _AVX512_LOAD(&q[(nb+2)*ldq]);

	q1 = _AVX512_NFMA(x1, h1, q1);

	_AVX512_STORE(&q[(nb+2)*ldq],q1);

	q2 = _AVX512_LOAD(&q[((nb+2)*ldq)+offset]);

	q2 = _AVX512_NFMA(x2, h1, q2);

	_AVX512_STORE(&q[((nb+2)*ldq)+offset],q2);
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
 * vectors + a rank 1 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_8_AVX512_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_16_AVX512_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__AVX512_DATATYPE a1_1 = _AVX512_LOAD(&q[ldq*3]);
	__AVX512_DATATYPE a2_1 = _AVX512_LOAD(&q[ldq*2]);
	__AVX512_DATATYPE a3_1 = _AVX512_LOAD(&q[ldq]);
	__AVX512_DATATYPE a4_1 = _AVX512_LOAD(&q[0]);

	__AVX512_DATATYPE h_2_1 = _AVX512_SET1(hh[ldh+1]);
	__AVX512_DATATYPE h_3_2 = _AVX512_SET1(hh[(ldh*2)+1]);
	__AVX512_DATATYPE h_3_1 = _AVX512_SET1(hh[(ldh*2)+2]);
	__AVX512_DATATYPE h_4_3 = _AVX512_SET1(hh[(ldh*3)+1]);
	__AVX512_DATATYPE h_4_2 = _AVX512_SET1(hh[(ldh*3)+2]);
	__AVX512_DATATYPE h_4_1 = _AVX512_SET1(hh[(ldh*3)+3]);

	__AVX512_DATATYPE w1 = _AVX512_FMA(a3_1, h_4_3, a4_1);
	w1 = _AVX512_FMA(a2_1, h_4_2, w1);
	w1 = _AVX512_FMA(a1_1, h_4_1, w1);
	__AVX512_DATATYPE z1 = _AVX512_FMA(a2_1, h_3_2, a3_1);
	z1 = _AVX512_FMA(a1_1, h_3_1, z1);
	__AVX512_DATATYPE y1 = _AVX512_FMA(a1_1, h_2_1, a2_1);
	__AVX512_DATATYPE x1 = a1_1;

	__AVX512_DATATYPE q1;

	__AVX512_DATATYPE h1;
	__AVX512_DATATYPE h2;
	__AVX512_DATATYPE h3;
	__AVX512_DATATYPE h4;

	for(i = 4; i < nb; i++)
	{
		h1 = _AVX512_SET1(hh[i-3]);
		h2 = _AVX512_SET1(hh[ldh+i-2]);
		h3 = _AVX512_SET1(hh[(ldh*2)+i-1]);
		h4 = _AVX512_SET1(hh[(ldh*3)+i]);

		q1 = _AVX512_LOAD(&q[i*ldq]);

		x1 = _AVX512_FMA(q1, h1, x1);
		y1 = _AVX512_FMA(q1, h2, y1);
		z1 = _AVX512_FMA(q1, h3, z1);
		w1 = _AVX512_FMA(q1, h4, w1);

	}

	h1 = _AVX512_SET1(hh[nb-3]);
	h2 = _AVX512_SET1(hh[ldh+nb-2]);
	h3 = _AVX512_SET1(hh[(ldh*2)+nb-1]);

	q1 = _AVX512_LOAD(&q[nb*ldq]);

	x1 = _AVX512_FMA(q1, h1, x1);
	y1 = _AVX512_FMA(q1, h2, y1);
	z1 = _AVX512_FMA(q1, h3, z1);

	h1 = _AVX512_SET1(hh[nb-2]);
	h2 = _AVX512_SET1(hh[(ldh*1)+nb-1]);

	q1 = _AVX512_LOAD(&q[(nb+1)*ldq]);

	x1 = _AVX512_FMA(q1, h1, x1);
	y1 = _AVX512_FMA(q1, h2, y1);

	h1 = _AVX512_SET1(hh[nb-1]);

	q1 = _AVX512_LOAD(&q[(nb+2)*ldq]);

	x1 = _AVX512_FMA(q1, h1, x1);

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [8 x nb+3]
	/////////////////////////////////////////////////////

	__AVX512_DATATYPE tau1 = _AVX512_SET1(hh[0]);
	__AVX512_DATATYPE tau2 = _AVX512_SET1(hh[ldh]);
	__AVX512_DATATYPE tau3 = _AVX512_SET1(hh[ldh*2]);
	__AVX512_DATATYPE tau4 = _AVX512_SET1(hh[ldh*3]);

	__AVX512_DATATYPE vs_1_2 = _AVX512_SET1(s_1_2);
	__AVX512_DATATYPE vs_1_3 = _AVX512_SET1(s_1_3);
	__AVX512_DATATYPE vs_2_3 = _AVX512_SET1(s_2_3);
	__AVX512_DATATYPE vs_1_4 = _AVX512_SET1(s_1_4);
	__AVX512_DATATYPE vs_2_4 = _AVX512_SET1(s_2_4);
	__AVX512_DATATYPE vs_3_4 = _AVX512_SET1(s_3_4);

	h1 = tau1;
	x1 = _AVX512_MUL(x1, h1);

	h1 = tau2;
	h2 = _AVX512_MUL(h1, vs_1_2);

	y1 = _AVX512_FMSUB(y1, h1, _AVX512_MUL(x1,h2));

	h1 = tau3;
	h2 = _AVX512_MUL(h1, vs_1_3);
	h3 = _AVX512_MUL(h1, vs_2_3);

	z1 = _AVX512_FMSUB(z1, h1, _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2)));

	h1 = tau4;
	h2 = _AVX512_MUL(h1, vs_1_4);
	h3 = _AVX512_MUL(h1, vs_2_4);
	h4 = _AVX512_MUL(h1, vs_3_4);

	w1 = _AVX512_FMSUB(w1, h1, _AVX512_FMA(z1, h4, _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2))));

	q1 = _AVX512_LOAD(&q[0]);
	q1 = _AVX512_SUB(q1, w1);
	_AVX512_STORE(&q[0],q1);

	h4 = _AVX512_SET1(hh[(ldh*3)+1]);
	q1 = _AVX512_LOAD(&q[ldq]);

	q1 = _AVX512_SUB(q1, _AVX512_FMA(w1, h4, z1));

	_AVX512_STORE(&q[ldq],q1);

	h3 = _AVX512_SET1(hh[(ldh*2)+1]);
	h4 = _AVX512_SET1(hh[(ldh*3)+2]);
	q1 = _AVX512_LOAD(&q[ldq*2]);

        q1 = _AVX512_SUB(q1, y1);
        q1 = _AVX512_NFMA(z1, h3, q1);
        q1 = _AVX512_NFMA(w1, h4, q1);

	_AVX512_STORE(&q[ldq*2],q1);

	h2 = _AVX512_SET1(hh[ldh+1]);
	h3 = _AVX512_SET1(hh[(ldh*2)+2]);
	h4 = _AVX512_SET1(hh[(ldh*3)+3]);

	q1 = _AVX512_LOAD(&q[ldq*3]);

	q1 = _AVX512_SUB(q1, x1);
        q1 = _AVX512_NFMA(y1, h2, q1);
        q1 = _AVX512_NFMA(z1, h3, q1);
        q1 = _AVX512_NFMA(w1, h4, q1);

	_AVX512_STORE(&q[ldq*3], q1);

	for (i = 4; i < nb; i++)
	{
		h1 = _AVX512_SET1(hh[i-3]);
		h2 = _AVX512_SET1(hh[ldh+i-2]);
		h3 = _AVX512_SET1(hh[(ldh*2)+i-1]);
		h4 = _AVX512_SET1(hh[(ldh*3)+i]);

		q1 = _AVX512_LOAD(&q[i*ldq]);
                q1 = _AVX512_NFMA(x1, h1, q1);
                q1 = _AVX512_NFMA(y1, h2, q1);
                q1 = _AVX512_NFMA(z1, h3, q1);
                q1 = _AVX512_NFMA(w1, h4, q1);
		_AVX512_STORE(&q[i*ldq],q1);
	}

	h1 = _AVX512_SET1(hh[nb-3]);
	h2 = _AVX512_SET1(hh[ldh+nb-2]);
	h3 = _AVX512_SET1(hh[(ldh*2)+nb-1]);
	q1 = _AVX512_LOAD(&q[nb*ldq]);

	q1 = _AVX512_NFMA(x1, h1, q1);
        q1 = _AVX512_NFMA(y1, h2, q1);
        q1 = _AVX512_NFMA(z1, h3, q1);

	_AVX512_STORE(&q[nb*ldq],q1);

	h1 = _AVX512_SET1(hh[nb-2]);
	h2 = _AVX512_SET1(hh[ldh+nb-1]);
	q1 = _AVX512_LOAD(&q[(nb+1)*ldq]);

	q1 = _AVX512_NFMA(x1, h1, q1);
	q1 = _AVX512_NFMA(y1, h2, q1);

	_AVX512_STORE(&q[(nb+1)*ldq],q1);

	h1 = _AVX512_SET1(hh[nb-1]);
	q1 = _AVX512_LOAD(&q[(nb+2)*ldq]);

	q1 = _AVX512_NFMA(x1, h1, q1);

	_AVX512_STORE(&q[(nb+2)*ldq],q1);


}


#if 0
/**
 * Unrolled kernel that computes
 * 4 rows of Q simultaneously, a
 * matrix Vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_4_AVX512_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m256d a1_1 = _mm256_load_pd(&q[ldq*3]);
	__m256d a2_1 = _mm256_load_pd(&q[ldq*2]);
	__m256d a3_1 = _mm256_load_pd(&q[ldq]);
	__m256d a4_1 = _mm256_load_pd(&q[0]);

	__m256d h_2_1 = _mm256_broadcast_sd(&hh[ldh+1]);
	__m256d h_3_2 = _mm256_broadcast_sd(&hh[(ldh*2)+1]);
	__m256d h_3_1 = _mm256_broadcast_sd(&hh[(ldh*2)+2]);
	__m256d h_4_3 = _mm256_broadcast_sd(&hh[(ldh*3)+1]);
	__m256d h_4_2 = _mm256_broadcast_sd(&hh[(ldh*3)+2]);
	__m256d h_4_1 = _mm256_broadcast_sd(&hh[(ldh*3)+3]);

#ifdef __ELPA_USE_FMA__
	__m256d w1 = _mm256_FMA_pd(a3_1, h_4_3, a4_1);
	w1 = _mm256_FMA_pd(a2_1, h_4_2, w1);
	w1 = _mm256_FMA_pd(a1_1, h_4_1, w1);
	__m256d z1 = _mm256_FMA_pd(a2_1, h_3_2, a3_1);
	z1 = _mm256_FMA_pd(a1_1, h_3_1, z1);
	__m256d y1 = _mm256_FMA_pd(a1_1, h_2_1, a2_1);
	__m256d x1 = a1_1;
#else
	__m256d w1 = _mm256_add_pd(a4_1, _mm256_mul_pd(a3_1, h_4_3));
	w1 = _mm256_add_pd(w1, _mm256_mul_pd(a2_1, h_4_2));
	w1 = _mm256_add_pd(w1, _mm256_mul_pd(a1_1, h_4_1));
	__m256d z1 = _mm256_add_pd(a3_1, _mm256_mul_pd(a2_1, h_3_2));
	z1 = _mm256_add_pd(z1, _mm256_mul_pd(a1_1, h_3_1));
	__m256d y1 = _mm256_add_pd(a2_1, _mm256_mul_pd(a1_1, h_2_1));
	__m256d x1 = a1_1;
#endif

	__m256d q1;

	__m256d h1;
	__m256d h2;
	__m256d h3;
	__m256d h4;

	for(i = 4; i < nb; i++)
	{
		h1 = _mm256_broadcast_sd(&hh[i-3]);
		h2 = _mm256_broadcast_sd(&hh[ldh+i-2]);
		h3 = _mm256_broadcast_sd(&hh[(ldh*2)+i-1]);
		h4 = _mm256_broadcast_sd(&hh[(ldh*3)+i]);

		q1 = _mm256_load_pd(&q[i*ldq]);
#ifdef __ELPA_USE_FMA__
		x1 = _mm256_FMA_pd(q1, h1, x1);
		y1 = _mm256_FMA_pd(q1, h2, y1);
		z1 = _mm256_FMA_pd(q1, h3, z1);
		w1 = _mm256_FMA_pd(q1, h4, w1);
#else
		x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
		y1 = _mm256_add_pd(y1, _mm256_mul_pd(q1,h2));
		z1 = _mm256_add_pd(z1, _mm256_mul_pd(q1,h3));
		w1 = _mm256_add_pd(w1, _mm256_mul_pd(q1,h4));
#endif
	}

	h1 = _mm256_broadcast_sd(&hh[nb-3]);
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-2]);
	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+nb-1]);
	q1 = _mm256_load_pd(&q[nb*ldq]);
#ifdef _FMA4__
	x1 = _mm256_FMA_pd(q1, h1, x1);
	y1 = _mm256_FMA_pd(q1, h2, y1);
	z1 = _mm256_FMA_pd(q1, h3, z1);
#else
	x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
	y1 = _mm256_add_pd(y1, _mm256_mul_pd(q1,h2));
	z1 = _mm256_add_pd(z1, _mm256_mul_pd(q1,h3));
#endif

	h1 = _mm256_broadcast_sd(&hh[nb-2]);
	h2 = _mm256_broadcast_sd(&hh[(ldh*1)+nb-1]);
	q1 = _mm256_load_pd(&q[(nb+1)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm256_FMA_pd(q1, h1, x1);
	y1 = _mm256_FMA_pd(q1, h2, y1);
#else
	x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
	y1 = _mm256_add_pd(y1, _mm256_mul_pd(q1,h2));
#endif

	h1 = _mm256_broadcast_sd(&hh[nb-1]);
	q1 = _mm256_load_pd(&q[(nb+2)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm256_FMA_pd(q1, h1, x1);
#else
	x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
#endif

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [4 x nb+3]
	/////////////////////////////////////////////////////

	__m256d tau1 = _mm256_broadcast_sd(&hh[0]);
	__m256d tau2 = _mm256_broadcast_sd(&hh[ldh]);
	__m256d tau3 = _mm256_broadcast_sd(&hh[ldh*2]);
	__m256d tau4 = _mm256_broadcast_sd(&hh[ldh*3]);

	__m256d vs_1_2 = _mm256_broadcast_sd(&s_1_2);
	__m256d vs_1_3 = _mm256_broadcast_sd(&s_1_3);
	__m256d vs_2_3 = _mm256_broadcast_sd(&s_2_3);
	__m256d vs_1_4 = _mm256_broadcast_sd(&s_1_4);
	__m256d vs_2_4 = _mm256_broadcast_sd(&s_2_4);
	__m256d vs_3_4 = _mm256_broadcast_sd(&s_3_4);

	h1 = tau1;
	x1 = _mm256_mul_pd(x1, h1);

	h1 = tau2;
	h2 = _mm256_mul_pd(h1, vs_1_2);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_FMSUB_pd(y1, h1, _mm256_mul_pd(x1,h2));
#else
	y1 = _mm256_sub_pd(_mm256_mul_pd(y1,h1), _mm256_mul_pd(x1,h2));
#endif

	h1 = tau3;
	h2 = _mm256_mul_pd(h1, vs_1_3);
	h3 = _mm256_mul_pd(h1, vs_2_3);
#ifdef __ELPA_USE_FMA__
	z1 = _mm256_FMSUB_pd(z1, h1, _mm256_FMA_pd(y1, h3, _mm256_mul_pd(x1,h2)));
#else
	z1 = _mm256_sub_pd(_mm256_mul_pd(z1,h1), _mm256_add_pd(_mm256_mul_pd(y1,h3), _mm256_mul_pd(x1,h2)));
#endif

	h1 = tau4;
	h2 = _mm256_mul_pd(h1, vs_1_4);
	h3 = _mm256_mul_pd(h1, vs_2_4);
	h4 = _mm256_mul_pd(h1, vs_3_4);
#ifdef __ELPA_USE_FMA__
	w1 = _mm256_FMSUB_pd(w1, h1, _mm256_FMA_pd(z1, h4, _mm256_FMA_pd(y1, h3, _mm256_mul_pd(x1,h2))));
#else
	w1 = _mm256_sub_pd(_mm256_mul_pd(w1,h1), _mm256_add_pd(_mm256_mul_pd(z1,h4), _mm256_add_pd(_mm256_mul_pd(y1,h3), _mm256_mul_pd(x1,h2))));
#endif

	q1 = _mm256_load_pd(&q[0]);
	q1 = _mm256_sub_pd(q1, w1);
	_mm256_store_pd(&q[0],q1);

	h4 = _mm256_broadcast_sd(&hh[(ldh*3)+1]);
	q1 = _mm256_load_pd(&q[ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_sub_pd(q1, _mm256_FMA_pd(w1, h4, z1));
#else
	q1 = _mm256_sub_pd(q1, _mm256_add_pd(z1, _mm256_mul_pd(w1, h4)));
#endif
	_mm256_store_pd(&q[ldq],q1);

	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+1]);
	h4 = _mm256_broadcast_sd(&hh[(ldh*3)+2]);
	q1 = _mm256_load_pd(&q[ldq*2]);
#ifdef __ELPA_USE_FMA__
        q1 = _mm256_sub_pd(q1, y1);
        q1 = _mm256_NFMA_pd(z1, h3, q1);
        q1 = _mm256_NFMA_pd(w1, h4, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_add_pd(y1, _mm256_add_pd(_mm256_mul_pd(z1, h3), _mm256_mul_pd(w1, h4))));
#endif
	_mm256_store_pd(&q[ldq*2],q1);

	h2 = _mm256_broadcast_sd(&hh[ldh+1]);
	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+2]);
	h4 = _mm256_broadcast_sd(&hh[(ldh*3)+3]);
	q1 = _mm256_load_pd(&q[ldq*3]);
#ifdef __ELPA_USE_FMA__
        q1 = _mm256_sub_pd(q1, x1);
        q1 = _mm256_NFMA_pd(y1, h2, q1);
        q1 = _mm256_NFMA_pd(z1, h3, q1);
        q1 = _mm256_NFMA_pd(w1, h4, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_add_pd(x1, _mm256_add_pd(_mm256_mul_pd(y1, h2), _mm256_add_pd(_mm256_mul_pd(z1, h3), _mm256_mul_pd(w1, h4)))));
#endif
	_mm256_store_pd(&q[ldq*3], q1);

	for (i = 4; i < nb; i++)
	{
		h1 = _mm256_broadcast_sd(&hh[i-3]);
		h2 = _mm256_broadcast_sd(&hh[ldh+i-2]);
		h3 = _mm256_broadcast_sd(&hh[(ldh*2)+i-1]);
		h4 = _mm256_broadcast_sd(&hh[(ldh*3)+i]);

		q1 = _mm256_load_pd(&q[i*ldq]);
#ifdef __ELPA_USE_FMA__
                q1 = _mm256_NFMA_pd(x1, h1, q1);
                q1 = _mm256_NFMA_pd(y1, h2, q1);
                q1 = _mm256_NFMA_pd(z1, h3, q1);
                q1 = _mm256_NFMA_pd(w1, h4, q1);
#else
		q1 = _mm256_sub_pd(q1, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(w1, h4), _mm256_mul_pd(z1, h3)), _mm256_add_pd(_mm256_mul_pd(x1,h1), _mm256_mul_pd(y1, h2))));
#endif
		_mm256_store_pd(&q[i*ldq],q1);
	}

	h1 = _mm256_broadcast_sd(&hh[nb-3]);
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-2]);
	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+nb-1]);
	q1 = _mm256_load_pd(&q[nb*ldq]);
#ifdef __ELPA_USE_FMA__
        q1 = _mm256_NFMA_pd(x1, h1, q1);
        q1 = _mm256_NFMA_pd(y1, h2, q1);
        q1 = _mm256_NFMA_pd(z1, h3, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(z1, h3), _mm256_mul_pd(y1, h2)) , _mm256_mul_pd(x1, h1)));
#endif
	_mm256_store_pd(&q[nb*ldq],q1);

	h1 = _mm256_broadcast_sd(&hh[nb-2]);
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-1]);
	q1 = _mm256_load_pd(&q[(nb+1)*ldq]);
#ifdef __ELPA_USE_FMA__
        q1 = _mm256_NFMA_pd(x1, h1, q1);
        q1 = _mm256_NFMA_pd(y1, h2, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_add_pd( _mm256_mul_pd(y1, h2) , _mm256_mul_pd(x1, h1)));
#endif
	_mm256_store_pd(&q[(nb+1)*ldq],q1);

	h1 = _mm256_broadcast_sd(&hh[nb-1]);
	q1 = _mm256_load_pd(&q[(nb+2)*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(x1, h1, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(x1, h1));
#endif
	_mm256_store_pd(&q[(nb+2)*ldq],q1);
}
#endif

