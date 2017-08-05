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

#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>

#define __forceinline __attribute__((always_inline)) static

#ifdef DOUBLE_PRECISION_REAL
#define offset 4
#define __AVX_DATATYPE __m256d
#define _AVX_LOAD _mm256_load_pd
#define _AVX_STORE _mm256_store_pd
#define _AVX_ADD _mm256_add_pd
#define _AVX_SUB _mm256_sub_pd
#define _AVX_MUL _mm256_mul_pd
#define _AVX_BROADCAST _mm256_broadcast_sd

#ifdef HAVE_AVX2

#ifdef __FMA4__
#define __ELPA_USE_FMA__
#define _mm256_FMA_pd(a,b,c) _mm256_macc_pd(a,b,c)
#define _mm256_NFMA_pd(a,b,c) _mm256_nmacc_pd(a,b,c)
#error "This should be prop _mm256_msub_pd instead of _mm256_msub"
#define _mm256_FMSUB_pd(a,b,c) _mm256_msub(a,b,c)
#endif

#ifdef __AVX2__
#define __ELPA_USE_FMA__
#define _mm256_FMA_pd(a,b,c) _mm256_fmadd_pd(a,b,c)
#define _mm256_NFMA_pd(a,b,c) _mm256_fnmadd_pd(a,b,c)
#define _mm256_FMSUB_pd(a,b,c) _mm256_fmsub_pd(a,b,c)
#endif

#endif

#define _AVX_FMA _mm256_FMA_pd
#define _AVX_NFMA _mm256_NFMA_pd
#define _AVX_FMSUB _mm256_FMSUB_pd
#endif /* DOUBLE_PRECISION */

#ifdef SINGLE_PRECISION_REAL
#define offset 8
#define __AVX_DATATYPE __m256
#define _AVX_LOAD _mm256_load_ps
#define _AVX_STORE _mm256_store_ps
#define _AVX_ADD _mm256_add_ps
#define _AVX_SUB _mm256_sub_ps
#define _AVX_MUL _mm256_mul_ps
#define _AVX_BROADCAST _mm256_broadcast_ss

#ifdef HAVE_AVX2

#ifdef __FMA4__
#define __ELPA_USE_FMA__
#define _mm256_FMA_ps(a,b,c) _mm256_macc_ps(a,b,c)
#define _mm256_NFMA_ps(a,b,c) _mm256_nmacc_ps(a,b,c)
#error "This should be prop _mm256_msub_ps instead of _mm256_msub"
#define _mm256_FMSUB_ps(a,b,c) _mm256_msub(a,b,c)
#endif

#ifdef __AVX2__
#define __ELPA_USE_FMA__
#define _mm256_FMA_ps(a,b,c) _mm256_fmadd_ps(a,b,c)
#define _mm256_NFMA_ps(a,b,c) _mm256_fnmadd_ps(a,b,c)
#define _mm256_FMSUB_ps(a,b,c) _mm256_fmsub_ps(a,b,c)
#endif

#endif
#define _AVX_FMA _mm256_FMA_ps
#define _AVX_NFMA _mm256_NFMA_ps
#define _AVX_FMSUB _mm256_FMSUB_ps
#endif /* SINGLE_PRECISION */

#ifdef DOUBLE_PRECISION_REAL
//Forward declaration
__forceinline void hh_trafo_kernel_4_AVX_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4);
__forceinline void hh_trafo_kernel_8_AVX_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4);
__forceinline void hh_trafo_kernel_12_AVX_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4);

void quad_hh_trafo_real_avx_avx2_4hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif
#ifdef SINGLE_PRECISION_REAL
//Forward declaration
__forceinline void hh_trafo_kernel_8_AVX_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);
__forceinline void hh_trafo_kernel_16_AVX_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);
__forceinline void hh_trafo_kernel_24_AVX_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);

void quad_hh_trafo_real_avx_avx2_4hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif

#ifdef DOUBLE_PRECISION_REAL
/*
!f>#if defined(HAVE_AVX) || defined(HAVE_AVX2)
!f> interface
!f>   subroutine quad_hh_trafo_real_avx_avx2_4hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="quad_hh_trafo_real_avx_avx2_4hv_double")
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
!f>#if defined(HAVE_AVX) || defined(HAVE_AVX2)
!f> interface
!f>   subroutine quad_hh_trafo_real_avx_avx2_4hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="quad_hh_trafo_real_avx_avx2_4hv_single")
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
void quad_hh_trafo_real_avx_avx2_4hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
#ifdef SINGLE_PRECISION_REAL
void quad_hh_trafo_real_avx_avx2_4hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
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
	for (i = 0; i < nq-8; i+=12)
	{
		hh_trafo_kernel_12_AVX_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
		worked_on += 12;
	}
#endif
#ifdef SINGLE_PRECISION_REAL
	for (i = 0; i < nq-16; i+=24)
	{
		hh_trafo_kernel_24_AVX_4hv_single(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
		worked_on += 24;
	}
#endif
	if (nq == i)
	{
		return;
	}

#ifdef DOUBLE_PRECISION_REAL
        if (nq-i == 8)
	{
		hh_trafo_kernel_8_AVX_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
		worked_on += 8;
	}
#endif

#ifdef SINGLE_PRECISION_REAL
        if (nq-i == 16)
	{
		hh_trafo_kernel_16_AVX_4hv_single(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
		worked_on += 16;
	}
#endif

#ifdef DOUBLE_PRECISION_REAL
        if (nq-i == 4)
	{
		hh_trafo_kernel_4_AVX_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
		worked_on += 4;
	}
#endif

#ifdef SINGLE_PRECISION_REAL
        if (nq-i == 8)
	{
		hh_trafo_kernel_8_AVX_4hv_single(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
		worked_on += 8;
	}
#endif

#ifdef WITH_DEBUG
        if (worked_on != nq)
	{
	    printf("Error in real AVX/AVX2 BLOCK4 kernel \n");
	    abort();
	}
#endif
}
/**
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 12 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 24 rows of Q simultaneously, a
#endif
 * matrix Vector product with two householder
 * vectors + a rank 1 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_12_AVX_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_24_AVX_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [12 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__AVX_DATATYPE a1_1 = _AVX_LOAD(&q[ldq*3]);
	__AVX_DATATYPE a2_1 = _AVX_LOAD(&q[ldq*2]);
	__AVX_DATATYPE a3_1 = _AVX_LOAD(&q[ldq]);
	__AVX_DATATYPE a4_1 = _AVX_LOAD(&q[0]);

	__AVX_DATATYPE h_2_1 = _AVX_BROADCAST(&hh[ldh+1]);
	__AVX_DATATYPE h_3_2 = _AVX_BROADCAST(&hh[(ldh*2)+1]);
	__AVX_DATATYPE h_3_1 = _AVX_BROADCAST(&hh[(ldh*2)+2]);
	__AVX_DATATYPE h_4_3 = _AVX_BROADCAST(&hh[(ldh*3)+1]);
	__AVX_DATATYPE h_4_2 = _AVX_BROADCAST(&hh[(ldh*3)+2]);
	__AVX_DATATYPE h_4_1 = _AVX_BROADCAST(&hh[(ldh*3)+3]);

#ifdef __ELPA_USE_FMA__
	register __AVX_DATATYPE w1 = _AVX_FMA(a3_1, h_4_3, a4_1);
	w1 = _AVX_FMA(a2_1, h_4_2, w1);
	w1 = _AVX_FMA(a1_1, h_4_1, w1);
	register __AVX_DATATYPE z1 = _AVX_FMA(a2_1, h_3_2, a3_1);
	z1 = _AVX_FMA(a1_1, h_3_1, z1);
	register __AVX_DATATYPE y1 = _AVX_FMA(a1_1, h_2_1, a2_1);
	register __AVX_DATATYPE x1 = a1_1;
#else
	register __AVX_DATATYPE w1 = _AVX_ADD(a4_1, _AVX_MUL(a3_1, h_4_3));
	w1 = _AVX_ADD(w1, _AVX_MUL(a2_1, h_4_2));
	w1 = _AVX_ADD(w1, _AVX_MUL(a1_1, h_4_1));
	register __AVX_DATATYPE z1 = _AVX_ADD(a3_1, _AVX_MUL(a2_1, h_3_2));
	z1 = _AVX_ADD(z1, _AVX_MUL(a1_1, h_3_1));
	register __AVX_DATATYPE y1 = _AVX_ADD(a2_1, _AVX_MUL(a1_1, h_2_1));
	register __AVX_DATATYPE x1 = a1_1;
#endif

	__AVX_DATATYPE a1_2 = _AVX_LOAD(&q[(ldq*3)+offset]);
	__AVX_DATATYPE a2_2 = _AVX_LOAD(&q[(ldq*2)+offset]);
	__AVX_DATATYPE a3_2 = _AVX_LOAD(&q[ldq+offset]);
	__AVX_DATATYPE a4_2 = _AVX_LOAD(&q[0+offset]);

#ifdef __ELPA_USE_FMA__
	register __AVX_DATATYPE w2 = _AVX_FMA(a3_2, h_4_3, a4_2);
	w2 = _AVX_FMA(a2_2, h_4_2, w2);
	w2 = _AVX_FMA(a1_2, h_4_1, w2);
	register __AVX_DATATYPE z2 = _AVX_FMA(a2_2, h_3_2, a3_2);
	z2 = _AVX_FMA(a1_2, h_3_1, z2);
	register __AVX_DATATYPE y2 = _AVX_FMA(a1_2, h_2_1, a2_2);
	register __AVX_DATATYPE x2 = a1_2;
#else
	register __AVX_DATATYPE w2 = _AVX_ADD(a4_2, _AVX_MUL(a3_2, h_4_3));
	w2 = _AVX_ADD(w2, _AVX_MUL(a2_2, h_4_2));
	w2 = _AVX_ADD(w2, _AVX_MUL(a1_2, h_4_1));
	register __AVX_DATATYPE z2 = _AVX_ADD(a3_2, _AVX_MUL(a2_2, h_3_2));
	z2 = _AVX_ADD(z2, _AVX_MUL(a1_2, h_3_1));
	register __AVX_DATATYPE y2 = _AVX_ADD(a2_2, _AVX_MUL(a1_2, h_2_1));
	register __AVX_DATATYPE x2 = a1_2;
#endif

	__AVX_DATATYPE a1_3 = _AVX_LOAD(&q[(ldq*3)+2*offset]);
	__AVX_DATATYPE a2_3 = _AVX_LOAD(&q[(ldq*2)+2*offset]);
	__AVX_DATATYPE a3_3 = _AVX_LOAD(&q[ldq+2*offset]);
	__AVX_DATATYPE a4_3 = _AVX_LOAD(&q[0+2*offset]);

#ifdef __ELPA_USE_FMA__
	register __AVX_DATATYPE w3 = _AVX_FMA(a3_3, h_4_3, a4_3);
	w3 = _AVX_FMA(a2_3, h_4_2, w3);
	w3 = _AVX_FMA(a1_3, h_4_1, w3);
	register __AVX_DATATYPE z3 = _AVX_FMA(a2_3, h_3_2, a3_3);
	z3 = _AVX_FMA(a1_3, h_3_1, z3);
	register __AVX_DATATYPE y3 = _AVX_FMA(a1_3, h_2_1, a2_3);
	register __AVX_DATATYPE x3 = a1_3;
#else
	register __AVX_DATATYPE w3 = _AVX_ADD(a4_3, _AVX_MUL(a3_3, h_4_3));
	w3 = _AVX_ADD(w3, _AVX_MUL(a2_3, h_4_2));
	w3 = _AVX_ADD(w3, _AVX_MUL(a1_3, h_4_1));
	register __AVX_DATATYPE z3 = _AVX_ADD(a3_3, _AVX_MUL(a2_3, h_3_2));
	z3 = _AVX_ADD(z3, _AVX_MUL(a1_3, h_3_1));
	register __AVX_DATATYPE y3 = _AVX_ADD(a2_3, _AVX_MUL(a1_3, h_2_1));
	register __AVX_DATATYPE x3 = a1_3;
#endif

	__AVX_DATATYPE q1;
	__AVX_DATATYPE q2;
	__AVX_DATATYPE q3;

	__AVX_DATATYPE h1;
	__AVX_DATATYPE h2;
	__AVX_DATATYPE h3;
	__AVX_DATATYPE h4;

	for(i = 4; i < nb; i++)
	{
		h1 = _AVX_BROADCAST(&hh[i-3]);
		q1 = _AVX_LOAD(&q[i*ldq]);
		q2 = _AVX_LOAD(&q[(i*ldq)+offset]);
		q3 = _AVX_LOAD(&q[(i*ldq)+2*offset]);
#ifdef __ELPA_USE_FMA__
		x1 = _AVX_FMA(q1, h1, x1);
		x2 = _AVX_FMA(q2, h1, x2);
		x3 = _AVX_FMA(q3, h1, x3);
#else
		x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
		x2 = _AVX_ADD(x2, _AVX_MUL(q2,h1));
		x3 = _AVX_ADD(x3, _AVX_MUL(q3,h1));
#endif

		h2 = _AVX_BROADCAST(&hh[ldh+i-2]);
#ifdef __ELPA_USE_FMA__
		y1 = _AVX_FMA(q1, h2, y1);
		y2 = _AVX_FMA(q2, h2, y2);
		y3 = _AVX_FMA(q3, h2, y3);
#else
		y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
		y2 = _AVX_ADD(y2, _AVX_MUL(q2,h2));
		y3 = _AVX_ADD(y3, _AVX_MUL(q3,h2));
#endif

		h3 = _AVX_BROADCAST(&hh[(ldh*2)+i-1]);
#ifdef __ELPA_USE_FMA__
		z1 = _AVX_FMA(q1, h3, z1);
		z2 = _AVX_FMA(q2, h3, z2);
		z3 = _AVX_FMA(q3, h3, z3);
#else
		z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
		z2 = _AVX_ADD(z2, _AVX_MUL(q2,h3));
		z3 = _AVX_ADD(z3, _AVX_MUL(q3,h3));
#endif

		h4 = _AVX_BROADCAST(&hh[(ldh*3)+i]);
#ifdef __ELPA_USE_FMA__
		w1 = _AVX_FMA(q1, h4, w1);
		w2 = _AVX_FMA(q2, h4, w2);
		w3 = _AVX_FMA(q3, h4, w3);
#else
		w1 = _AVX_ADD(w1, _AVX_MUL(q1,h4));
		w2 = _AVX_ADD(w2, _AVX_MUL(q2,h4));
		w3 = _AVX_ADD(w3, _AVX_MUL(q3,h4));
#endif
	}

	h1 = _AVX_BROADCAST(&hh[nb-3]);

	q1 = _AVX_LOAD(&q[nb*ldq]);
	q2 = _AVX_LOAD(&q[(nb*ldq)+offset]);
	q3 = _AVX_LOAD(&q[(nb*ldq)+2*offset]);

#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
	x2 = _AVX_FMA(q2, h1, x2);
	x3 = _AVX_FMA(q3, h1, x3);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
	x2 = _AVX_ADD(x2, _AVX_MUL(q2,h1));
	x3 = _AVX_ADD(x3, _AVX_MUL(q3,h1));
#endif

	h2 = _AVX_BROADCAST(&hh[ldh+nb-2]);
#ifdef __FMA4_
	y1 = _AVX_FMA(q1, h2, y1);
	y2 = _AVX_FMA(q2, h2, y2);
	y3 = _AVX_FMA(q3, h2, y3);
#else
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
	y2 = _AVX_ADD(y2, _AVX_MUL(q2,h2));
	y3 = _AVX_ADD(y3, _AVX_MUL(q3,h2));
#endif

	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-1]);
#ifdef __ELPA_USE_FMA__
	z1 = _AVX_FMA(q1, h3, z1);
	z2 = _AVX_FMA(q2, h3, z2);
	z3 = _AVX_FMA(q3, h3, z3);
#else
	z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
	z2 = _AVX_ADD(z2, _AVX_MUL(q2,h3));
	z3 = _AVX_ADD(z3, _AVX_MUL(q3,h3));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-2]);

	q1 = _AVX_LOAD(&q[(nb+1)*ldq]);
	q2 = _AVX_LOAD(&q[((nb+1)*ldq)+offset]);
	q3 = _AVX_LOAD(&q[((nb+1)*ldq)+2*offset]);

#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
	x2 = _AVX_FMA(q2, h1, x2);
	x3 = _AVX_FMA(q3, h1, x3);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
	x2 = _AVX_ADD(x2, _AVX_MUL(q2,h1));
	x3 = _AVX_ADD(x3, _AVX_MUL(q3,h1));
#endif

	h2 = _AVX_BROADCAST(&hh[(ldh*1)+nb-1]);

#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMA(q1, h2, y1);
	y2 = _AVX_FMA(q2, h2, y2);
	y3 = _AVX_FMA(q3, h2, y3);
#else
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
	y2 = _AVX_ADD(y2, _AVX_MUL(q2,h2));
	y3 = _AVX_ADD(y3, _AVX_MUL(q3,h2));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-1]);

	q1 = _AVX_LOAD(&q[(nb+2)*ldq]);
	q2 = _AVX_LOAD(&q[((nb+2)*ldq)+offset]);
	q3 = _AVX_LOAD(&q[((nb+2)*ldq)+2*offset]);

#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
	x2 = _AVX_FMA(q2, h1, x2);
	x3 = _AVX_FMA(q3, h1, x3);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
	x2 = _AVX_ADD(x2, _AVX_MUL(q2,h1));
	x3 = _AVX_ADD(x3, _AVX_MUL(q3,h1));
#endif

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [12 x nb+3]
	/////////////////////////////////////////////////////

	__AVX_DATATYPE tau1 = _AVX_BROADCAST(&hh[0]);

	h1 = tau1;
	x1 = _AVX_MUL(x1, h1);
	x2 = _AVX_MUL(x2, h1);
	x3 = _AVX_MUL(x3, h1);

	__AVX_DATATYPE tau2 = _AVX_BROADCAST(&hh[ldh]);
	__AVX_DATATYPE vs_1_2 = _AVX_BROADCAST(&s_1_2);

	h1 = tau2;
	h2 = _AVX_MUL(h1, vs_1_2);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMSUB(y1, h1, _AVX_MUL(x1,h2));
	y2 = _AVX_FMSUB(y2, h1, _AVX_MUL(x2,h2));
	y3 = _AVX_FMSUB(y3, h1, _AVX_MUL(x3,h2));
#else
	y1 = _AVX_SUB(_AVX_MUL(y1,h1), _AVX_MUL(x1,h2));
	y2 = _AVX_SUB(_AVX_MUL(y2,h1), _AVX_MUL(x2,h2));
	y3 = _AVX_SUB(_AVX_MUL(y3,h1), _AVX_MUL(x3,h2));
#endif

	__AVX_DATATYPE tau3 = _AVX_BROADCAST(&hh[ldh*2]);
	__AVX_DATATYPE vs_1_3 = _AVX_BROADCAST(&s_1_3);
	__AVX_DATATYPE vs_2_3 = _AVX_BROADCAST(&s_2_3);

	h1 = tau3;
	h2 = _AVX_MUL(h1, vs_1_3);
	h3 = _AVX_MUL(h1, vs_2_3);
#ifdef __ELPA_USE_FMA__
	z1 = _AVX_FMSUB(z1, h1, _AVX_FMA(y1, h3, _AVX_MUL(x1,h2)));
	z2 = _AVX_FMSUB(z2, h1, _AVX_FMA(y2, h3, _AVX_MUL(x2,h2)));
	z3 = _AVX_FMSUB(z3, h1, _AVX_FMA(y3, h3, _AVX_MUL(x3,h2)));
#else
	z1 = _AVX_SUB(_AVX_MUL(z1,h1), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2)));
	z2 = _AVX_SUB(_AVX_MUL(z2,h1), _AVX_ADD(_AVX_MUL(y2,h3), _AVX_MUL(x2,h2)));
	z3 = _AVX_SUB(_AVX_MUL(z3,h1), _AVX_ADD(_AVX_MUL(y3,h3), _AVX_MUL(x3,h2)));
#endif

	__AVX_DATATYPE tau4 = _AVX_BROADCAST(&hh[ldh*3]);
	__AVX_DATATYPE vs_1_4 = _AVX_BROADCAST(&s_1_4);
	__AVX_DATATYPE vs_2_4 = _AVX_BROADCAST(&s_2_4);
	__AVX_DATATYPE vs_3_4 = _AVX_BROADCAST(&s_3_4);

	h1 = tau4;
	h2 = _AVX_MUL(h1, vs_1_4);
	h3 = _AVX_MUL(h1, vs_2_4);
	h4 = _AVX_MUL(h1, vs_3_4);
#ifdef __ELPA_USE_FMA__
	w1 = _AVX_FMSUB(w1, h1, _AVX_FMA(z1, h4, _AVX_FMA(y1, h3, _AVX_MUL(x1,h2))));
	w2 = _AVX_FMSUB(w2, h1, _AVX_FMA(z2, h4, _AVX_FMA(y2, h3, _AVX_MUL(x2,h2))));
	w3 = _AVX_FMSUB(w3, h1, _AVX_FMA(z3, h4, _AVX_FMA(y3, h3, _AVX_MUL(x3,h2))));
#else
	w1 = _AVX_SUB(_AVX_MUL(w1,h1), _AVX_ADD(_AVX_MUL(z1,h4), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2))));
	w2 = _AVX_SUB(_AVX_MUL(w2,h1), _AVX_ADD(_AVX_MUL(z2,h4), _AVX_ADD(_AVX_MUL(y2,h3), _AVX_MUL(x2,h2))));
	w3 = _AVX_SUB(_AVX_MUL(w3,h1), _AVX_ADD(_AVX_MUL(z3,h4), _AVX_ADD(_AVX_MUL(y3,h3), _AVX_MUL(x3,h2))));
#endif

	q1 = _AVX_LOAD(&q[0]);
	q2 = _AVX_LOAD(&q[offset]);
	q3 = _AVX_LOAD(&q[2*offset]);
	q1 = _AVX_SUB(q1, w1);
	q2 = _AVX_SUB(q2, w2);
	q3 = _AVX_SUB(q3, w3);
	_AVX_STORE(&q[0],q1);
	_AVX_STORE(&q[offset],q2);
	_AVX_STORE(&q[2*offset],q3);

	h4 = _AVX_BROADCAST(&hh[(ldh*3)+1]);
	q1 = _AVX_LOAD(&q[ldq]);
	q2 = _AVX_LOAD(&q[ldq+offset]);
	q3 = _AVX_LOAD(&q[ldq+2*offset]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_SUB(q1, _AVX_FMA(w1, h4, z1));
	q2 = _AVX_SUB(q2, _AVX_FMA(w2, h4, z2));
	q3 = _AVX_SUB(q3, _AVX_FMA(w3, h4, z3));
#else
	q1 = _AVX_SUB(q1, _AVX_ADD(z1, _AVX_MUL(w1, h4)));
	q2 = _AVX_SUB(q2, _AVX_ADD(z2, _AVX_MUL(w2, h4)));
	q3 = _AVX_SUB(q3, _AVX_ADD(z3, _AVX_MUL(w3, h4)));
#endif
	_AVX_STORE(&q[ldq],q1);
	_AVX_STORE(&q[ldq+offset],q2);
	_AVX_STORE(&q[ldq+2*offset],q3);

	h4 = _AVX_BROADCAST(&hh[(ldh*3)+2]);
	q1 = _AVX_LOAD(&q[ldq*2]);
	q2 = _AVX_LOAD(&q[(ldq*2)+offset]);
	q3 = _AVX_LOAD(&q[(ldq*2)+2*offset]);
	q1 = _AVX_SUB(q1, y1);
	q2 = _AVX_SUB(q2, y2);
	q3 = _AVX_SUB(q3, y3);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(w1, h4, q1);
	q2 = _AVX_NFMA(w2, h4, q2);
	q3 = _AVX_NFMA(w3, h4, q3);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
	q2 = _AVX_SUB(q2, _AVX_MUL(w2, h4));
	q3 = _AVX_SUB(q3, _AVX_MUL(w3, h4));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+1]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
	q2 = _AVX_NFMA(z2, h3, q2);
	q3 = _AVX_NFMA(z3, h3, q3);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
	q2 = _AVX_SUB(q2, _AVX_MUL(z2, h3));
	q3 = _AVX_SUB(q3, _AVX_MUL(z3, h3));
#endif
	_AVX_STORE(&q[ldq*2],q1);
	_AVX_STORE(&q[(ldq*2)+offset],q2);
	_AVX_STORE(&q[(ldq*2)+2*offset],q3);

	h4 = _AVX_BROADCAST(&hh[(ldh*3)+3]);
	q1 = _AVX_LOAD(&q[ldq*3]);
	q2 = _AVX_LOAD(&q[(ldq*3)+offset]);
	q3 = _AVX_LOAD(&q[(ldq*3)+2*offset]);
	q1 = _AVX_SUB(q1, x1);
	q2 = _AVX_SUB(q2, x2);
	q3 = _AVX_SUB(q3, x3);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(w1, h4, q1);
	q2 = _AVX_NFMA(w2, h4, q2);
	q3 = _AVX_NFMA(w3, h4, q3);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
	q2 = _AVX_SUB(q2, _AVX_MUL(w2, h4));
	q3 = _AVX_SUB(q3, _AVX_MUL(w3, h4));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+1]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
	q2 = _AVX_NFMA(y2, h2, q2);
	q3 = _AVX_NFMA(y3, h2, q3);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
	q2 = _AVX_SUB(q2, _AVX_MUL(y2, h2));
	q3 = _AVX_SUB(q3, _AVX_MUL(y3, h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
	q2 = _AVX_NFMA(z2, h3, q2);
	q3 = _AVX_NFMA(z3, h3, q3);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
	q2 = _AVX_SUB(q2, _AVX_MUL(z2, h3));
	q3 = _AVX_SUB(q3, _AVX_MUL(z3, h3));
#endif
	_AVX_STORE(&q[ldq*3], q1);
	_AVX_STORE(&q[(ldq*3)+offset], q2);
	_AVX_STORE(&q[(ldq*3)+2*offset], q3);

	for (i = 4; i < nb; i++)
	{
		h1 = _AVX_BROADCAST(&hh[i-3]);

		q1 = _AVX_LOAD(&q[i*ldq]);
		q2 = _AVX_LOAD(&q[(i*ldq)+offset]);
		q3 = _AVX_LOAD(&q[(i*ldq)+2*offset]);

#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(x1, h1, q1);
		q2 = _AVX_NFMA(x2, h1, q2);
		q3 = _AVX_NFMA(x3, h1, q3);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(x1,h1));
		q2 = _AVX_SUB(q2, _AVX_MUL(x2,h1));
		q3 = _AVX_SUB(q3, _AVX_MUL(x3,h1));
#endif

		h2 = _AVX_BROADCAST(&hh[ldh+i-2]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(y1, h2, q1);
		q2 = _AVX_NFMA(y2, h2, q2);
		q3 = _AVX_NFMA(y3, h2, q3);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(y1,h2));
		q2 = _AVX_SUB(q2, _AVX_MUL(y2,h2));
		q3 = _AVX_SUB(q3, _AVX_MUL(y3,h2));
#endif

		h3 = _AVX_BROADCAST(&hh[(ldh*2)+i-1]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(z1, h3, q1);
		q2 = _AVX_NFMA(z2, h3, q2);
		q3 = _AVX_NFMA(z3, h3, q3);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(z1,h3));
		q2 = _AVX_SUB(q2, _AVX_MUL(z2,h3));
		q3 = _AVX_SUB(q3, _AVX_MUL(z3,h3));
#endif

		h4 = _AVX_BROADCAST(&hh[(ldh*3)+i]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(w1, h4, q1);
		q2 = _AVX_NFMA(w2, h4, q2);
		q3 = _AVX_NFMA(w3, h4, q3);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(w1,h4));
		q2 = _AVX_SUB(q2, _AVX_MUL(w2,h4));
		q3 = _AVX_SUB(q3, _AVX_MUL(w3,h4));
#endif

		_AVX_STORE(&q[i*ldq],q1);
		_AVX_STORE(&q[(i*ldq)+offset],q2);
		_AVX_STORE(&q[(i*ldq)+2*offset],q3);
	}

	h1 = _AVX_BROADCAST(&hh[nb-3]);
	q1 = _AVX_LOAD(&q[nb*ldq]);
	q2 = _AVX_LOAD(&q[(nb*ldq)+offset]);
	q3 = _AVX_LOAD(&q[(nb*ldq)+2*offset]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
	q2 = _AVX_NFMA(x2, h1, q2);
	q3 = _AVX_NFMA(x3, h1, q3);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1,h1));
	q2 = _AVX_SUB(q2, _AVX_MUL(x2,h1));
	q3 = _AVX_SUB(q3, _AVX_MUL(x3,h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
	q2 = _AVX_NFMA(y2, h2, q2);
	q3 = _AVX_NFMA(y3, h2, q3);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1,h2));
	q2 = _AVX_SUB(q2, _AVX_MUL(y2,h2));
	q3 = _AVX_SUB(q3, _AVX_MUL(y3,h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
	q2 = _AVX_NFMA(z2, h3, q2);
	q3 = _AVX_NFMA(z3, h3, q3);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1,h3));
	q2 = _AVX_SUB(q2, _AVX_MUL(z2,h3));
	q3 = _AVX_SUB(q3, _AVX_MUL(z3,h3));
#endif
	_AVX_STORE(&q[nb*ldq],q1);
	_AVX_STORE(&q[(nb*ldq)+offset],q2);
	_AVX_STORE(&q[(nb*ldq)+2*offset],q3);

	h1 = _AVX_BROADCAST(&hh[nb-2]);
	q1 = _AVX_LOAD(&q[(nb+1)*ldq]);
	q2 = _AVX_LOAD(&q[((nb+1)*ldq)+offset]);
	q3 = _AVX_LOAD(&q[((nb+1)*ldq)+2*offset]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
	q2 = _AVX_NFMA(x2, h1, q2);
	q3 = _AVX_NFMA(x3, h1, q3);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1,h1));
	q2 = _AVX_SUB(q2, _AVX_MUL(x2,h1));
	q3 = _AVX_SUB(q3, _AVX_MUL(x3,h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
	q2 = _AVX_NFMA(y2, h2, q2);
	q3 = _AVX_NFMA(y3, h2, q3);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1,h2));
	q2 = _AVX_SUB(q2, _AVX_MUL(y2,h2));
	q3 = _AVX_SUB(q3, _AVX_MUL(y3,h2));
#endif
	_AVX_STORE(&q[(nb+1)*ldq],q1);
	_AVX_STORE(&q[((nb+1)*ldq)+offset],q2);
	_AVX_STORE(&q[((nb+1)*ldq)+2*offset],q3);

	h1 = _AVX_BROADCAST(&hh[nb-1]);
	q1 = _AVX_LOAD(&q[(nb+2)*ldq]);
	q2 = _AVX_LOAD(&q[((nb+2)*ldq)+offset]);
	q3 = _AVX_LOAD(&q[((nb+2)*ldq)+2*offset]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
	q2 = _AVX_NFMA(x2, h1, q2);
	q3 = _AVX_NFMA(x3, h1, q3);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1,h1));
	q2 = _AVX_SUB(q2, _AVX_MUL(x2,h1));
	q3 = _AVX_SUB(q3, _AVX_MUL(x3,h1));
#endif
	_AVX_STORE(&q[(nb+2)*ldq],q1);
	_AVX_STORE(&q[((nb+2)*ldq)+offset],q2);
	_AVX_STORE(&q[((nb+2)*ldq)+2*offset],q3);
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
__forceinline void hh_trafo_kernel_8_AVX_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_16_AVX_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__AVX_DATATYPE a1_1 = _AVX_LOAD(&q[ldq*3]);
	__AVX_DATATYPE a2_1 = _AVX_LOAD(&q[ldq*2]);
	__AVX_DATATYPE a3_1 = _AVX_LOAD(&q[ldq]);
	__AVX_DATATYPE a4_1 = _AVX_LOAD(&q[0]);

	__AVX_DATATYPE h_2_1 = _AVX_BROADCAST(&hh[ldh+1]);
	__AVX_DATATYPE h_3_2 = _AVX_BROADCAST(&hh[(ldh*2)+1]);
	__AVX_DATATYPE h_3_1 = _AVX_BROADCAST(&hh[(ldh*2)+2]);
	__AVX_DATATYPE h_4_3 = _AVX_BROADCAST(&hh[(ldh*3)+1]);
	__AVX_DATATYPE h_4_2 = _AVX_BROADCAST(&hh[(ldh*3)+2]);
	__AVX_DATATYPE h_4_1 = _AVX_BROADCAST(&hh[(ldh*3)+3]);

#ifdef __ELPA_USE_FMA__
	__AVX_DATATYPE w1 = _AVX_FMA(a3_1, h_4_3, a4_1);
	w1 = _AVX_FMA(a2_1, h_4_2, w1);
	w1 = _AVX_FMA(a1_1, h_4_1, w1);
	__AVX_DATATYPE z1 = _AVX_FMA(a2_1, h_3_2, a3_1);
	z1 = _AVX_FMA(a1_1, h_3_1, z1);
	__AVX_DATATYPE y1 = _AVX_FMA(a1_1, h_2_1, a2_1);
	__AVX_DATATYPE x1 = a1_1;
#else
	__AVX_DATATYPE w1 = _AVX_ADD(a4_1, _AVX_MUL(a3_1, h_4_3));
	w1 = _AVX_ADD(w1, _AVX_MUL(a2_1, h_4_2));
	w1 = _AVX_ADD(w1, _AVX_MUL(a1_1, h_4_1));
	__AVX_DATATYPE z1 = _AVX_ADD(a3_1, _AVX_MUL(a2_1, h_3_2));
	z1 = _AVX_ADD(z1, _AVX_MUL(a1_1, h_3_1));
	__AVX_DATATYPE y1 = _AVX_ADD(a2_1, _AVX_MUL(a1_1, h_2_1));
	__AVX_DATATYPE x1 = a1_1;
#endif

	__AVX_DATATYPE a1_2 = _AVX_LOAD(&q[(ldq*3)+offset]);
	__AVX_DATATYPE a2_2 = _AVX_LOAD(&q[(ldq*2)+offset]);
	__AVX_DATATYPE a3_2 = _AVX_LOAD(&q[ldq+offset]);
	__AVX_DATATYPE a4_2 = _AVX_LOAD(&q[0+offset]);

#ifdef __ELPA_USE_FMA__
	__AVX_DATATYPE w2 = _AVX_FMA(a3_2, h_4_3, a4_2);
	w2 = _AVX_FMA(a2_2, h_4_2, w2);
	w2 = _AVX_FMA(a1_2, h_4_1, w2);
	__AVX_DATATYPE z2 = _AVX_FMA(a2_2, h_3_2, a3_2);
	z2 = _AVX_FMA(a1_2, h_3_1, z2);
	__AVX_DATATYPE y2 = _AVX_FMA(a1_2, h_2_1, a2_2);
	__AVX_DATATYPE x2 = a1_2;
#else
	__AVX_DATATYPE w2 = _AVX_ADD(a4_2, _AVX_MUL(a3_2, h_4_3));
	w2 = _AVX_ADD(w2, _AVX_MUL(a2_2, h_4_2));
	w2 = _AVX_ADD(w2, _AVX_MUL(a1_2, h_4_1));
	__AVX_DATATYPE z2 = _AVX_ADD(a3_2, _AVX_MUL(a2_2, h_3_2));
	z2 = _AVX_ADD(z2, _AVX_MUL(a1_2, h_3_1));
	__AVX_DATATYPE y2 = _AVX_ADD(a2_2, _AVX_MUL(a1_2, h_2_1));
	__AVX_DATATYPE x2 = a1_2;
#endif

	__AVX_DATATYPE q1;
	__AVX_DATATYPE q2;

	__AVX_DATATYPE h1;
	__AVX_DATATYPE h2;
	__AVX_DATATYPE h3;
	__AVX_DATATYPE h4;

	for(i = 4; i < nb; i++)
	{
		h1 = _AVX_BROADCAST(&hh[i-3]);
		h2 = _AVX_BROADCAST(&hh[ldh+i-2]);
		h3 = _AVX_BROADCAST(&hh[(ldh*2)+i-1]);
		h4 = _AVX_BROADCAST(&hh[(ldh*3)+i]);

		q1 = _AVX_LOAD(&q[i*ldq]);
#ifdef __ELPA_USE_FMA__
		x1 = _AVX_FMA(q1, h1, x1);
		y1 = _AVX_FMA(q1, h2, y1);
		z1 = _AVX_FMA(q1, h3, z1);
		w1 = _AVX_FMA(q1, h4, w1);
#else
		x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
		y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
		z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
		w1 = _AVX_ADD(w1, _AVX_MUL(q1,h4));
#endif

		q2 = _AVX_LOAD(&q[(i*ldq)+offset]);
#ifdef __ELPA_USE_FMA__
		x2 = _AVX_FMA(q2, h1, x2);
		y2 = _AVX_FMA(q2, h2, y2);
		z2 = _AVX_FMA(q2, h3, z2);
		w2 = _AVX_FMA(q2, h4, w2);
#else
		x2 = _AVX_ADD(x2, _AVX_MUL(q2,h1));
		y2 = _AVX_ADD(y2, _AVX_MUL(q2,h2));
		z2 = _AVX_ADD(z2, _AVX_MUL(q2,h3));
		w2 = _AVX_ADD(w2, _AVX_MUL(q2,h4));
#endif
	}

	h1 = _AVX_BROADCAST(&hh[nb-3]);
	h2 = _AVX_BROADCAST(&hh[ldh+nb-2]);
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-1]);

	q1 = _AVX_LOAD(&q[nb*ldq]);
	q2 = _AVX_LOAD(&q[(nb*ldq)+offset]);

#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
	x2 = _AVX_FMA(q2, h1, x2);
	y1 = _AVX_FMA(q1, h2, y1);
	y2 = _AVX_FMA(q2, h2, y2);
	z1 = _AVX_FMA(q1, h3, z1);
	z2 = _AVX_FMA(q2, h3, z2);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
	x2 = _AVX_ADD(x2, _AVX_MUL(q2,h1));
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
	y2 = _AVX_ADD(y2, _AVX_MUL(q2,h2));
	z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
	z2 = _AVX_ADD(z2, _AVX_MUL(q2,h3));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-2]);
	h2 = _AVX_BROADCAST(&hh[(ldh*1)+nb-1]);

	q1 = _AVX_LOAD(&q[(nb+1)*ldq]);
	q2 = _AVX_LOAD(&q[((nb+1)*ldq)+offset]);

#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
	x2 = _AVX_FMA(q2, h1, x2);
	y1 = _AVX_FMA(q1, h2, y1);
	y2 = _AVX_FMA(q2, h2, y2);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
	x2 = _AVX_ADD(x2, _AVX_MUL(q2,h1));
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
	y2 = _AVX_ADD(y2, _AVX_MUL(q2,h2));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-1]);

	q1 = _AVX_LOAD(&q[(nb+2)*ldq]);
	q2 = _AVX_LOAD(&q[((nb+2)*ldq)+offset]);

#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
	x2 = _AVX_FMA(q2, h1, x2);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
	x2 = _AVX_ADD(x2, _AVX_MUL(q2,h1));
#endif

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [8 x nb+3]
	/////////////////////////////////////////////////////

	__AVX_DATATYPE tau1 = _AVX_BROADCAST(&hh[0]);
	__AVX_DATATYPE tau2 = _AVX_BROADCAST(&hh[ldh]);
	__AVX_DATATYPE tau3 = _AVX_BROADCAST(&hh[ldh*2]);
	__AVX_DATATYPE tau4 = _AVX_BROADCAST(&hh[ldh*3]);

	__AVX_DATATYPE vs_1_2 = _AVX_BROADCAST(&s_1_2);
	__AVX_DATATYPE vs_1_3 = _AVX_BROADCAST(&s_1_3);
	__AVX_DATATYPE vs_2_3 = _AVX_BROADCAST(&s_2_3);
	__AVX_DATATYPE vs_1_4 = _AVX_BROADCAST(&s_1_4);
	__AVX_DATATYPE vs_2_4 = _AVX_BROADCAST(&s_2_4);
	__AVX_DATATYPE vs_3_4 = _AVX_BROADCAST(&s_3_4);

	h1 = tau1;
	x1 = _AVX_MUL(x1, h1);
	x2 = _AVX_MUL(x2, h1);

	h1 = tau2;
	h2 = _AVX_MUL(h1, vs_1_2);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMSUB(y1, h1, _AVX_MUL(x1,h2));
	y2 = _AVX_FMSUB(y2, h1, _AVX_MUL(x2,h2));
#else
	y1 = _AVX_SUB(_AVX_MUL(y1,h1), _AVX_MUL(x1,h2));
	y2 = _AVX_SUB(_AVX_MUL(y2,h1), _AVX_MUL(x2,h2));
#endif

	h1 = tau3;
	h2 = _AVX_MUL(h1, vs_1_3);
	h3 = _AVX_MUL(h1, vs_2_3);
#ifdef __ELPA_USE_FMA__
	z1 = _AVX_FMSUB(z1, h1, _AVX_FMA(y1, h3, _AVX_MUL(x1,h2)));
	z2 = _AVX_FMSUB(z2, h1, _AVX_FMA(y2, h3, _AVX_MUL(x2,h2)));
#else
	z1 = _AVX_SUB(_AVX_MUL(z1,h1), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2)));
	z2 = _AVX_SUB(_AVX_MUL(z2,h1), _AVX_ADD(_AVX_MUL(y2,h3), _AVX_MUL(x2,h2)));
#endif

	h1 = tau4;
	h2 = _AVX_MUL(h1, vs_1_4);
	h3 = _AVX_MUL(h1, vs_2_4);
	h4 = _AVX_MUL(h1, vs_3_4);
#ifdef __ELPA_USE_FMA__
	w1 = _AVX_FMSUB(w1, h1, _AVX_FMA(z1, h4, _AVX_FMA(y1, h3, _AVX_MUL(x1,h2))));
	w2 = _AVX_FMSUB(w2, h1, _AVX_FMA(z2, h4, _AVX_FMA(y2, h3, _AVX_MUL(x2,h2))));
#else
	w1 = _AVX_SUB(_AVX_MUL(w1,h1), _AVX_ADD(_AVX_MUL(z1,h4), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2))));
	w2 = _AVX_SUB(_AVX_MUL(w2,h1), _AVX_ADD(_AVX_MUL(z2,h4), _AVX_ADD(_AVX_MUL(y2,h3), _AVX_MUL(x2,h2))));
#endif

	q1 = _AVX_LOAD(&q[0]);
	q2 = _AVX_LOAD(&q[offset]);
	q1 = _AVX_SUB(q1, w1);
	q2 = _AVX_SUB(q2, w2);
	_AVX_STORE(&q[0],q1);
	_AVX_STORE(&q[offset],q2);

	h4 = _AVX_BROADCAST(&hh[(ldh*3)+1]);
	q1 = _AVX_LOAD(&q[ldq]);
	q2 = _AVX_LOAD(&q[ldq+offset]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_SUB(q1, _AVX_FMA(w1, h4, z1));
	q2 = _AVX_SUB(q2, _AVX_FMA(w2, h4, z2));
#else
	q1 = _AVX_SUB(q1, _AVX_ADD(z1, _AVX_MUL(w1, h4)));
	q2 = _AVX_SUB(q2, _AVX_ADD(z2, _AVX_MUL(w2, h4)));
#endif
	_AVX_STORE(&q[ldq],q1);
	_AVX_STORE(&q[ldq+offset],q2);

	h3 = _AVX_BROADCAST(&hh[(ldh*2)+1]);
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+2]);
	q1 = _AVX_LOAD(&q[ldq*2]);
	q2 = _AVX_LOAD(&q[(ldq*2)+offset]);
#ifdef __ELPA_USE_FMA__
        q1 = _AVX_SUB(q1, y1);
        q1 = _AVX_NFMA(z1, h3, q1);
        q1 = _AVX_NFMA(w1, h4, q1);
        q2 = _AVX_SUB(q2, y2);
        q2 = _AVX_NFMA(z2, h3, q2);
        q2 = _AVX_NFMA(w2, h4, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_ADD(y1, _AVX_ADD(_AVX_MUL(z1, h3), _AVX_MUL(w1, h4))));
	q2 = _AVX_SUB(q2, _AVX_ADD(y2, _AVX_ADD(_AVX_MUL(z2, h3), _AVX_MUL(w2, h4))));
#endif
	_AVX_STORE(&q[ldq*2],q1);
	_AVX_STORE(&q[(ldq*2)+offset],q2);

	h2 = _AVX_BROADCAST(&hh[ldh+1]);
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+2]);
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+3]);
	q1 = _AVX_LOAD(&q[ldq*3]);
	q2 = _AVX_LOAD(&q[(ldq*3)+offset]);
#ifdef __ELPA_USE_FMA__
        q1 = _AVX_SUB(q1, x1);
        q1 = _AVX_NFMA(y1, h2, q1);
        q1 = _AVX_NFMA(z1, h3, q1);
        q1 = _AVX_NFMA(w1, h4, q1);
        q2 = _AVX_SUB(q2, x2);
        q2 = _AVX_NFMA(y2, h2, q2);
        q2 = _AVX_NFMA(z2, h3, q2);
        q2 = _AVX_NFMA(w2, h4, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_ADD(x1, _AVX_ADD(_AVX_MUL(y1, h2), _AVX_ADD(_AVX_MUL(z1, h3), _AVX_MUL(w1, h4)))));
	q2 = _AVX_SUB(q2, _AVX_ADD(x2, _AVX_ADD(_AVX_MUL(y2, h2), _AVX_ADD(_AVX_MUL(z2, h3), _AVX_MUL(w2, h4)))));
#endif
	_AVX_STORE(&q[ldq*3], q1);
	_AVX_STORE(&q[(ldq*3)+offset], q2);

	for (i = 4; i < nb; i++)
	{
		h1 = _AVX_BROADCAST(&hh[i-3]);
		h2 = _AVX_BROADCAST(&hh[ldh+i-2]);
		h3 = _AVX_BROADCAST(&hh[(ldh*2)+i-1]);
		h4 = _AVX_BROADCAST(&hh[(ldh*3)+i]);

#ifdef __ELPA_USE_FMA__
		q1 = _AVX_LOAD(&q[i*ldq]);
		q2 = _AVX_LOAD(&q[(i*ldq)+offset]);
                q1 = _AVX_NFMA(x1, h1, q1);
                q1 = _AVX_NFMA(y1, h2, q1);
                q1 = _AVX_NFMA(z1, h3, q1);
                q1 = _AVX_NFMA(w1, h4, q1);
                q2 = _AVX_NFMA(x2, h1, q2);
                q2 = _AVX_NFMA(y2, h2, q2);
                q2 = _AVX_NFMA(z2, h3, q2);
                q2 = _AVX_NFMA(w2, h4, q2);
		_AVX_STORE(&q[i*ldq],q1);
		_AVX_STORE(&q[(i*ldq)+offset],q2);
#else
		q1 = _AVX_LOAD(&q[i*ldq]);
		q1 = _AVX_SUB(q1, _AVX_ADD(_AVX_ADD(_AVX_MUL(w1, h4), _AVX_MUL(z1, h3)), _AVX_ADD(_AVX_MUL(x1,h1), _AVX_MUL(y1, h2))));
		_AVX_STORE(&q[i*ldq],q1);

		q2 = _AVX_LOAD(&q[(i*ldq)+offset]);
		q2 = _AVX_SUB(q2, _AVX_ADD(_AVX_ADD(_AVX_MUL(w2, h4), _AVX_MUL(z2, h3)), _AVX_ADD(_AVX_MUL(x2,h1), _AVX_MUL(y2, h2))));
		_AVX_STORE(&q[(i*ldq)+offset],q2);
#endif
	}

	h1 = _AVX_BROADCAST(&hh[nb-3]);
	h2 = _AVX_BROADCAST(&hh[ldh+nb-2]);
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-1]);
	q1 = _AVX_LOAD(&q[nb*ldq]);
	q2 = _AVX_LOAD(&q[(nb*ldq)+offset]);
#ifdef __ELPA_USE_FMA__
        q1 = _AVX_NFMA(x1, h1, q1);
        q1 = _AVX_NFMA(y1, h2, q1);
        q1 = _AVX_NFMA(z1, h3, q1);
        q2 = _AVX_NFMA(x2, h1, q2);
        q2 = _AVX_NFMA(y2, h2, q2);
        q2 = _AVX_NFMA(z2, h3, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_ADD(_AVX_ADD(_AVX_MUL(z1, h3), _AVX_MUL(y1, h2)) , _AVX_MUL(x1, h1)));
	q2 = _AVX_SUB(q2, _AVX_ADD(_AVX_ADD(_AVX_MUL(z2, h3), _AVX_MUL(y2, h2)) , _AVX_MUL(x2, h1)));
#endif
	_AVX_STORE(&q[nb*ldq],q1);
	_AVX_STORE(&q[(nb*ldq)+offset],q2);

	h1 = _AVX_BROADCAST(&hh[nb-2]);
	h2 = _AVX_BROADCAST(&hh[ldh+nb-1]);
	q1 = _AVX_LOAD(&q[(nb+1)*ldq]);
	q2 = _AVX_LOAD(&q[((nb+1)*ldq)+offset]);
#ifdef __ELPA_USE_FMA__
        q1 = _AVX_NFMA(x1, h1, q1);
        q1 = _AVX_NFMA(y1, h2, q1);
        q2 = _AVX_NFMA(x2, h1, q2);
        q2 = _AVX_NFMA(y2, h2, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_ADD( _AVX_MUL(y1, h2) , _AVX_MUL(x1, h1)));
	q2 = _AVX_SUB(q2, _AVX_ADD( _AVX_MUL(y2, h2) , _AVX_MUL(x2, h1)));
#endif
	_AVX_STORE(&q[(nb+1)*ldq],q1);
	_AVX_STORE(&q[((nb+1)*ldq)+offset],q2);

	h1 = _AVX_BROADCAST(&hh[nb-1]);
	q1 = _AVX_LOAD(&q[(nb+2)*ldq]);
	q2 = _AVX_LOAD(&q[((nb+2)*ldq)+offset]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
	q2 = _AVX_NFMA(x2, h1, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
	q2 = _AVX_SUB(q2, _AVX_MUL(x2, h1));
#endif
	_AVX_STORE(&q[(nb+2)*ldq],q1);
	_AVX_STORE(&q[((nb+2)*ldq)+offset],q2);
}

/**
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 4 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 8 rows of Q simultaneously, a
#endif
 * matrix Vector product with two householder
 * vectors + a rank 1 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_4_AVX_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_8_AVX_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__AVX_DATATYPE a1_1 = _AVX_LOAD(&q[ldq*3]);
	__AVX_DATATYPE a2_1 = _AVX_LOAD(&q[ldq*2]);
	__AVX_DATATYPE a3_1 = _AVX_LOAD(&q[ldq]);
	__AVX_DATATYPE a4_1 = _AVX_LOAD(&q[0]);

	__AVX_DATATYPE h_2_1 = _AVX_BROADCAST(&hh[ldh+1]);
	__AVX_DATATYPE h_3_2 = _AVX_BROADCAST(&hh[(ldh*2)+1]);
	__AVX_DATATYPE h_3_1 = _AVX_BROADCAST(&hh[(ldh*2)+2]);
	__AVX_DATATYPE h_4_3 = _AVX_BROADCAST(&hh[(ldh*3)+1]);
	__AVX_DATATYPE h_4_2 = _AVX_BROADCAST(&hh[(ldh*3)+2]);
	__AVX_DATATYPE h_4_1 = _AVX_BROADCAST(&hh[(ldh*3)+3]);

#ifdef __ELPA_USE_FMA__
	__AVX_DATATYPE w1 = _AVX_FMA(a3_1, h_4_3, a4_1);
	w1 = _AVX_FMA(a2_1, h_4_2, w1);
	w1 = _AVX_FMA(a1_1, h_4_1, w1);
	__AVX_DATATYPE z1 = _AVX_FMA(a2_1, h_3_2, a3_1);
	z1 = _AVX_FMA(a1_1, h_3_1, z1);
	__AVX_DATATYPE y1 = _AVX_FMA(a1_1, h_2_1, a2_1);
	__AVX_DATATYPE x1 = a1_1;
#else
	__AVX_DATATYPE w1 = _AVX_ADD(a4_1, _AVX_MUL(a3_1, h_4_3));
	w1 = _AVX_ADD(w1, _AVX_MUL(a2_1, h_4_2));
	w1 = _AVX_ADD(w1, _AVX_MUL(a1_1, h_4_1));
	__AVX_DATATYPE z1 = _AVX_ADD(a3_1, _AVX_MUL(a2_1, h_3_2));
	z1 = _AVX_ADD(z1, _AVX_MUL(a1_1, h_3_1));
	__AVX_DATATYPE y1 = _AVX_ADD(a2_1, _AVX_MUL(a1_1, h_2_1));
	__AVX_DATATYPE x1 = a1_1;
#endif

	__AVX_DATATYPE q1;

	__AVX_DATATYPE h1;
	__AVX_DATATYPE h2;
	__AVX_DATATYPE h3;
	__AVX_DATATYPE h4;

	for(i = 4; i < nb; i++)
	{
		h1 = _AVX_BROADCAST(&hh[i-3]);
		h2 = _AVX_BROADCAST(&hh[ldh+i-2]);
		h3 = _AVX_BROADCAST(&hh[(ldh*2)+i-1]);
		h4 = _AVX_BROADCAST(&hh[(ldh*3)+i]);

		q1 = _AVX_LOAD(&q[i*ldq]);
#ifdef __ELPA_USE_FMA__
		x1 = _AVX_FMA(q1, h1, x1);
		y1 = _AVX_FMA(q1, h2, y1);
		z1 = _AVX_FMA(q1, h3, z1);
		w1 = _AVX_FMA(q1, h4, w1);
#else
		x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
		y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
		z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
		w1 = _AVX_ADD(w1, _AVX_MUL(q1,h4));
#endif
	}

	h1 = _AVX_BROADCAST(&hh[nb-3]);
	h2 = _AVX_BROADCAST(&hh[ldh+nb-2]);
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-1]);
	q1 = _AVX_LOAD(&q[nb*ldq]);
#ifdef _FMA4__
	x1 = _AVX_FMA(q1, h1, x1);
	y1 = _AVX_FMA(q1, h2, y1);
	z1 = _AVX_FMA(q1, h3, z1);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
	z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-2]);
	h2 = _AVX_BROADCAST(&hh[(ldh*1)+nb-1]);
	q1 = _AVX_LOAD(&q[(nb+1)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
	y1 = _AVX_FMA(q1, h2, y1);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-1]);
	q1 = _AVX_LOAD(&q[(nb+2)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
#endif

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [4 x nb+3]
	/////////////////////////////////////////////////////

	__AVX_DATATYPE tau1 = _AVX_BROADCAST(&hh[0]);
	__AVX_DATATYPE tau2 = _AVX_BROADCAST(&hh[ldh]);
	__AVX_DATATYPE tau3 = _AVX_BROADCAST(&hh[ldh*2]);
	__AVX_DATATYPE tau4 = _AVX_BROADCAST(&hh[ldh*3]);

	__AVX_DATATYPE vs_1_2 = _AVX_BROADCAST(&s_1_2);
	__AVX_DATATYPE vs_1_3 = _AVX_BROADCAST(&s_1_3);
	__AVX_DATATYPE vs_2_3 = _AVX_BROADCAST(&s_2_3);
	__AVX_DATATYPE vs_1_4 = _AVX_BROADCAST(&s_1_4);
	__AVX_DATATYPE vs_2_4 = _AVX_BROADCAST(&s_2_4);
	__AVX_DATATYPE vs_3_4 = _AVX_BROADCAST(&s_3_4);

	h1 = tau1;
	x1 = _AVX_MUL(x1, h1);

	h1 = tau2;
	h2 = _AVX_MUL(h1, vs_1_2);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMSUB(y1, h1, _AVX_MUL(x1,h2));
#else
	y1 = _AVX_SUB(_AVX_MUL(y1,h1), _AVX_MUL(x1,h2));
#endif

	h1 = tau3;
	h2 = _AVX_MUL(h1, vs_1_3);
	h3 = _AVX_MUL(h1, vs_2_3);
#ifdef __ELPA_USE_FMA__
	z1 = _AVX_FMSUB(z1, h1, _AVX_FMA(y1, h3, _AVX_MUL(x1,h2)));
#else
	z1 = _AVX_SUB(_AVX_MUL(z1,h1), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2)));
#endif

	h1 = tau4;
	h2 = _AVX_MUL(h1, vs_1_4);
	h3 = _AVX_MUL(h1, vs_2_4);
	h4 = _AVX_MUL(h1, vs_3_4);
#ifdef __ELPA_USE_FMA__
	w1 = _AVX_FMSUB(w1, h1, _AVX_FMA(z1, h4, _AVX_FMA(y1, h3, _AVX_MUL(x1,h2))));
#else
	w1 = _AVX_SUB(_AVX_MUL(w1,h1), _AVX_ADD(_AVX_MUL(z1,h4), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2))));
#endif

	q1 = _AVX_LOAD(&q[0]);
	q1 = _AVX_SUB(q1, w1);
	_AVX_STORE(&q[0],q1);

	h4 = _AVX_BROADCAST(&hh[(ldh*3)+1]);
	q1 = _AVX_LOAD(&q[ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_SUB(q1, _AVX_FMA(w1, h4, z1));
#else
	q1 = _AVX_SUB(q1, _AVX_ADD(z1, _AVX_MUL(w1, h4)));
#endif
	_AVX_STORE(&q[ldq],q1);

	h3 = _AVX_BROADCAST(&hh[(ldh*2)+1]);
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+2]);
	q1 = _AVX_LOAD(&q[ldq*2]);
#ifdef __ELPA_USE_FMA__
        q1 = _AVX_SUB(q1, y1);
        q1 = _AVX_NFMA(z1, h3, q1);
        q1 = _AVX_NFMA(w1, h4, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_ADD(y1, _AVX_ADD(_AVX_MUL(z1, h3), _AVX_MUL(w1, h4))));
#endif
	_AVX_STORE(&q[ldq*2],q1);

	h2 = _AVX_BROADCAST(&hh[ldh+1]);
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+2]);
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+3]);
	q1 = _AVX_LOAD(&q[ldq*3]);
#ifdef __ELPA_USE_FMA__
        q1 = _AVX_SUB(q1, x1);
        q1 = _AVX_NFMA(y1, h2, q1);
        q1 = _AVX_NFMA(z1, h3, q1);
        q1 = _AVX_NFMA(w1, h4, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_ADD(x1, _AVX_ADD(_AVX_MUL(y1, h2), _AVX_ADD(_AVX_MUL(z1, h3), _AVX_MUL(w1, h4)))));
#endif
	_AVX_STORE(&q[ldq*3], q1);

	for (i = 4; i < nb; i++)
	{
		h1 = _AVX_BROADCAST(&hh[i-3]);
		h2 = _AVX_BROADCAST(&hh[ldh+i-2]);
		h3 = _AVX_BROADCAST(&hh[(ldh*2)+i-1]);
		h4 = _AVX_BROADCAST(&hh[(ldh*3)+i]);

		q1 = _AVX_LOAD(&q[i*ldq]);
#ifdef __ELPA_USE_FMA__
                q1 = _AVX_NFMA(x1, h1, q1);
                q1 = _AVX_NFMA(y1, h2, q1);
                q1 = _AVX_NFMA(z1, h3, q1);
                q1 = _AVX_NFMA(w1, h4, q1);
#else
		q1 = _AVX_SUB(q1, _AVX_ADD(_AVX_ADD(_AVX_MUL(w1, h4), _AVX_MUL(z1, h3)), _AVX_ADD(_AVX_MUL(x1,h1), _AVX_MUL(y1, h2))));
#endif
		_AVX_STORE(&q[i*ldq],q1);
	}

	h1 = _AVX_BROADCAST(&hh[nb-3]);
	h2 = _AVX_BROADCAST(&hh[ldh+nb-2]);
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-1]);
	q1 = _AVX_LOAD(&q[nb*ldq]);
#ifdef __ELPA_USE_FMA__
        q1 = _AVX_NFMA(x1, h1, q1);
        q1 = _AVX_NFMA(y1, h2, q1);
        q1 = _AVX_NFMA(z1, h3, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_ADD(_AVX_ADD(_AVX_MUL(z1, h3), _AVX_MUL(y1, h2)) , _AVX_MUL(x1, h1)));
#endif
	_AVX_STORE(&q[nb*ldq],q1);

	h1 = _AVX_BROADCAST(&hh[nb-2]);
	h2 = _AVX_BROADCAST(&hh[ldh+nb-1]);
	q1 = _AVX_LOAD(&q[(nb+1)*ldq]);
#ifdef __ELPA_USE_FMA__
        q1 = _AVX_NFMA(x1, h1, q1);
        q1 = _AVX_NFMA(y1, h2, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_ADD( _AVX_MUL(y1, h2) , _AVX_MUL(x1, h1)));
#endif
	_AVX_STORE(&q[(nb+1)*ldq],q1);

	h1 = _AVX_BROADCAST(&hh[nb-1]);
	q1 = _AVX_LOAD(&q[(nb+2)*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
#endif
	_AVX_STORE(&q[(nb+2)*ldq],q1);
}


#if 0
#ifdef SINGLE_PRECISION_REAL
/**
 * Unrolled kernel that computes
 * 4 rows of Q simultaneously, a
 * matrix Vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_4_AVX_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__AVX_DATATYPE a1_1 = _mm256_castps128_ps256(_mm_load_ps(&q[ldq*3]));
	__AVX_DATATYPE a2_1 = _mm256_castps128_ps256(_mm_load_ps(&q[ldq*2]));
	__AVX_DATATYPE a3_1 = _mm256_castps128_ps256(_mm_load_ps(&q[ldq]));
        __AVX_DATATYPE a4_1 = _mm256_castps128_ps256(_mm_load_ps(&q[0])); // q(1,1) | q(2,1) | q(3,1) | q(4,1) |

	__AVX_DATATYPE h_2_1 = _AVX_BROADCAST(&hh[ldh+1]);
	__AVX_DATATYPE h_3_2 = _AVX_BROADCAST(&hh[(ldh*2)+1]);
	__AVX_DATATYPE h_3_1 = _AVX_BROADCAST(&hh[(ldh*2)+2]);
	__AVX_DATATYPE h_4_3 = _AVX_BROADCAST(&hh[(ldh*3)+1]);
	__AVX_DATATYPE h_4_2 = _AVX_BROADCAST(&hh[(ldh*3)+2]);
	__AVX_DATATYPE h_4_1 = _AVX_BROADCAST(&hh[(ldh*3)+3]);

#ifdef __ELPA_USE_FMA__
	__AVX_DATATYPE w1 = _AVX_FMA(a3_1, h_4_3, a4_1);
	w1 = _AVX_FMA(a2_1, h_4_2, w1);
	w1 = _AVX_FMA(a1_1, h_4_1, w1);
	__AVX_DATATYPE z1 = _AVX_FMA(a2_1, h_3_2, a3_1);
	z1 = _AVX_FMA(a1_1, h_3_1, z1);
	__AVX_DATATYPE y1 = _AVX_FMA(a1_1, h_2_1, a2_1);
	__AVX_DATATYPE x1 = a1_1;
#else
	__AVX_DATATYPE w1 = _AVX_ADD(a4_1, _AVX_MUL(a3_1, h_4_3));
	w1 = _AVX_ADD(w1, _AVX_MUL(a2_1, h_4_2));
	w1 = _AVX_ADD(w1, _AVX_MUL(a1_1, h_4_1));
	__AVX_DATATYPE z1 = _AVX_ADD(a3_1, _AVX_MUL(a2_1, h_3_2));
	z1 = _AVX_ADD(z1, _AVX_MUL(a1_1, h_3_1));
	__AVX_DATATYPE y1 = _AVX_ADD(a2_1, _AVX_MUL(a1_1, h_2_1));
	__AVX_DATATYPE x1 = a1_1;
#endif

	__AVX_DATATYPE q1;

	__AVX_DATATYPE h1;
	__AVX_DATATYPE h2;
	__AVX_DATATYPE h3;
	__AVX_DATATYPE h4;

	for(i = 4; i < nb; i++)
	{
		h1 = _AVX_BROADCAST(&hh[i-3]);
		h2 = _AVX_BROADCAST(&hh[ldh+i-2]);
		h3 = _AVX_BROADCAST(&hh[(ldh*2)+i-1]);
		h4 = _AVX_BROADCAST(&hh[(ldh*3)+i]);

		q1 = _AVX_LOAD(&q[i*ldq]);
#ifdef __ELPA_USE_FMA__
		x1 = _AVX_FMA(q1, h1, x1);
		y1 = _AVX_FMA(q1, h2, y1);
		z1 = _AVX_FMA(q1, h3, z1);
		w1 = _AVX_FMA(q1, h4, w1);
#else
		x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
		y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
		z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
		w1 = _AVX_ADD(w1, _AVX_MUL(q1,h4));
#endif
	}

	h1 = _AVX_BROADCAST(&hh[nb-3]);
	h2 = _AVX_BROADCAST(&hh[ldh+nb-2]);
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-1]);
	q1 = _AVX_LOAD(&q[nb*ldq]);
#ifdef _FMA4__
	x1 = _AVX_FMA(q1, h1, x1);
	y1 = _AVX_FMA(q1, h2, y1);
	z1 = _AVX_FMA(q1, h3, z1);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
	z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-2]);
	h2 = _AVX_BROADCAST(&hh[(ldh*1)+nb-1]);
	q1 = _AVX_LOAD(&q[(nb+1)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
	y1 = _AVX_FMA(q1, h2, y1);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-1]);
	q1 = _AVX_LOAD(&q[(nb+2)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
#endif

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [4 x nb+3]
	/////////////////////////////////////////////////////

	__AVX_DATATYPE tau1 = _AVX_BROADCAST(&hh[0]);
	__AVX_DATATYPE tau2 = _AVX_BROADCAST(&hh[ldh]);
	__AVX_DATATYPE tau3 = _AVX_BROADCAST(&hh[ldh*2]);
	__AVX_DATATYPE tau4 = _AVX_BROADCAST(&hh[ldh*3]);

	__AVX_DATATYPE vs_1_2 = _AVX_BROADCAST(&s_1_2);
	__AVX_DATATYPE vs_1_3 = _AVX_BROADCAST(&s_1_3);
	__AVX_DATATYPE vs_2_3 = _AVX_BROADCAST(&s_2_3);
	__AVX_DATATYPE vs_1_4 = _AVX_BROADCAST(&s_1_4);
	__AVX_DATATYPE vs_2_4 = _AVX_BROADCAST(&s_2_4);
	__AVX_DATATYPE vs_3_4 = _AVX_BROADCAST(&s_3_4);

	h1 = tau1;
	x1 = _AVX_MUL(x1, h1);

	h1 = tau2;
	h2 = _AVX_MUL(h1, vs_1_2);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMSUB(y1, h1, _AVX_MUL(x1,h2));
#else
	y1 = _AVX_SUB(_AVX_MUL(y1,h1), _AVX_MUL(x1,h2));
#endif

	h1 = tau3;
	h2 = _AVX_MUL(h1, vs_1_3);
	h3 = _AVX_MUL(h1, vs_2_3);
#ifdef __ELPA_USE_FMA__
	z1 = _AVX_FMSUB(z1, h1, _AVX_FMA(y1, h3, _AVX_MUL(x1,h2)));
#else
	z1 = _AVX_SUB(_AVX_MUL(z1,h1), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2)));
#endif

	h1 = tau4;
	h2 = _AVX_MUL(h1, vs_1_4);
	h3 = _AVX_MUL(h1, vs_2_4);
	h4 = _AVX_MUL(h1, vs_3_4);
#ifdef __ELPA_USE_FMA__
	w1 = _AVX_FMSUB(w1, h1, _AVX_FMA(z1, h4, _AVX_FMA(y1, h3, _AVX_MUL(x1,h2))));
#else
	w1 = _AVX_SUB(_AVX_MUL(w1,h1), _AVX_ADD(_AVX_MUL(z1,h4), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2))));
#endif

	q1 = _AVX_LOAD(&q[0]);
	q1 = _AVX_SUB(q1, w1);
	_mm_store_ps(&q[0],  _mm256_castps256_ps128(q1));
//	_AVX_STORE(&q[0],q1);

	h4 = _AVX_BROADCAST(&hh[(ldh*3)+1]);
	q1 = _AVX_LOAD(&q[ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_SUB(q1, _AVX_FMA(w1, h4, z1));
#else
	q1 = _AVX_SUB(q1, _AVX_ADD(z1, _AVX_MUL(w1, h4)));
#endif
	_mm_store_ps(&q[ldq],  _mm256_castps256_ps128(q1));
//	_AVX_STORE(&q[ldq],q1);

	h3 = _AVX_BROADCAST(&hh[(ldh*2)+1]);
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+2]);
	q1 = _AVX_LOAD(&q[ldq*2]);
#ifdef __ELPA_USE_FMA__
        q1 = _AVX_SUB(q1, y1);
        q1 = _AVX_NFMA(z1, h3, q1);
        q1 = _AVX_NFMA(w1, h4, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_ADD(y1, _AVX_ADD(_AVX_MUL(z1, h3), _AVX_MUL(w1, h4))));
#endif
	_mm_store_ps(&q[ldq*2],  _mm256_castps256_ps128(q1));
//	_AVX_STORE(&q[ldq*2],q1);

	h2 = _AVX_BROADCAST(&hh[ldh+1]);
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+2]);
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+3]);
	q1 = _AVX_LOAD(&q[ldq*3]);
#ifdef __ELPA_USE_FMA__
        q1 = _AVX_SUB(q1, x1);
        q1 = _AVX_NFMA(y1, h2, q1);
        q1 = _AVX_NFMA(z1, h3, q1);
        q1 = _AVX_NFMA(w1, h4, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_ADD(x1, _AVX_ADD(_AVX_MUL(y1, h2), _AVX_ADD(_AVX_MUL(z1, h3), _AVX_MUL(w1, h4)))));
#endif
	_mm_store_ps(&q[ldq*3],  _mm256_castps256_ps128(q1));
//	_AVX_STORE(&q[ldq*3], q1);

	for (i = 4; i < nb; i++)
	{
		h1 = _AVX_BROADCAST(&hh[i-3]);
		h2 = _AVX_BROADCAST(&hh[ldh+i-2]);
		h3 = _AVX_BROADCAST(&hh[(ldh*2)+i-1]);
		h4 = _AVX_BROADCAST(&hh[(ldh*3)+i]);

		q1 = _AVX_LOAD(&q[i*ldq]);
#ifdef __ELPA_USE_FMA__
                q1 = _AVX_NFMA(x1, h1, q1);
                q1 = _AVX_NFMA(y1, h2, q1);
                q1 = _AVX_NFMA(z1, h3, q1);
                q1 = _AVX_NFMA(w1, h4, q1);
#else
		q1 = _AVX_SUB(q1, _AVX_ADD(_AVX_ADD(_AVX_MUL(w1, h4), _AVX_MUL(z1, h3)), _AVX_ADD(_AVX_MUL(x1,h1), _AVX_MUL(y1, h2))));
#endif
	        _mm_store_ps(&q[ldq*i],  _mm256_castps256_ps128(q1));
//		_AVX_STORE(&q[i*ldq],q1);
	}

	h1 = _AVX_BROADCAST(&hh[nb-3]);
	h2 = _AVX_BROADCAST(&hh[ldh+nb-2]);
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-1]);
	q1 = _AVX_LOAD(&q[nb*ldq]);
#ifdef __ELPA_USE_FMA__
        q1 = _AVX_NFMA(x1, h1, q1);
        q1 = _AVX_NFMA(y1, h2, q1);
        q1 = _AVX_NFMA(z1, h3, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_ADD(_AVX_ADD(_AVX_MUL(z1, h3), _AVX_MUL(y1, h2)) , _AVX_MUL(x1, h1)));
#endif
        _mm_store_ps(&q[ldq*nb],  _mm256_castps256_ps128(q1));
//	_AVX_STORE(&q[nb*ldq],q1);

	h1 = _AVX_BROADCAST(&hh[nb-2]);
	h2 = _AVX_BROADCAST(&hh[ldh+nb-1]);
	q1 = _AVX_LOAD(&q[(nb+1)*ldq]);
#ifdef __ELPA_USE_FMA__
        q1 = _AVX_NFMA(x1, h1, q1);
        q1 = _AVX_NFMA(y1, h2, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_ADD( _AVX_MUL(y1, h2) , _AVX_MUL(x1, h1)));
#endif
        _mm_store_ps(&q[ldq*(nb+1)],  _mm256_castps256_ps128(q1));
//	_AVX_STORE(&q[(nb+1)*ldq],q1);

	h1 = _AVX_BROADCAST(&hh[nb-1]);
	q1 = _AVX_LOAD(&q[(nb+2)*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
#endif
        _mm_store_ps(&q[ldq*(nb+2)],  _mm256_castps256_ps128(q1));
//	_AVX_STORE(&q[(nb+2)*ldq],q1);
}
#endif /* SINGLE_PRECISION_REAL */
#endif
