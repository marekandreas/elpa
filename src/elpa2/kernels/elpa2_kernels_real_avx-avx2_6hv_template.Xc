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
#endif /* DOUBLE_PRECISION_REAL */

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
#endif /* SINGLE_PRECISION_REAL */

#ifdef DOUBLE_PRECISION_REAL
//Forward declaration
static void hh_trafo_kernel_4_AVX_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);
static void hh_trafo_kernel_8_AVX_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);

void hexa_hh_trafo_real_avx_avx2_6hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif

#ifdef SINGLE_PRECISION_REAL
//Forward declaration
static void hh_trafo_kernel_4_AVX_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);
static void hh_trafo_kernel_8_AVX_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);
static void hh_trafo_kernel_16_AVX_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);

void hexa_hh_trafo_real_avx_avx2_6hv_single_(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);

#endif

#ifdef DOUBLE_PRECISION_REAL
/*
!f>#if defined(HAVE_AVX) || defined(HAVE_AVX2)
!f> interface
!f>   subroutine hexa_hh_trafo_real_avx_avx2_6hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="hexa_hh_trafo_real_avx_avx2_6hv_double")
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
!f>   subroutine hexa_hh_trafo_real_avx_avx2_6hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="hexa_hh_trafo_real_avx_avx2_6hv_single")
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
void hexa_hh_trafo_real_avx_avx2_6hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
#ifdef SINGLE_PRECISION_REAL
void hexa_hh_trafo_real_avx_avx2_6hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;

	// calculating scalar products to compute
	// 6 householder vectors simultaneously
#ifdef DOUBLE_PRECISION_REAL
	double scalarprods[15];
#endif
#ifdef SINGLE_PRECISION_REAL
	float scalarprods[15];
#endif

	scalarprods[0] = hh[(ldh+1)];
	scalarprods[1] = hh[(ldh*2)+2];
	scalarprods[2] = hh[(ldh*2)+1];
	scalarprods[3] = hh[(ldh*3)+3];
	scalarprods[4] = hh[(ldh*3)+2];
	scalarprods[5] = hh[(ldh*3)+1];
	scalarprods[6] = hh[(ldh*4)+4];
	scalarprods[7] = hh[(ldh*4)+3];
	scalarprods[8] = hh[(ldh*4)+2];
	scalarprods[9] = hh[(ldh*4)+1];
	scalarprods[10] = hh[(ldh*5)+5];
	scalarprods[11] = hh[(ldh*5)+4];
	scalarprods[12] = hh[(ldh*5)+3];
	scalarprods[13] = hh[(ldh*5)+2];
	scalarprods[14] = hh[(ldh*5)+1];

	// calculate scalar product of first and fourth householder Vector
	// loop counter = 2
	scalarprods[0] += hh[1] * hh[(2+ldh)];
	scalarprods[2] += hh[(ldh)+1] * hh[2+(ldh*2)];
	scalarprods[5] += hh[(ldh*2)+1] * hh[2+(ldh*3)];
	scalarprods[9] += hh[(ldh*3)+1] * hh[2+(ldh*4)];
	scalarprods[14] += hh[(ldh*4)+1] * hh[2+(ldh*5)];

	// loop counter = 3
	scalarprods[0] += hh[2] * hh[(3+ldh)];
	scalarprods[2] += hh[(ldh)+2] * hh[3+(ldh*2)];
	scalarprods[5] += hh[(ldh*2)+2] * hh[3+(ldh*3)];
	scalarprods[9] += hh[(ldh*3)+2] * hh[3+(ldh*4)];
	scalarprods[14] += hh[(ldh*4)+2] * hh[3+(ldh*5)];

	scalarprods[1] += hh[1] * hh[3+(ldh*2)];
	scalarprods[4] += hh[(ldh*1)+1] * hh[3+(ldh*3)];
	scalarprods[8] += hh[(ldh*2)+1] * hh[3+(ldh*4)];
	scalarprods[13] += hh[(ldh*3)+1] * hh[3+(ldh*5)];

	// loop counter = 4
	scalarprods[0] += hh[3] * hh[(4+ldh)];
	scalarprods[2] += hh[(ldh)+3] * hh[4+(ldh*2)];
	scalarprods[5] += hh[(ldh*2)+3] * hh[4+(ldh*3)];
	scalarprods[9] += hh[(ldh*3)+3] * hh[4+(ldh*4)];
	scalarprods[14] += hh[(ldh*4)+3] * hh[4+(ldh*5)];

	scalarprods[1] += hh[2] * hh[4+(ldh*2)];
	scalarprods[4] += hh[(ldh*1)+2] * hh[4+(ldh*3)];
	scalarprods[8] += hh[(ldh*2)+2] * hh[4+(ldh*4)];
	scalarprods[13] += hh[(ldh*3)+2] * hh[4+(ldh*5)];

	scalarprods[3] += hh[1] * hh[4+(ldh*3)];
	scalarprods[7] += hh[(ldh)+1] * hh[4+(ldh*4)];
	scalarprods[12] += hh[(ldh*2)+1] * hh[4+(ldh*5)];

	// loop counter = 5
	scalarprods[0] += hh[4] * hh[(5+ldh)];
	scalarprods[2] += hh[(ldh)+4] * hh[5+(ldh*2)];
	scalarprods[5] += hh[(ldh*2)+4] * hh[5+(ldh*3)];
	scalarprods[9] += hh[(ldh*3)+4] * hh[5+(ldh*4)];
	scalarprods[14] += hh[(ldh*4)+4] * hh[5+(ldh*5)];

	scalarprods[1] += hh[3] * hh[5+(ldh*2)];
	scalarprods[4] += hh[(ldh*1)+3] * hh[5+(ldh*3)];
	scalarprods[8] += hh[(ldh*2)+3] * hh[5+(ldh*4)];
	scalarprods[13] += hh[(ldh*3)+3] * hh[5+(ldh*5)];

	scalarprods[3] += hh[2] * hh[5+(ldh*3)];
	scalarprods[7] += hh[(ldh)+2] * hh[5+(ldh*4)];
	scalarprods[12] += hh[(ldh*2)+2] * hh[5+(ldh*5)];

	scalarprods[6] += hh[1] * hh[5+(ldh*4)];
	scalarprods[11] += hh[(ldh)+1] * hh[5+(ldh*5)];

	#pragma ivdep
	for (i = 6; i < nb; i++)
	{
		scalarprods[0] += hh[i-1] * hh[(i+ldh)];
		scalarprods[2] += hh[(ldh)+i-1] * hh[i+(ldh*2)];
		scalarprods[5] += hh[(ldh*2)+i-1] * hh[i+(ldh*3)];
		scalarprods[9] += hh[(ldh*3)+i-1] * hh[i+(ldh*4)];
		scalarprods[14] += hh[(ldh*4)+i-1] * hh[i+(ldh*5)];

		scalarprods[1] += hh[i-2] * hh[i+(ldh*2)];
		scalarprods[4] += hh[(ldh*1)+i-2] * hh[i+(ldh*3)];
		scalarprods[8] += hh[(ldh*2)+i-2] * hh[i+(ldh*4)];
		scalarprods[13] += hh[(ldh*3)+i-2] * hh[i+(ldh*5)];

		scalarprods[3] += hh[i-3] * hh[i+(ldh*3)];
		scalarprods[7] += hh[(ldh)+i-3] * hh[i+(ldh*4)];
		scalarprods[12] += hh[(ldh*2)+i-3] * hh[i+(ldh*5)];

		scalarprods[6] += hh[i-4] * hh[i+(ldh*4)];
		scalarprods[11] += hh[(ldh)+i-4] * hh[i+(ldh*5)];

		scalarprods[10] += hh[i-5] * hh[i+(ldh*5)];
	}

	// Production level kernel calls with padding
#ifdef DOUBLE_PRECISION_REAL
	for (i = 0; i < nq-4; i+=8)
	{
		hh_trafo_kernel_8_AVX_6hv_double(&q[i], hh, nb, ldq, ldh, scalarprods);
	}
#endif
#ifdef SINGLE_PRECISION_REAL
	for (i = 0; i < nq-8; i+=16)
	{
		hh_trafo_kernel_16_AVX_6hv_single(&q[i], hh, nb, ldq, ldh, scalarprods);
	}
#endif
	if (nq == i)
	{
		return;
	}
	else
	{
#ifdef DOUBLE_PRECISION_REAL
		hh_trafo_kernel_4_AVX_6hv_double(&q[i], hh, nb, ldq, ldh, scalarprods);
#endif
#ifdef SINGLE_PRECISION_REAL
		hh_trafo_kernel_8_AVX_6hv_single(&q[i], hh, nb, ldq, ldh, scalarprods);
#endif
	}
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
__forceinline void hh_trafo_kernel_8_AVX_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_16_AVX_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [8 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__AVX_DATATYPE a1_1 = _AVX_LOAD(&q[ldq*5]);
	__AVX_DATATYPE a2_1 = _AVX_LOAD(&q[ldq*4]);
	__AVX_DATATYPE a3_1 = _AVX_LOAD(&q[ldq*3]);
	__AVX_DATATYPE a4_1 = _AVX_LOAD(&q[ldq*2]);
	__AVX_DATATYPE a5_1 = _AVX_LOAD(&q[ldq]);
	__AVX_DATATYPE a6_1 = _AVX_LOAD(&q[0]);

	__AVX_DATATYPE h_6_5 = _AVX_BROADCAST(&hh[(ldh*5)+1]);
	__AVX_DATATYPE h_6_4 = _AVX_BROADCAST(&hh[(ldh*5)+2]);
	__AVX_DATATYPE h_6_3 = _AVX_BROADCAST(&hh[(ldh*5)+3]);
	__AVX_DATATYPE h_6_2 = _AVX_BROADCAST(&hh[(ldh*5)+4]);
	__AVX_DATATYPE h_6_1 = _AVX_BROADCAST(&hh[(ldh*5)+5]);
#ifdef __ELPA_USE_FMA__
	register __AVX_DATATYPE t1 = _AVX_FMA(a5_1, h_6_5, a6_1);
	t1 = _AVX_FMA(a4_1, h_6_4, t1);
	t1 = _AVX_FMA(a3_1, h_6_3, t1);
	t1 = _AVX_FMA(a2_1, h_6_2, t1);
	t1 = _AVX_FMA(a1_1, h_6_1, t1);
#else
	register __AVX_DATATYPE t1 = _AVX_ADD(a6_1, _AVX_MUL(a5_1, h_6_5));
	t1 = _AVX_ADD(t1, _AVX_MUL(a4_1, h_6_4));
	t1 = _AVX_ADD(t1, _AVX_MUL(a3_1, h_6_3));
	t1 = _AVX_ADD(t1, _AVX_MUL(a2_1, h_6_2));
	t1 = _AVX_ADD(t1, _AVX_MUL(a1_1, h_6_1));
#endif
	__AVX_DATATYPE h_5_4 = _AVX_BROADCAST(&hh[(ldh*4)+1]);
	__AVX_DATATYPE h_5_3 = _AVX_BROADCAST(&hh[(ldh*4)+2]);
	__AVX_DATATYPE h_5_2 = _AVX_BROADCAST(&hh[(ldh*4)+3]);
	__AVX_DATATYPE h_5_1 = _AVX_BROADCAST(&hh[(ldh*4)+4]);
#ifdef __ELPA_USE_FMA__
	register __AVX_DATATYPE v1 = _AVX_FMA(a4_1, h_5_4, a5_1);
	v1 = _AVX_FMA(a3_1, h_5_3, v1);
	v1 = _AVX_FMA(a2_1, h_5_2, v1);
	v1 = _AVX_FMA(a1_1, h_5_1, v1);
#else
	register __AVX_DATATYPE v1 = _AVX_ADD(a5_1, _AVX_MUL(a4_1, h_5_4));
	v1 = _AVX_ADD(v1, _AVX_MUL(a3_1, h_5_3));
	v1 = _AVX_ADD(v1, _AVX_MUL(a2_1, h_5_2));
	v1 = _AVX_ADD(v1, _AVX_MUL(a1_1, h_5_1));
#endif
	__AVX_DATATYPE h_4_3 = _AVX_BROADCAST(&hh[(ldh*3)+1]);
	__AVX_DATATYPE h_4_2 = _AVX_BROADCAST(&hh[(ldh*3)+2]);
	__AVX_DATATYPE h_4_1 = _AVX_BROADCAST(&hh[(ldh*3)+3]);
#ifdef __ELPA_USE_FMA__
	register __AVX_DATATYPE w1 = _AVX_FMA(a3_1, h_4_3, a4_1);
	w1 = _AVX_FMA(a2_1, h_4_2, w1);
	w1 = _AVX_FMA(a1_1, h_4_1, w1);
#else
	register __AVX_DATATYPE w1 = _AVX_ADD(a4_1, _AVX_MUL(a3_1, h_4_3));
	w1 = _AVX_ADD(w1, _AVX_MUL(a2_1, h_4_2));
	w1 = _AVX_ADD(w1, _AVX_MUL(a1_1, h_4_1));
#endif
	__AVX_DATATYPE h_2_1 = _AVX_BROADCAST(&hh[ldh+1]);
	__AVX_DATATYPE h_3_2 = _AVX_BROADCAST(&hh[(ldh*2)+1]);
	__AVX_DATATYPE h_3_1 = _AVX_BROADCAST(&hh[(ldh*2)+2]);
#ifdef __ELPA_USE_FMA__
	register __AVX_DATATYPE z1 = _AVX_FMA(a2_1, h_3_2, a3_1);
	z1 = _AVX_FMA(a1_1, h_3_1, z1);
	register __AVX_DATATYPE y1 = _AVX_FMA(a1_1, h_2_1, a2_1);
#else
	register __AVX_DATATYPE z1 = _AVX_ADD(a3_1, _AVX_MUL(a2_1, h_3_2));
	z1 = _AVX_ADD(z1, _AVX_MUL(a1_1, h_3_1));
	register __AVX_DATATYPE y1 = _AVX_ADD(a2_1, _AVX_MUL(a1_1, h_2_1));
#endif
	register __AVX_DATATYPE x1 = a1_1;

	__AVX_DATATYPE a1_2 = _AVX_LOAD(&q[(ldq*5)+offset]);
	__AVX_DATATYPE a2_2 = _AVX_LOAD(&q[(ldq*4)+offset]);
	__AVX_DATATYPE a3_2 = _AVX_LOAD(&q[(ldq*3)+offset]);
	__AVX_DATATYPE a4_2 = _AVX_LOAD(&q[(ldq*2)+offset]);
	__AVX_DATATYPE a5_2 = _AVX_LOAD(&q[(ldq)+offset]);
	__AVX_DATATYPE a6_2 = _AVX_LOAD(&q[offset]);

#ifdef __ELPA_USE_FMA__
	register __AVX_DATATYPE t2 = _AVX_FMA(a5_2, h_6_5, a6_2);
	t2 = _AVX_FMA(a4_2, h_6_4, t2);
	t2 = _AVX_FMA(a3_2, h_6_3, t2);
	t2 = _AVX_FMA(a2_2, h_6_2, t2);
	t2 = _AVX_FMA(a1_2, h_6_1, t2);
	register __AVX_DATATYPE v2 = _AVX_FMA(a4_2, h_5_4, a5_2);
	v2 = _AVX_FMA(a3_2, h_5_3, v2);
	v2 = _AVX_FMA(a2_2, h_5_2, v2);
	v2 = _AVX_FMA(a1_2, h_5_1, v2);
	register __AVX_DATATYPE w2 = _AVX_FMA(a3_2, h_4_3, a4_2);
	w2 = _AVX_FMA(a2_2, h_4_2, w2);
	w2 = _AVX_FMA(a1_2, h_4_1, w2);
	register __AVX_DATATYPE z2 = _AVX_FMA(a2_2, h_3_2, a3_2);
	z2 = _AVX_FMA(a1_2, h_3_1, z2);
	register __AVX_DATATYPE y2 = _AVX_FMA(a1_2, h_2_1, a2_2);
#else
	register __AVX_DATATYPE t2 = _AVX_ADD(a6_2, _AVX_MUL(a5_2, h_6_5));
	t2 = _AVX_ADD(t2, _AVX_MUL(a4_2, h_6_4));
	t2 = _AVX_ADD(t2, _AVX_MUL(a3_2, h_6_3));
	t2 = _AVX_ADD(t2, _AVX_MUL(a2_2, h_6_2));
	t2 = _AVX_ADD(t2, _AVX_MUL(a1_2, h_6_1));
	register __AVX_DATATYPE v2 = _AVX_ADD(a5_2, _AVX_MUL(a4_2, h_5_4));
	v2 = _AVX_ADD(v2, _AVX_MUL(a3_2, h_5_3));
	v2 = _AVX_ADD(v2, _AVX_MUL(a2_2, h_5_2));
	v2 = _AVX_ADD(v2, _AVX_MUL(a1_2, h_5_1));
	register __AVX_DATATYPE w2 = _AVX_ADD(a4_2, _AVX_MUL(a3_2, h_4_3));
	w2 = _AVX_ADD(w2, _AVX_MUL(a2_2, h_4_2));
	w2 = _AVX_ADD(w2, _AVX_MUL(a1_2, h_4_1));
	register __AVX_DATATYPE z2 = _AVX_ADD(a3_2, _AVX_MUL(a2_2, h_3_2));
	z2 = _AVX_ADD(z2, _AVX_MUL(a1_2, h_3_1));
	register __AVX_DATATYPE y2 = _AVX_ADD(a2_2, _AVX_MUL(a1_2, h_2_1));
#endif
	register __AVX_DATATYPE x2 = a1_2;

	__AVX_DATATYPE q1;
	__AVX_DATATYPE q2;

	__AVX_DATATYPE h1;
	__AVX_DATATYPE h2;
	__AVX_DATATYPE h3;
	__AVX_DATATYPE h4;
	__AVX_DATATYPE h5;
	__AVX_DATATYPE h6;

	for(i = 6; i < nb; i++)
	{
		h1 = _AVX_BROADCAST(&hh[i-5]);
		q1 = _AVX_LOAD(&q[i*ldq]);
		q2 = _AVX_LOAD(&q[(i*ldq)+offset]);
#ifdef __ELPA_USE_FMA__
		x1 = _AVX_FMA(q1, h1, x1);
		x2 = _AVX_FMA(q2, h1, x2);
#else
		x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
		x2 = _AVX_ADD(x2, _AVX_MUL(q2,h1));
#endif
		h2 = _AVX_BROADCAST(&hh[ldh+i-4]);
#ifdef __ELPA_USE_FMA__
		y1 = _AVX_FMA(q1, h2, y1);
		y2 = _AVX_FMA(q2, h2, y2);
#else
		y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
		y2 = _AVX_ADD(y2, _AVX_MUL(q2,h2));
#endif
		h3 = _AVX_BROADCAST(&hh[(ldh*2)+i-3]);
#ifdef __ELPA_USE_FMA__
		z1 = _AVX_FMA(q1, h3, z1);
		z2 = _AVX_FMA(q2, h3, z2);
#else
		z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
		z2 = _AVX_ADD(z2, _AVX_MUL(q2,h3));
#endif
		h4 = _AVX_BROADCAST(&hh[(ldh*3)+i-2]);
#ifdef __ELPA_USE_FMA__
		w1 = _AVX_FMA(q1, h4, w1);
		w2 = _AVX_FMA(q2, h4, w2);
#else
		w1 = _AVX_ADD(w1, _AVX_MUL(q1,h4));
		w2 = _AVX_ADD(w2, _AVX_MUL(q2,h4));
#endif
		h5 = _AVX_BROADCAST(&hh[(ldh*4)+i-1]);
#ifdef __ELPA_USE_FMA__
		v1 = _AVX_FMA(q1, h5, v1);
		v2 = _AVX_FMA(q2, h5, v2);
#else
		v1 = _AVX_ADD(v1, _AVX_MUL(q1,h5));
		v2 = _AVX_ADD(v2, _AVX_MUL(q2,h5));
#endif
		h6 = _AVX_BROADCAST(&hh[(ldh*5)+i]);
#ifdef __ELPA_USE_FMA__
		t1 = _AVX_FMA(q1, h6, t1);
		t2 = _AVX_FMA(q2, h6, t2);
#else
		t1 = _AVX_ADD(t1, _AVX_MUL(q1,h6));
		t2 = _AVX_ADD(t2, _AVX_MUL(q2,h6));
#endif
	}

	h1 = _AVX_BROADCAST(&hh[nb-5]);
	q1 = _AVX_LOAD(&q[nb*ldq]);
	q2 = _AVX_LOAD(&q[(nb*ldq)+offset]);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
	x2 = _AVX_FMA(q2, h1, x2);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
	x2 = _AVX_ADD(x2, _AVX_MUL(q2,h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-4]);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMA(q1, h2, y1);
	y2 = _AVX_FMA(q2, h2, y2);
#else
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
	y2 = _AVX_ADD(y2, _AVX_MUL(q2,h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-3]);
#ifdef __ELPA_USE_FMA__
	z1 = _AVX_FMA(q1, h3, z1);
	z2 = _AVX_FMA(q2, h3, z2);
#else
	z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
	z2 = _AVX_ADD(z2, _AVX_MUL(q2,h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+nb-2]);
#ifdef __ELPA_USE_FMA__
	w1 = _AVX_FMA(q1, h4, w1);
	w2 = _AVX_FMA(q2, h4, w2);
#else
	w1 = _AVX_ADD(w1, _AVX_MUL(q1,h4));
	w2 = _AVX_ADD(w2, _AVX_MUL(q2,h4));
#endif
	h5 = _AVX_BROADCAST(&hh[(ldh*4)+nb-1]);
#ifdef __ELPA_USE_FMA__
	v1 = _AVX_FMA(q1, h5, v1);
	v2 = _AVX_FMA(q2, h5, v2);
#else
	v1 = _AVX_ADD(v1, _AVX_MUL(q1,h5));
	v2 = _AVX_ADD(v2, _AVX_MUL(q2,h5));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-4]);
	q1 = _AVX_LOAD(&q[(nb+1)*ldq]);
	q2 = _AVX_LOAD(&q[((nb+1)*ldq)+offset]);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
	x2 = _AVX_FMA(q2, h1, x2);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
	x2 = _AVX_ADD(x2, _AVX_MUL(q2,h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-3]);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMA(q1, h2, y1);
	y2 = _AVX_FMA(q2, h2, y2);
#else
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
	y2 = _AVX_ADD(y2, _AVX_MUL(q2,h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-2]);
#ifdef __ELPA_USE_FMA__
	z1 = _AVX_FMA(q1, h3, z1);
	z2 = _AVX_FMA(q2, h3, z2);
#else
	z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
	z2 = _AVX_ADD(z2, _AVX_MUL(q2,h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+nb-1]);
#ifdef __ELPA_USE_FMA__
	w1 = _AVX_FMA(q1, h4, w1);
	w2 = _AVX_FMA(q2, h4, w2);
#else
	w1 = _AVX_ADD(w1, _AVX_MUL(q1,h4));
	w2 = _AVX_ADD(w2, _AVX_MUL(q2,h4));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-3]);
	q1 = _AVX_LOAD(&q[(nb+2)*ldq]);
	q2 = _AVX_LOAD(&q[((nb+2)*ldq)+offset]);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
	x2 = _AVX_FMA(q2, h1, x2);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
	x2 = _AVX_ADD(x2, _AVX_MUL(q2,h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-2]);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMA(q1, h2, y1);
	y2 = _AVX_FMA(q2, h2, y2);
#else
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
	y2 = _AVX_ADD(y2, _AVX_MUL(q2,h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-1]);
#ifdef __ELPA_USE_FMA__
	z1 = _AVX_FMA(q1, h3, z1);
	z2 = _AVX_FMA(q2, h3, z2);
#else
	z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
	z2 = _AVX_ADD(z2, _AVX_MUL(q2,h3));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-2]);
	q1 = _AVX_LOAD(&q[(nb+3)*ldq]);
	q2 = _AVX_LOAD(&q[((nb+3)*ldq)+offset]);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
	x2 = _AVX_FMA(q2, h1, x2);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
	x2 = _AVX_ADD(x2, _AVX_MUL(q2,h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-1]);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMA(q1, h2, y1);
	y2 = _AVX_FMA(q2, h2, y2);
#else
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
	y2 = _AVX_ADD(y2, _AVX_MUL(q2,h2));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-1]);
	q1 = _AVX_LOAD(&q[(nb+4)*ldq]);
	q2 = _AVX_LOAD(&q[((nb+4)*ldq)+offset]);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
	x2 = _AVX_FMA(q2, h1, x2);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
	x2 = _AVX_ADD(x2, _AVX_MUL(q2,h1));
#endif

	/////////////////////////////////////////////////////
	// Apply tau, correct wrong calculation using pre-calculated scalar products
	/////////////////////////////////////////////////////

	__AVX_DATATYPE tau1 = _AVX_BROADCAST(&hh[0]);
	x1 = _AVX_MUL(x1, tau1);
	x2 = _AVX_MUL(x2, tau1);

	__AVX_DATATYPE tau2 = _AVX_BROADCAST(&hh[ldh]);
	__AVX_DATATYPE vs_1_2 = _AVX_BROADCAST(&scalarprods[0]);
	h2 = _AVX_MUL(tau2, vs_1_2);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMSUB(y1, tau2, _AVX_MUL(x1,h2));
	y2 = _AVX_FMSUB(y2, tau2, _AVX_MUL(x2,h2));
#else
	y1 = _AVX_SUB(_AVX_MUL(y1,tau2), _AVX_MUL(x1,h2));
	y2 = _AVX_SUB(_AVX_MUL(y2,tau2), _AVX_MUL(x2,h2));
#endif

	__AVX_DATATYPE tau3 = _AVX_BROADCAST(&hh[ldh*2]);
	__AVX_DATATYPE vs_1_3 = _AVX_BROADCAST(&scalarprods[1]);
	__AVX_DATATYPE vs_2_3 = _AVX_BROADCAST(&scalarprods[2]);
	h2 = _AVX_MUL(tau3, vs_1_3);
	h3 = _AVX_MUL(tau3, vs_2_3);
#ifdef __ELPA_USE_FMA__
	z1 = _AVX_FMSUB(z1, tau3, _AVX_FMA(y1, h3, _AVX_MUL(x1,h2)));
	z2 = _AVX_FMSUB(z2, tau3, _AVX_FMA(y2, h3, _AVX_MUL(x2,h2)));
#else
	z1 = _AVX_SUB(_AVX_MUL(z1,tau3), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2)));
	z2 = _AVX_SUB(_AVX_MUL(z2,tau3), _AVX_ADD(_AVX_MUL(y2,h3), _AVX_MUL(x2,h2)));
#endif

	__AVX_DATATYPE tau4 = _AVX_BROADCAST(&hh[ldh*3]);
	__AVX_DATATYPE vs_1_4 = _AVX_BROADCAST(&scalarprods[3]);
	__AVX_DATATYPE vs_2_4 = _AVX_BROADCAST(&scalarprods[4]);
	h2 = _AVX_MUL(tau4, vs_1_4);
	h3 = _AVX_MUL(tau4, vs_2_4);
	__AVX_DATATYPE vs_3_4 = _AVX_BROADCAST(&scalarprods[5]);
	h4 = _AVX_MUL(tau4, vs_3_4);
#ifdef __ELPA_USE_FMA__
	w1 = _AVX_FMSUB(w1, tau4, _AVX_FMA(z1, h4, _AVX_FMA(y1, h3, _AVX_MUL(x1,h2))));
	w2 = _AVX_FMSUB(w2, tau4, _AVX_FMA(z2, h4, _AVX_FMA(y2, h3, _AVX_MUL(x2,h2))));
#else
	w1 = _AVX_SUB(_AVX_MUL(w1,tau4), _AVX_ADD(_AVX_MUL(z1,h4), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2))));
	w2 = _AVX_SUB(_AVX_MUL(w2,tau4), _AVX_ADD(_AVX_MUL(z2,h4), _AVX_ADD(_AVX_MUL(y2,h3), _AVX_MUL(x2,h2))));
#endif

	__AVX_DATATYPE tau5 = _AVX_BROADCAST(&hh[ldh*4]);
	__AVX_DATATYPE vs_1_5 = _AVX_BROADCAST(&scalarprods[6]);
	__AVX_DATATYPE vs_2_5 = _AVX_BROADCAST(&scalarprods[7]);
	h2 = _AVX_MUL(tau5, vs_1_5);
	h3 = _AVX_MUL(tau5, vs_2_5);
	__AVX_DATATYPE vs_3_5 = _AVX_BROADCAST(&scalarprods[8]);
	__AVX_DATATYPE vs_4_5 = _AVX_BROADCAST(&scalarprods[9]);
	h4 = _AVX_MUL(tau5, vs_3_5);
	h5 = _AVX_MUL(tau5, vs_4_5);
#ifdef __ELPA_USE_FMA__
	v1 = _AVX_FMSUB(v1, tau5, _AVX_ADD(_AVX_FMA(w1, h5, _AVX_MUL(z1,h4)), _AVX_FMA(y1, h3, _AVX_MUL(x1,h2))));
	v2 = _AVX_FMSUB(v2, tau5, _AVX_ADD(_AVX_FMA(w2, h5, _AVX_MUL(z2,h4)), _AVX_FMA(y2, h3, _AVX_MUL(x2,h2))));
#else
	v1 = _AVX_SUB(_AVX_MUL(v1,tau5), _AVX_ADD(_AVX_ADD(_AVX_MUL(w1,h5), _AVX_MUL(z1,h4)), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2))));
	v2 = _AVX_SUB(_AVX_MUL(v2,tau5), _AVX_ADD(_AVX_ADD(_AVX_MUL(w2,h5), _AVX_MUL(z2,h4)), _AVX_ADD(_AVX_MUL(y2,h3), _AVX_MUL(x2,h2))));
#endif

	__AVX_DATATYPE tau6 = _AVX_BROADCAST(&hh[ldh*5]);
	__AVX_DATATYPE vs_1_6 = _AVX_BROADCAST(&scalarprods[10]);
	__AVX_DATATYPE vs_2_6 = _AVX_BROADCAST(&scalarprods[11]);
	h2 = _AVX_MUL(tau6, vs_1_6);
	h3 = _AVX_MUL(tau6, vs_2_6);
	__AVX_DATATYPE vs_3_6 = _AVX_BROADCAST(&scalarprods[12]);
	__AVX_DATATYPE vs_4_6 = _AVX_BROADCAST(&scalarprods[13]);
	__AVX_DATATYPE vs_5_6 = _AVX_BROADCAST(&scalarprods[14]);
	h4 = _AVX_MUL(tau6, vs_3_6);
	h5 = _AVX_MUL(tau6, vs_4_6);
	h6 = _AVX_MUL(tau6, vs_5_6);
#ifdef __ELPA_USE_FMA__
	t1 = _AVX_FMSUB(t1, tau6, _AVX_FMA(v1, h6, _AVX_ADD(_AVX_FMA(w1, h5, _AVX_MUL(z1,h4)), _AVX_FMA(y1, h3, _AVX_MUL(x1,h2)))));
	t2 = _AVX_FMSUB(t2, tau6, _AVX_FMA(v2, h6, _AVX_ADD(_AVX_FMA(w2, h5, _AVX_MUL(z2,h4)), _AVX_FMA(y2, h3, _AVX_MUL(x2,h2)))));
#else
	t1 = _AVX_SUB(_AVX_MUL(t1,tau6), _AVX_ADD( _AVX_MUL(v1,h6), _AVX_ADD(_AVX_ADD(_AVX_MUL(w1,h5), _AVX_MUL(z1,h4)), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2)))));
	t2 = _AVX_SUB(_AVX_MUL(t2,tau6), _AVX_ADD( _AVX_MUL(v2,h6), _AVX_ADD(_AVX_ADD(_AVX_MUL(w2,h5), _AVX_MUL(z2,h4)), _AVX_ADD(_AVX_MUL(y2,h3), _AVX_MUL(x2,h2)))));
#endif

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [8 x nb+3]
	/////////////////////////////////////////////////////

	q1 = _AVX_LOAD(&q[0]);
	q2 = _AVX_LOAD(&q[offset]);
	q1 = _AVX_SUB(q1, t1);
	q2 = _AVX_SUB(q2, t2);
	_AVX_STORE(&q[0],q1);
	_AVX_STORE(&q[offset],q2);

	h6 = _AVX_BROADCAST(&hh[(ldh*5)+1]);
	q1 = _AVX_LOAD(&q[ldq]);
	q2 = _AVX_LOAD(&q[(ldq+offset)]);
	q1 = _AVX_SUB(q1, v1);
	q2 = _AVX_SUB(q2, v2);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(t1, h6, q1);
	q2 = _AVX_NFMA(t2, h6, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
	q2 = _AVX_SUB(q2, _AVX_MUL(t2, h6));
#endif
	_AVX_STORE(&q[ldq],q1);
	_AVX_STORE(&q[(ldq+offset)],q2);

	h5 = _AVX_BROADCAST(&hh[(ldh*4)+1]);
	q1 = _AVX_LOAD(&q[ldq*2]);
	q2 = _AVX_LOAD(&q[(ldq*2)+offset]);
	q1 = _AVX_SUB(q1, w1);
	q2 = _AVX_SUB(q2, w2);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(v1, h5, q1);
	q2 = _AVX_NFMA(v2, h5, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
	q2 = _AVX_SUB(q2, _AVX_MUL(v2, h5));
#endif
	h6 = _AVX_BROADCAST(&hh[(ldh*5)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(t1, h6, q1);
	q2 = _AVX_NFMA(t2, h6, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
	q2 = _AVX_SUB(q2, _AVX_MUL(t2, h6));
#endif
	_AVX_STORE(&q[ldq*2],q1);
	_AVX_STORE(&q[(ldq*2)+offset],q2);

	h4 = _AVX_BROADCAST(&hh[(ldh*3)+1]);
	q1 = _AVX_LOAD(&q[ldq*3]);
	q2 = _AVX_LOAD(&q[(ldq*3)+offset]);
	q1 = _AVX_SUB(q1, z1);
	q2 = _AVX_SUB(q2, z2);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(w1, h4, q1);
	q2 = _AVX_NFMA(w2, h4, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
	q2 = _AVX_SUB(q2, _AVX_MUL(w2, h4));
#endif
	h5 = _AVX_BROADCAST(&hh[(ldh*4)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(v1, h5, q1);
	q2 = _AVX_NFMA(v2, h5, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
	q2 = _AVX_SUB(q2, _AVX_MUL(v2, h5));
#endif
	h6 = _AVX_BROADCAST(&hh[(ldh*5)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(t1, h6, q1);
	q2 = _AVX_NFMA(t2, h6, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
	q2 = _AVX_SUB(q2, _AVX_MUL(t2, h6));
#endif
	_AVX_STORE(&q[ldq*3],q1);
	_AVX_STORE(&q[(ldq*3)+offset],q2);

	h3 = _AVX_BROADCAST(&hh[(ldh*2)+1]);
	q1 = _AVX_LOAD(&q[ldq*4]);
	q2 = _AVX_LOAD(&q[(ldq*4)+offset]);
	q1 = _AVX_SUB(q1, y1);
	q2 = _AVX_SUB(q2, y2);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
	q2 = _AVX_NFMA(z2, h3, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
	q2 = _AVX_SUB(q2, _AVX_MUL(z2, h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(w1, h4, q1);
	q2 = _AVX_NFMA(w2, h4, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
	q2 = _AVX_SUB(q2, _AVX_MUL(w2, h4));
#endif
	h5 = _AVX_BROADCAST(&hh[(ldh*4)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(v1, h5, q1);
	q2 = _AVX_NFMA(v2, h5, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
	q2 = _AVX_SUB(q2, _AVX_MUL(v2, h5));
#endif
	h6 = _AVX_BROADCAST(&hh[(ldh*5)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(t1, h6, q1);
	q2 = _AVX_NFMA(t2, h6, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
	q2 = _AVX_SUB(q2, _AVX_MUL(t2, h6));
#endif
	_AVX_STORE(&q[ldq*4],q1);
	_AVX_STORE(&q[(ldq*4)+offset],q2);

	h2 = _AVX_BROADCAST(&hh[(ldh)+1]);
	q1 = _AVX_LOAD(&q[ldq*5]);
	q2 = _AVX_LOAD(&q[(ldq*5)+offset]);
	q1 = _AVX_SUB(q1, x1);
	q2 = _AVX_SUB(q2, x2);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
	q2 = _AVX_NFMA(y2, h2, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
	q2 = _AVX_SUB(q2, _AVX_MUL(y2, h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
	q2 = _AVX_NFMA(z2, h3, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
	q2 = _AVX_SUB(q2, _AVX_MUL(z2, h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(w1, h4, q1);
	q2 = _AVX_NFMA(w2, h4, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
	q2 = _AVX_SUB(q2, _AVX_MUL(w2, h4));
#endif
	h5 = _AVX_BROADCAST(&hh[(ldh*4)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(v1, h5, q1);
	q2 = _AVX_NFMA(v2, h5, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
	q2 = _AVX_SUB(q2, _AVX_MUL(v2, h5));
#endif
	h6 = _AVX_BROADCAST(&hh[(ldh*5)+5]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(t1, h6, q1);
	q2 = _AVX_NFMA(t2, h6, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
	q2 = _AVX_SUB(q2, _AVX_MUL(t2, h6));
#endif
	_AVX_STORE(&q[ldq*5],q1);
	_AVX_STORE(&q[(ldq*5)+offset],q2);

	for (i = 6; i < nb; i++)
	{
		q1 = _AVX_LOAD(&q[i*ldq]);
		q2 = _AVX_LOAD(&q[(i*ldq)+offset]);
		h1 = _AVX_BROADCAST(&hh[i-5]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(x1, h1, q1);
		q2 = _AVX_NFMA(x2, h1, q2);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
		q2 = _AVX_SUB(q2, _AVX_MUL(x2, h1));
#endif
		h2 = _AVX_BROADCAST(&hh[ldh+i-4]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(y1, h2, q1);
		q2 = _AVX_NFMA(y2, h2, q2);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
		q2 = _AVX_SUB(q2, _AVX_MUL(y2, h2));
#endif
		h3 = _AVX_BROADCAST(&hh[(ldh*2)+i-3]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(z1, h3, q1);
		q2 = _AVX_NFMA(z2, h3, q2);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
		q2 = _AVX_SUB(q2, _AVX_MUL(z2, h3));
#endif
		h4 = _AVX_BROADCAST(&hh[(ldh*3)+i-2]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(w1, h4, q1);
		q2 = _AVX_NFMA(w2, h4, q2);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
		q2 = _AVX_SUB(q2, _AVX_MUL(w2, h4));
#endif
		h5 = _AVX_BROADCAST(&hh[(ldh*4)+i-1]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(v1, h5, q1);
		q2 = _AVX_NFMA(v2, h5, q2);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
		q2 = _AVX_SUB(q2, _AVX_MUL(v2, h5));
#endif
		h6 = _AVX_BROADCAST(&hh[(ldh*5)+i]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(t1, h6, q1);
		q2 = _AVX_NFMA(t2, h6, q2);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
		q2 = _AVX_SUB(q2, _AVX_MUL(t2, h6));
#endif
		_AVX_STORE(&q[i*ldq],q1);
		_AVX_STORE(&q[(i*ldq)+offset],q2);
	}

	h1 = _AVX_BROADCAST(&hh[nb-5]);
	q1 = _AVX_LOAD(&q[nb*ldq]);
	q2 = _AVX_LOAD(&q[(nb*ldq)+offset]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
	q2 = _AVX_NFMA(x2, h1, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
	q2 = _AVX_SUB(q2, _AVX_MUL(x2, h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-4]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
	q2 = _AVX_NFMA(y2, h2, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
	q2 = _AVX_SUB(q2, _AVX_MUL(y2, h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-3]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
	q2 = _AVX_NFMA(z2, h3, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
	q2 = _AVX_SUB(q2, _AVX_MUL(z2, h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(w1, h4, q1);
	q2 = _AVX_NFMA(w2, h4, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
	q2 = _AVX_SUB(q2, _AVX_MUL(w2, h4));
#endif
	h5 = _AVX_BROADCAST(&hh[(ldh*4)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(v1, h5, q1);
	q2 = _AVX_NFMA(v2, h5, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
	q2 = _AVX_SUB(q2, _AVX_MUL(v2, h5));
#endif
	_AVX_STORE(&q[nb*ldq],q1);
	_AVX_STORE(&q[(nb*ldq)+offset],q2);

	h1 = _AVX_BROADCAST(&hh[nb-4]);
	q1 = _AVX_LOAD(&q[(nb+1)*ldq]);
	q2 = _AVX_LOAD(&q[((nb+1)*ldq)+offset]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
	q2 = _AVX_NFMA(x2, h1, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
	q2 = _AVX_SUB(q2, _AVX_MUL(x2, h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-3]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
	q2 = _AVX_NFMA(y2, h2, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
	q2 = _AVX_SUB(q2, _AVX_MUL(y2, h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
	q2 = _AVX_NFMA(z2, h3, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
	q2 = _AVX_SUB(q2, _AVX_MUL(z2, h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(w1, h4, q1);
	q2 = _AVX_NFMA(w2, h4, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
	q2 = _AVX_SUB(q2, _AVX_MUL(w2, h4));
#endif
	_AVX_STORE(&q[(nb+1)*ldq],q1);
	_AVX_STORE(&q[((nb+1)*ldq)+offset],q2);

	h1 = _AVX_BROADCAST(&hh[nb-3]);
	q1 = _AVX_LOAD(&q[(nb+2)*ldq]);
	q2 = _AVX_LOAD(&q[((nb+2)*ldq)+offset]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
	q2 = _AVX_NFMA(x2, h1, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
	q2 = _AVX_SUB(q2, _AVX_MUL(x2, h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
	q2 = _AVX_NFMA(y2, h2, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
	q2 = _AVX_SUB(q2, _AVX_MUL(y2, h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
	q2 = _AVX_NFMA(z2, h3, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
	q2 = _AVX_SUB(q2, _AVX_MUL(z2, h3));
#endif
	_AVX_STORE(&q[(nb+2)*ldq],q1);
	_AVX_STORE(&q[((nb+2)*ldq)+offset],q2);

	h1 = _AVX_BROADCAST(&hh[nb-2]);
	q1 = _AVX_LOAD(&q[(nb+3)*ldq]);
	q2 = _AVX_LOAD(&q[((nb+3)*ldq)+offset]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
	q2 = _AVX_NFMA(x2, h1, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
	q2 = _AVX_SUB(q2, _AVX_MUL(x2, h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
	q2 = _AVX_NFMA(y2, h2, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
	q2 = _AVX_SUB(q2, _AVX_MUL(y2, h2));
#endif
	_AVX_STORE(&q[(nb+3)*ldq],q1);
	_AVX_STORE(&q[((nb+3)*ldq)+offset],q2);

	h1 = _AVX_BROADCAST(&hh[nb-1]);
	q1 = _AVX_LOAD(&q[(nb+4)*ldq]);
	q2 = _AVX_LOAD(&q[((nb+4)*ldq)+offset]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
	q2 = _AVX_NFMA(x2, h1, q2);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
	q2 = _AVX_SUB(q2, _AVX_MUL(x2, h1));
#endif
	_AVX_STORE(&q[(nb+4)*ldq],q1);
	_AVX_STORE(&q[((nb+4)*ldq)+offset],q2);
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
__forceinline void hh_trafo_kernel_4_AVX_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_8_AVX_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [8 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__AVX_DATATYPE a1_1 = _AVX_LOAD(&q[ldq*5]);
	__AVX_DATATYPE a2_1 = _AVX_LOAD(&q[ldq*4]);
	__AVX_DATATYPE a3_1 = _AVX_LOAD(&q[ldq*3]);
	__AVX_DATATYPE a4_1 = _AVX_LOAD(&q[ldq*2]);
	__AVX_DATATYPE a5_1 = _AVX_LOAD(&q[ldq]);
	__AVX_DATATYPE a6_1 = _AVX_LOAD(&q[0]);

	__AVX_DATATYPE h_6_5 = _AVX_BROADCAST(&hh[(ldh*5)+1]);
	__AVX_DATATYPE h_6_4 = _AVX_BROADCAST(&hh[(ldh*5)+2]);
	__AVX_DATATYPE h_6_3 = _AVX_BROADCAST(&hh[(ldh*5)+3]);
	__AVX_DATATYPE h_6_2 = _AVX_BROADCAST(&hh[(ldh*5)+4]);
	__AVX_DATATYPE h_6_1 = _AVX_BROADCAST(&hh[(ldh*5)+5]);
#ifdef __ELPA_USE_FMA__
	register __AVX_DATATYPE t1 = _AVX_FMA(a5_1, h_6_5, a6_1);
	t1 = _AVX_FMA(a4_1, h_6_4, t1);
	t1 = _AVX_FMA(a3_1, h_6_3, t1);
	t1 = _AVX_FMA(a2_1, h_6_2, t1);
	t1 = _AVX_FMA(a1_1, h_6_1, t1);
#else
	register __AVX_DATATYPE t1 = _AVX_ADD(a6_1, _AVX_MUL(a5_1, h_6_5));
	t1 = _AVX_ADD(t1, _AVX_MUL(a4_1, h_6_4));
	t1 = _AVX_ADD(t1, _AVX_MUL(a3_1, h_6_3));
	t1 = _AVX_ADD(t1, _AVX_MUL(a2_1, h_6_2));
	t1 = _AVX_ADD(t1, _AVX_MUL(a1_1, h_6_1));
#endif
	__AVX_DATATYPE h_5_4 = _AVX_BROADCAST(&hh[(ldh*4)+1]);
	__AVX_DATATYPE h_5_3 = _AVX_BROADCAST(&hh[(ldh*4)+2]);
	__AVX_DATATYPE h_5_2 = _AVX_BROADCAST(&hh[(ldh*4)+3]);
	__AVX_DATATYPE h_5_1 = _AVX_BROADCAST(&hh[(ldh*4)+4]);
#ifdef __ELPA_USE_FMA__
	register __AVX_DATATYPE v1 = _AVX_FMA(a4_1, h_5_4, a5_1);
	v1 = _AVX_FMA(a3_1, h_5_3, v1);
	v1 = _AVX_FMA(a2_1, h_5_2, v1);
	v1 = _AVX_FMA(a1_1, h_5_1, v1);
#else
	register __AVX_DATATYPE v1 = _AVX_ADD(a5_1, _AVX_MUL(a4_1, h_5_4));
	v1 = _AVX_ADD(v1, _AVX_MUL(a3_1, h_5_3));
	v1 = _AVX_ADD(v1, _AVX_MUL(a2_1, h_5_2));
	v1 = _AVX_ADD(v1, _AVX_MUL(a1_1, h_5_1));
#endif
	__AVX_DATATYPE h_4_3 = _AVX_BROADCAST(&hh[(ldh*3)+1]);
	__AVX_DATATYPE h_4_2 = _AVX_BROADCAST(&hh[(ldh*3)+2]);
	__AVX_DATATYPE h_4_1 = _AVX_BROADCAST(&hh[(ldh*3)+3]);
#ifdef __ELPA_USE_FMA__
	register __AVX_DATATYPE w1 = _AVX_FMA(a3_1, h_4_3, a4_1);
	w1 = _AVX_FMA(a2_1, h_4_2, w1);
	w1 = _AVX_FMA(a1_1, h_4_1, w1);
#else
	register __AVX_DATATYPE w1 = _AVX_ADD(a4_1, _AVX_MUL(a3_1, h_4_3));
	w1 = _AVX_ADD(w1, _AVX_MUL(a2_1, h_4_2));
	w1 = _AVX_ADD(w1, _AVX_MUL(a1_1, h_4_1));
#endif
	__AVX_DATATYPE h_2_1 = _AVX_BROADCAST(&hh[ldh+1]);
	__AVX_DATATYPE h_3_2 = _AVX_BROADCAST(&hh[(ldh*2)+1]);
	__AVX_DATATYPE h_3_1 = _AVX_BROADCAST(&hh[(ldh*2)+2]);
#ifdef __ELPA_USE_FMA__
	register __AVX_DATATYPE z1 = _AVX_FMA(a2_1, h_3_2, a3_1);
	z1 = _AVX_FMA(a1_1, h_3_1, z1);
	register __AVX_DATATYPE y1 = _AVX_FMA(a1_1, h_2_1, a2_1);
#else
	register __AVX_DATATYPE z1 = _AVX_ADD(a3_1, _AVX_MUL(a2_1, h_3_2));
	z1 = _AVX_ADD(z1, _AVX_MUL(a1_1, h_3_1));
	register __AVX_DATATYPE y1 = _AVX_ADD(a2_1, _AVX_MUL(a1_1, h_2_1));
#endif
	register __AVX_DATATYPE x1 = a1_1;

	__AVX_DATATYPE q1;

	__AVX_DATATYPE h1;
	__AVX_DATATYPE h2;
	__AVX_DATATYPE h3;
	__AVX_DATATYPE h4;
	__AVX_DATATYPE h5;
	__AVX_DATATYPE h6;

	for(i = 6; i < nb; i++)
	{
		h1 = _AVX_BROADCAST(&hh[i-5]);
		q1 = _AVX_LOAD(&q[i*ldq]);
#ifdef __ELPA_USE_FMA__
		x1 = _AVX_FMA(q1, h1, x1);
#else
		x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
#endif
		h2 = _AVX_BROADCAST(&hh[ldh+i-4]);
#ifdef __ELPA_USE_FMA__
		y1 = _AVX_FMA(q1, h2, y1);
#else
		y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
#endif
		h3 = _AVX_BROADCAST(&hh[(ldh*2)+i-3]);
#ifdef __ELPA_USE_FMA__
		z1 = _AVX_FMA(q1, h3, z1);
#else
		z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
#endif
		h4 = _AVX_BROADCAST(&hh[(ldh*3)+i-2]);
#ifdef __ELPA_USE_FMA__
		w1 = _AVX_FMA(q1, h4, w1);
#else
		w1 = _AVX_ADD(w1, _AVX_MUL(q1,h4));
#endif
		h5 = _AVX_BROADCAST(&hh[(ldh*4)+i-1]);
#ifdef __ELPA_USE_FMA__
		v1 = _AVX_FMA(q1, h5, v1);
#else
		v1 = _AVX_ADD(v1, _AVX_MUL(q1,h5));
#endif
		h6 = _AVX_BROADCAST(&hh[(ldh*5)+i]);
#ifdef __ELPA_USE_FMA__
		t1 = _AVX_FMA(q1, h6, t1);
#else
		t1 = _AVX_ADD(t1, _AVX_MUL(q1,h6));
#endif
	}

	h1 = _AVX_BROADCAST(&hh[nb-5]);
	q1 = _AVX_LOAD(&q[nb*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-4]);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMA(q1, h2, y1);
#else
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-3]);
#ifdef __ELPA_USE_FMA__
	z1 = _AVX_FMA(q1, h3, z1);
#else
	z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+nb-2]);
#ifdef __ELPA_USE_FMA__
	w1 = _AVX_FMA(q1, h4, w1);
#else
	w1 = _AVX_ADD(w1, _AVX_MUL(q1,h4));
#endif
	h5 = _AVX_BROADCAST(&hh[(ldh*4)+nb-1]);
#ifdef __ELPA_USE_FMA__
	v1 = _AVX_FMA(q1, h5, v1);
#else
	v1 = _AVX_ADD(v1, _AVX_MUL(q1,h5));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-4]);
	q1 = _AVX_LOAD(&q[(nb+1)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-3]);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMA(q1, h2, y1);
#else
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-2]);
#ifdef __ELPA_USE_FMA__
	z1 = _AVX_FMA(q1, h3, z1);
#else
	z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+nb-1]);
#ifdef __ELPA_USE_FMA__
	w1 = _AVX_FMA(q1, h4, w1);
#else
	w1 = _AVX_ADD(w1, _AVX_MUL(q1,h4));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-3]);
	q1 = _AVX_LOAD(&q[(nb+2)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-2]);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMA(q1, h2, y1);
#else
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-1]);
#ifdef __ELPA_USE_FMA__
	z1 = _AVX_FMA(q1, h3, z1);
#else
	z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-2]);
	q1 = _AVX_LOAD(&q[(nb+3)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-1]);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMA(q1, h2, y1);
#else
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-1]);
	q1 = _AVX_LOAD(&q[(nb+4)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
#endif

	/////////////////////////////////////////////////////
	// Apply tau, correct wrong calculation using pre-calculated scalar products
	/////////////////////////////////////////////////////

	__AVX_DATATYPE tau1 = _AVX_BROADCAST(&hh[0]);
	x1 = _AVX_MUL(x1, tau1);

	__AVX_DATATYPE tau2 = _AVX_BROADCAST(&hh[ldh]);
	__AVX_DATATYPE vs_1_2 = _AVX_BROADCAST(&scalarprods[0]);
	h2 = _AVX_MUL(tau2, vs_1_2);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMSUB(y1, tau2, _AVX_MUL(x1,h2));
#else
	y1 = _AVX_SUB(_AVX_MUL(y1,tau2), _AVX_MUL(x1,h2));
#endif

	__AVX_DATATYPE tau3 = _AVX_BROADCAST(&hh[ldh*2]);
	__AVX_DATATYPE vs_1_3 = _AVX_BROADCAST(&scalarprods[1]);
	__AVX_DATATYPE vs_2_3 = _AVX_BROADCAST(&scalarprods[2]);
	h2 = _AVX_MUL(tau3, vs_1_3);
	h3 = _AVX_MUL(tau3, vs_2_3);
#ifdef __ELPA_USE_FMA__
	z1 = _AVX_FMSUB(z1, tau3, _AVX_FMA(y1, h3, _AVX_MUL(x1,h2)));
#else
	z1 = _AVX_SUB(_AVX_MUL(z1,tau3), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2)));
#endif

	__AVX_DATATYPE tau4 = _AVX_BROADCAST(&hh[ldh*3]);
	__AVX_DATATYPE vs_1_4 = _AVX_BROADCAST(&scalarprods[3]);
	__AVX_DATATYPE vs_2_4 = _AVX_BROADCAST(&scalarprods[4]);
	h2 = _AVX_MUL(tau4, vs_1_4);
	h3 = _AVX_MUL(tau4, vs_2_4);
	__AVX_DATATYPE vs_3_4 = _AVX_BROADCAST(&scalarprods[5]);
	h4 = _AVX_MUL(tau4, vs_3_4);
#ifdef __ELPA_USE_FMA__
	w1 = _AVX_FMSUB(w1, tau4, _AVX_FMA(z1, h4, _AVX_FMA(y1, h3, _AVX_MUL(x1,h2))));
#else
	w1 = _AVX_SUB(_AVX_MUL(w1,tau4), _AVX_ADD(_AVX_MUL(z1,h4), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2))));
#endif

	__AVX_DATATYPE tau5 = _AVX_BROADCAST(&hh[ldh*4]);
	__AVX_DATATYPE vs_1_5 = _AVX_BROADCAST(&scalarprods[6]);
	__AVX_DATATYPE vs_2_5 = _AVX_BROADCAST(&scalarprods[7]);
	h2 = _AVX_MUL(tau5, vs_1_5);
	h3 = _AVX_MUL(tau5, vs_2_5);
	__AVX_DATATYPE vs_3_5 = _AVX_BROADCAST(&scalarprods[8]);
	__AVX_DATATYPE vs_4_5 = _AVX_BROADCAST(&scalarprods[9]);
	h4 = _AVX_MUL(tau5, vs_3_5);
	h5 = _AVX_MUL(tau5, vs_4_5);
#ifdef __ELPA_USE_FMA__
	v1 = _AVX_FMSUB(v1, tau5, _AVX_ADD(_AVX_FMA(w1, h5, _AVX_MUL(z1,h4)), _AVX_FMA(y1, h3, _AVX_MUL(x1,h2))));
#else
	v1 = _AVX_SUB(_AVX_MUL(v1,tau5), _AVX_ADD(_AVX_ADD(_AVX_MUL(w1,h5), _AVX_MUL(z1,h4)), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2))));
#endif

	__AVX_DATATYPE tau6 = _AVX_BROADCAST(&hh[ldh*5]);
	__AVX_DATATYPE vs_1_6 = _AVX_BROADCAST(&scalarprods[10]);
	__AVX_DATATYPE vs_2_6 = _AVX_BROADCAST(&scalarprods[11]);
	h2 = _AVX_MUL(tau6, vs_1_6);
	h3 = _AVX_MUL(tau6, vs_2_6);
	__AVX_DATATYPE vs_3_6 = _AVX_BROADCAST(&scalarprods[12]);
	__AVX_DATATYPE vs_4_6 = _AVX_BROADCAST(&scalarprods[13]);
	__AVX_DATATYPE vs_5_6 = _AVX_BROADCAST(&scalarprods[14]);
	h4 = _AVX_MUL(tau6, vs_3_6);
	h5 = _AVX_MUL(tau6, vs_4_6);
	h6 = _AVX_MUL(tau6, vs_5_6);
#ifdef __ELPA_USE_FMA__
	t1 = _AVX_FMSUB(t1, tau6, _AVX_FMA(v1, h6, _AVX_ADD(_AVX_FMA(w1, h5, _AVX_MUL(z1,h4)), _AVX_FMA(y1, h3, _AVX_MUL(x1,h2)))));
#else
	t1 = _AVX_SUB(_AVX_MUL(t1,tau6), _AVX_ADD( _AVX_MUL(v1,h6), _AVX_ADD(_AVX_ADD(_AVX_MUL(w1,h5), _AVX_MUL(z1,h4)), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2)))));
#endif

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [4 x nb+3]
	/////////////////////////////////////////////////////

	q1 = _AVX_LOAD(&q[0]);
	q1 = _AVX_SUB(q1, t1);
	_AVX_STORE(&q[0],q1);

	h6 = _AVX_BROADCAST(&hh[(ldh*5)+1]);
	q1 = _AVX_LOAD(&q[ldq]);
	q1 = _AVX_SUB(q1, v1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(t1, h6, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
#endif
	_AVX_STORE(&q[ldq],q1);

	h5 = _AVX_BROADCAST(&hh[(ldh*4)+1]);
	q1 = _AVX_LOAD(&q[ldq*2]);
	q1 = _AVX_SUB(q1, w1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(v1, h5, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
#endif
	h6 = _AVX_BROADCAST(&hh[(ldh*5)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(t1, h6, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
#endif
	_AVX_STORE(&q[ldq*2],q1);

	h4 = _AVX_BROADCAST(&hh[(ldh*3)+1]);
	q1 = _AVX_LOAD(&q[ldq*3]);
	q1 = _AVX_SUB(q1, z1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(w1, h4, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
#endif
	h5 = _AVX_BROADCAST(&hh[(ldh*4)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(v1, h5, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
#endif
	h6 = _AVX_BROADCAST(&hh[(ldh*5)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(t1, h6, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
#endif
	_AVX_STORE(&q[ldq*3],q1);

	h3 = _AVX_BROADCAST(&hh[(ldh*2)+1]);
	q1 = _AVX_LOAD(&q[ldq*4]);
	q1 = _AVX_SUB(q1, y1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(w1, h4, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
#endif
	h5 = _AVX_BROADCAST(&hh[(ldh*4)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(v1, h5, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
#endif
	h6 = _AVX_BROADCAST(&hh[(ldh*5)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(t1, h6, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
#endif
	_AVX_STORE(&q[ldq*4],q1);

	h2 = _AVX_BROADCAST(&hh[(ldh)+1]);
	q1 = _AVX_LOAD(&q[ldq*5]);
	q1 = _AVX_SUB(q1, x1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(w1, h4, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
#endif
	h5 = _AVX_BROADCAST(&hh[(ldh*4)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(v1, h5, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
#endif
	h6 = _AVX_BROADCAST(&hh[(ldh*5)+5]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(t1, h6, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
#endif
	_AVX_STORE(&q[ldq*5],q1);

	for (i = 6; i < nb; i++)
	{
		q1 = _AVX_LOAD(&q[i*ldq]);
		h1 = _AVX_BROADCAST(&hh[i-5]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(x1, h1, q1);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
#endif
		h2 = _AVX_BROADCAST(&hh[ldh+i-4]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(y1, h2, q1);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
#endif
		h3 = _AVX_BROADCAST(&hh[(ldh*2)+i-3]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(z1, h3, q1);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
#endif
		h4 = _AVX_BROADCAST(&hh[(ldh*3)+i-2]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(w1, h4, q1);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
#endif
		h5 = _AVX_BROADCAST(&hh[(ldh*4)+i-1]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(v1, h5, q1);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
#endif
		h6 = _AVX_BROADCAST(&hh[(ldh*5)+i]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(t1, h6, q1);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
#endif
		_AVX_STORE(&q[i*ldq],q1);
	}

	h1 = _AVX_BROADCAST(&hh[nb-5]);
	q1 = _AVX_LOAD(&q[nb*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-4]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-3]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(w1, h4, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
#endif
	h5 = _AVX_BROADCAST(&hh[(ldh*4)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(v1, h5, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
#endif
	_AVX_STORE(&q[nb*ldq],q1);

	h1 = _AVX_BROADCAST(&hh[nb-4]);
	q1 = _AVX_LOAD(&q[(nb+1)*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-3]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(w1, h4, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
#endif
	_AVX_STORE(&q[(nb+1)*ldq],q1);

	h1 = _AVX_BROADCAST(&hh[nb-3]);
	q1 = _AVX_LOAD(&q[(nb+2)*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
#endif
	_AVX_STORE(&q[(nb+2)*ldq],q1);

	h1 = _AVX_BROADCAST(&hh[nb-2]);
	q1 = _AVX_LOAD(&q[(nb+3)*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
#endif
	_AVX_STORE(&q[(nb+3)*ldq],q1);

	h1 = _AVX_BROADCAST(&hh[nb-1]);
	q1 = _AVX_LOAD(&q[(nb+4)*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
#endif
	_AVX_STORE(&q[(nb+4)*ldq],q1);
}

#ifdef SINGLE_PRECISION_REAL
/**
 * Unrolled kernel that computes
 * 4 rows of Q simultaneously, a
 * matrix Vector product with two householder
 * vectors + a rank 1 upsate is performed
 */
__forceinline void hh_trafo_kernel_4_AVX_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [8 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__AVX_DATATYPE a1_1 = _mm256_castps128_ps256(_mm_load_ps(&q[ldq*5]));
	__AVX_DATATYPE a2_1 = _mm256_castps128_ps256(_mm_load_ps(&q[ldq*4]));
	__AVX_DATATYPE a3_1 = _mm256_castps128_ps256(_mm_load_ps(&q[ldq*3]));
	__AVX_DATATYPE a4_1 = _mm256_castps128_ps256(_mm_load_ps(&q[ldq*2]));
	__AVX_DATATYPE a5_1 = _mm256_castps128_ps256(_mm_load_ps(&q[ldq]));
	// we just want to load 4 floats not 8
	__AVX_DATATYPE a6_1 = _mm256_castps128_ps256(_mm_load_ps(&q[0]));      // q(1,1) | q(2,1) | q(3,1) | q(4,1)

	__AVX_DATATYPE h_6_5 = _AVX_BROADCAST(&hh[(ldh*5)+1]);
	__AVX_DATATYPE h_6_4 = _AVX_BROADCAST(&hh[(ldh*5)+2]);
	__AVX_DATATYPE h_6_3 = _AVX_BROADCAST(&hh[(ldh*5)+3]);
	__AVX_DATATYPE h_6_2 = _AVX_BROADCAST(&hh[(ldh*5)+4]);
	__AVX_DATATYPE h_6_1 = _AVX_BROADCAST(&hh[(ldh*5)+5]);
#ifdef __ELPA_USE_FMA__
	register __AVX_DATATYPE t1 = _AVX_FMA(a5_1, h_6_5, a6_1);
	t1 = _AVX_FMA(a4_1, h_6_4, t1);
	t1 = _AVX_FMA(a3_1, h_6_3, t1);
	t1 = _AVX_FMA(a2_1, h_6_2, t1);
	t1 = _AVX_FMA(a1_1, h_6_1, t1);
#else
	register __AVX_DATATYPE t1 = _AVX_ADD(a6_1, _AVX_MUL(a5_1, h_6_5));
	t1 = _AVX_ADD(t1, _AVX_MUL(a4_1, h_6_4));
	t1 = _AVX_ADD(t1, _AVX_MUL(a3_1, h_6_3));
	t1 = _AVX_ADD(t1, _AVX_MUL(a2_1, h_6_2));
	t1 = _AVX_ADD(t1, _AVX_MUL(a1_1, h_6_1));
#endif
	__AVX_DATATYPE h_5_4 = _AVX_BROADCAST(&hh[(ldh*4)+1]);
	__AVX_DATATYPE h_5_3 = _AVX_BROADCAST(&hh[(ldh*4)+2]);
	__AVX_DATATYPE h_5_2 = _AVX_BROADCAST(&hh[(ldh*4)+3]);
	__AVX_DATATYPE h_5_1 = _AVX_BROADCAST(&hh[(ldh*4)+4]);
#ifdef __ELPA_USE_FMA__
	register __AVX_DATATYPE v1 = _AVX_FMA(a4_1, h_5_4, a5_1);
	v1 = _AVX_FMA(a3_1, h_5_3, v1);
	v1 = _AVX_FMA(a2_1, h_5_2, v1);
	v1 = _AVX_FMA(a1_1, h_5_1, v1);
#else
	register __AVX_DATATYPE v1 = _AVX_ADD(a5_1, _AVX_MUL(a4_1, h_5_4));
	v1 = _AVX_ADD(v1, _AVX_MUL(a3_1, h_5_3));
	v1 = _AVX_ADD(v1, _AVX_MUL(a2_1, h_5_2));
	v1 = _AVX_ADD(v1, _AVX_MUL(a1_1, h_5_1));
#endif
	__AVX_DATATYPE h_4_3 = _AVX_BROADCAST(&hh[(ldh*3)+1]);
	__AVX_DATATYPE h_4_2 = _AVX_BROADCAST(&hh[(ldh*3)+2]);
	__AVX_DATATYPE h_4_1 = _AVX_BROADCAST(&hh[(ldh*3)+3]);
#ifdef __ELPA_USE_FMA__
	register __AVX_DATATYPE w1 = _AVX_FMA(a3_1, h_4_3, a4_1);
	w1 = _AVX_FMA(a2_1, h_4_2, w1);
	w1 = _AVX_FMA(a1_1, h_4_1, w1);
#else
	register __AVX_DATATYPE w1 = _AVX_ADD(a4_1, _AVX_MUL(a3_1, h_4_3));
	w1 = _AVX_ADD(w1, _AVX_MUL(a2_1, h_4_2));
	w1 = _AVX_ADD(w1, _AVX_MUL(a1_1, h_4_1));
#endif
	__AVX_DATATYPE h_2_1 = _AVX_BROADCAST(&hh[ldh+1]);
	__AVX_DATATYPE h_3_2 = _AVX_BROADCAST(&hh[(ldh*2)+1]);
	__AVX_DATATYPE h_3_1 = _AVX_BROADCAST(&hh[(ldh*2)+2]);
#ifdef __ELPA_USE_FMA__
	register __AVX_DATATYPE z1 = _AVX_FMA(a2_1, h_3_2, a3_1);
	z1 = _AVX_FMA(a1_1, h_3_1, z1);
	register __AVX_DATATYPE y1 = _AVX_FMA(a1_1, h_2_1, a2_1);
#else
	register __AVX_DATATYPE z1 = _AVX_ADD(a3_1, _AVX_MUL(a2_1, h_3_2));
	z1 = _AVX_ADD(z1, _AVX_MUL(a1_1, h_3_1));
	register __AVX_DATATYPE y1 = _AVX_ADD(a2_1, _AVX_MUL(a1_1, h_2_1));
#endif
	register __AVX_DATATYPE x1 = a1_1;

	__AVX_DATATYPE q1;

	__AVX_DATATYPE h1;
	__AVX_DATATYPE h2;
	__AVX_DATATYPE h3;
	__AVX_DATATYPE h4;
	__AVX_DATATYPE h5;
	__AVX_DATATYPE h6;

	for(i = 6; i < nb; i++)
	{
		h1 = _AVX_BROADCAST(&hh[i-5]);
		q1 = _mm256_castps128_ps256(_mm_load_ps(&q[i*ldq]));        // q(1,i) | q(2,i) ... | q(4,i)
#ifdef __ELPA_USE_FMA__
		x1 = _AVX_FMA(q1, h1, x1);
#else
		x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
#endif
		h2 = _AVX_BROADCAST(&hh[ldh+i-4]);
#ifdef __ELPA_USE_FMA__
		y1 = _AVX_FMA(q1, h2, y1);
#else
		y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
#endif
		h3 = _AVX_BROADCAST(&hh[(ldh*2)+i-3]);
#ifdef __ELPA_USE_FMA__
		z1 = _AVX_FMA(q1, h3, z1);
#else
		z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
#endif
		h4 = _AVX_BROADCAST(&hh[(ldh*3)+i-2]);
#ifdef __ELPA_USE_FMA__
		w1 = _AVX_FMA(q1, h4, w1);
#else
		w1 = _AVX_ADD(w1, _AVX_MUL(q1,h4));
#endif
		h5 = _AVX_BROADCAST(&hh[(ldh*4)+i-1]);
#ifdef __ELPA_USE_FMA__
		v1 = _AVX_FMA(q1, h5, v1);
#else
		v1 = _AVX_ADD(v1, _AVX_MUL(q1,h5));
#endif
		h6 = _AVX_BROADCAST(&hh[(ldh*5)+i]);
#ifdef __ELPA_USE_FMA__
		t1 = _AVX_FMA(q1, h6, t1);
#else
		t1 = _AVX_ADD(t1, _AVX_MUL(q1,h6));
#endif
	}

	h1 = _AVX_BROADCAST(&hh[nb-5]);
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[nb*ldq]));
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-4]);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMA(q1, h2, y1);
#else
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-3]);
#ifdef __ELPA_USE_FMA__
	z1 = _AVX_FMA(q1, h3, z1);
#else
	z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+nb-2]);
#ifdef __ELPA_USE_FMA__
	w1 = _AVX_FMA(q1, h4, w1);
#else
	w1 = _AVX_ADD(w1, _AVX_MUL(q1,h4));
#endif
	h5 = _AVX_BROADCAST(&hh[(ldh*4)+nb-1]);
#ifdef __ELPA_USE_FMA__
	v1 = _AVX_FMA(q1, h5, v1);
#else
	v1 = _AVX_ADD(v1, _AVX_MUL(q1,h5));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-4]);
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[(nb+1)*ldq]));
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-3]);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMA(q1, h2, y1);
#else
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-2]);
#ifdef __ELPA_USE_FMA__
	z1 = _AVX_FMA(q1, h3, z1);
#else
	z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+nb-1]);
#ifdef __ELPA_USE_FMA__
	w1 = _AVX_FMA(q1, h4, w1);
#else
	w1 = _AVX_ADD(w1, _AVX_MUL(q1,h4));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-3]);
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[(nb+2)*ldq]));
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-2]);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMA(q1, h2, y1);
#else
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-1]);
#ifdef __ELPA_USE_FMA__
	z1 = _AVX_FMA(q1, h3, z1);
#else
	z1 = _AVX_ADD(z1, _AVX_MUL(q1,h3));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-2]);
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[(nb+3)*ldq]));
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-1]);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMA(q1, h2, y1);
#else
	y1 = _AVX_ADD(y1, _AVX_MUL(q1,h2));
#endif

	h1 = _AVX_BROADCAST(&hh[nb-1]);
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[(nb+4)*ldq]));
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMA(q1, h1, x1);
#else
	x1 = _AVX_ADD(x1, _AVX_MUL(q1,h1));
#endif

	/////////////////////////////////////////////////////
	// Apply tau, correct wrong calculation using pre-calculated scalar products
	/////////////////////////////////////////////////////

	__AVX_DATATYPE tau1 = _AVX_BROADCAST(&hh[0]);
	x1 = _AVX_MUL(x1, tau1);

	__AVX_DATATYPE tau2 = _AVX_BROADCAST(&hh[ldh]);
	__AVX_DATATYPE vs_1_2 = _AVX_BROADCAST(&scalarprods[0]);
	h2 = _AVX_MUL(tau2, vs_1_2);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMSUB(y1, tau2, _AVX_MUL(x1,h2));
#else
	y1 = _AVX_SUB(_AVX_MUL(y1,tau2), _AVX_MUL(x1,h2));
#endif

	__AVX_DATATYPE tau3 = _AVX_BROADCAST(&hh[ldh*2]);
	__AVX_DATATYPE vs_1_3 = _AVX_BROADCAST(&scalarprods[1]);
	__AVX_DATATYPE vs_2_3 = _AVX_BROADCAST(&scalarprods[2]);
	h2 = _AVX_MUL(tau3, vs_1_3);
	h3 = _AVX_MUL(tau3, vs_2_3);
#ifdef __ELPA_USE_FMA__
	z1 = _AVX_FMSUB(z1, tau3, _AVX_FMA(y1, h3, _AVX_MUL(x1,h2)));
#else
	z1 = _AVX_SUB(_AVX_MUL(z1,tau3), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2)));
#endif

	__AVX_DATATYPE tau4 = _AVX_BROADCAST(&hh[ldh*3]);
	__AVX_DATATYPE vs_1_4 = _AVX_BROADCAST(&scalarprods[3]);
	__AVX_DATATYPE vs_2_4 = _AVX_BROADCAST(&scalarprods[4]);
	h2 = _AVX_MUL(tau4, vs_1_4);
	h3 = _AVX_MUL(tau4, vs_2_4);
	__AVX_DATATYPE vs_3_4 = _AVX_BROADCAST(&scalarprods[5]);
	h4 = _AVX_MUL(tau4, vs_3_4);
#ifdef __ELPA_USE_FMA__
	w1 = _AVX_FMSUB(w1, tau4, _AVX_FMA(z1, h4, _AVX_FMA(y1, h3, _AVX_MUL(x1,h2))));
#else
	w1 = _AVX_SUB(_AVX_MUL(w1,tau4), _AVX_ADD(_AVX_MUL(z1,h4), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2))));
#endif

	__AVX_DATATYPE tau5 = _AVX_BROADCAST(&hh[ldh*4]);
	__AVX_DATATYPE vs_1_5 = _AVX_BROADCAST(&scalarprods[6]);
	__AVX_DATATYPE vs_2_5 = _AVX_BROADCAST(&scalarprods[7]);
	h2 = _AVX_MUL(tau5, vs_1_5);
	h3 = _AVX_MUL(tau5, vs_2_5);
	__AVX_DATATYPE vs_3_5 = _AVX_BROADCAST(&scalarprods[8]);
	__AVX_DATATYPE vs_4_5 = _AVX_BROADCAST(&scalarprods[9]);
	h4 = _AVX_MUL(tau5, vs_3_5);
	h5 = _AVX_MUL(tau5, vs_4_5);
#ifdef __ELPA_USE_FMA__
	v1 = _AVX_FMSUB(v1, tau5, _AVX_ADD(_AVX_FMA(w1, h5, _AVX_MUL(z1,h4)), _AVX_FMA(y1, h3, _AVX_MUL(x1,h2))));
#else
	v1 = _AVX_SUB(_AVX_MUL(v1,tau5), _AVX_ADD(_AVX_ADD(_AVX_MUL(w1,h5), _AVX_MUL(z1,h4)), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2))));
#endif

	__AVX_DATATYPE tau6 = _AVX_BROADCAST(&hh[ldh*5]);
	__AVX_DATATYPE vs_1_6 = _AVX_BROADCAST(&scalarprods[10]);
	__AVX_DATATYPE vs_2_6 = _AVX_BROADCAST(&scalarprods[11]);
	h2 = _AVX_MUL(tau6, vs_1_6);
	h3 = _AVX_MUL(tau6, vs_2_6);
	__AVX_DATATYPE vs_3_6 = _AVX_BROADCAST(&scalarprods[12]);
	__AVX_DATATYPE vs_4_6 = _AVX_BROADCAST(&scalarprods[13]);
	__AVX_DATATYPE vs_5_6 = _AVX_BROADCAST(&scalarprods[14]);
	h4 = _AVX_MUL(tau6, vs_3_6);
	h5 = _AVX_MUL(tau6, vs_4_6);
	h6 = _AVX_MUL(tau6, vs_5_6);
#ifdef __ELPA_USE_FMA__
	t1 = _AVX_FMSUB(t1, tau6, _AVX_FMA(v1, h6, _AVX_ADD(_AVX_FMA(w1, h5, _AVX_MUL(z1,h4)), _AVX_FMA(y1, h3, _AVX_MUL(x1,h2)))));
#else
	t1 = _AVX_SUB(_AVX_MUL(t1,tau6), _AVX_ADD( _AVX_MUL(v1,h6), _AVX_ADD(_AVX_ADD(_AVX_MUL(w1,h5), _AVX_MUL(z1,h4)), _AVX_ADD(_AVX_MUL(y1,h3), _AVX_MUL(x1,h2)))));
#endif

	/////////////////////////////////////////////////////
	// Rank-1 upsate of Q [4 x nb+3]
	/////////////////////////////////////////////////////

	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[0]));
	q1 = _AVX_SUB(q1, t1);
	_mm_store_ps(&q[0], _mm256_castps256_ps128(q1));

	h6 = _AVX_BROADCAST(&hh[(ldh*5)+1]);
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[ldq]));
	q1 = _AVX_SUB(q1, v1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(t1, h6, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
#endif
	_mm_store_ps(&q[ldq], _mm256_castps256_ps128(q1));

	h5 = _AVX_BROADCAST(&hh[(ldh*4)+1]);
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[ldq*2]));
	q1 = _AVX_SUB(q1, w1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(v1, h5, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
#endif
	h6 = _AVX_BROADCAST(&hh[(ldh*5)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(t1, h6, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
#endif
	_mm_store_ps(&q[ldq*2], _mm256_castps256_ps128(q1));

	h4 = _AVX_BROADCAST(&hh[(ldh*3)+1]);
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[ldq*3]));
	q1 = _AVX_SUB(q1, z1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(w1, h4, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
#endif
	h5 = _AVX_BROADCAST(&hh[(ldh*4)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(v1, h5, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
#endif
	h6 = _AVX_BROADCAST(&hh[(ldh*5)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(t1, h6, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
#endif
	_mm_store_ps(&q[ldq*3], _mm256_castps256_ps128(q1));

	h3 = _AVX_BROADCAST(&hh[(ldh*2)+1]);
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[ldq*4]));
	q1 = _AVX_SUB(q1, y1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(w1, h4, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
#endif
	h5 = _AVX_BROADCAST(&hh[(ldh*4)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(v1, h5, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
#endif
	h6 = _AVX_BROADCAST(&hh[(ldh*5)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(t1, h6, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
#endif
	_mm_store_ps(&q[ldq*4], _mm256_castps256_ps128(q1));

	h2 = _AVX_BROADCAST(&hh[(ldh)+1]);
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[ldq*5]));
	q1 = _AVX_SUB(q1, x1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(w1, h4, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
#endif
	h5 = _AVX_BROADCAST(&hh[(ldh*4)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(v1, h5, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
#endif
	h6 = _AVX_BROADCAST(&hh[(ldh*5)+5]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(t1, h6, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
#endif
	_mm_store_ps(&q[ldq*5], _mm256_castps256_ps128(q1));

	for (i = 6; i < nb; i++)
	{
		q1 = _mm256_castps128_ps256(_mm_load_ps(&q[i*ldq]));
		h1 = _AVX_BROADCAST(&hh[i-5]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(x1, h1, q1);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
#endif
		h2 = _AVX_BROADCAST(&hh[ldh+i-4]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(y1, h2, q1);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
#endif
		h3 = _AVX_BROADCAST(&hh[(ldh*2)+i-3]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(z1, h3, q1);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
#endif
		h4 = _AVX_BROADCAST(&hh[(ldh*3)+i-2]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(w1, h4, q1);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
#endif
		h5 = _AVX_BROADCAST(&hh[(ldh*4)+i-1]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(v1, h5, q1);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
#endif
		h6 = _AVX_BROADCAST(&hh[(ldh*5)+i]);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_NFMA(t1, h6, q1);
#else
		q1 = _AVX_SUB(q1, _AVX_MUL(t1, h6));
#endif
		_mm_store_ps(&q[i*ldq], _mm256_castps256_ps128(q1));
	}

	h1 = _AVX_BROADCAST(&hh[nb-5]);
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[nb*ldq]));
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-4]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-3]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(w1, h4, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
#endif
	h5 = _AVX_BROADCAST(&hh[(ldh*4)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(v1, h5, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(v1, h5));
#endif
	_mm_store_ps(&q[nb*ldq], _mm256_castps256_ps128(q1));

	h1 = _AVX_BROADCAST(&hh[nb-4]);
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[(nb+1)*ldq]));
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-3]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
#endif
	h4 = _AVX_BROADCAST(&hh[(ldh*3)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(w1, h4, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(w1, h4));
#endif
	_mm_store_ps(&q[(nb+1)*ldq], _mm256_castps256_ps128(q1));

	h1 = _AVX_BROADCAST(&hh[nb-3]);
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[(nb+2)*ldq]));
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
#endif
	h3 = _AVX_BROADCAST(&hh[(ldh*2)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(z1, h3, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(z1, h3));
#endif
	_mm_store_ps(&q[(nb+2)*ldq], _mm256_castps256_ps128(q1));

	h1 = _AVX_BROADCAST(&hh[nb-2]);
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[(nb+3)*ldq]));
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
#endif
	h2 = _AVX_BROADCAST(&hh[ldh+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(y1, h2, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(y1, h2));
#endif
	_mm_store_ps(&q[(nb+3)*ldq], _mm256_castps256_ps128(q1));

	h1 = _AVX_BROADCAST(&hh[nb-1]);
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[(nb+4)*ldq]));
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_NFMA(x1, h1, q1);
#else
	q1 = _AVX_SUB(q1, _AVX_MUL(x1, h1));
#endif
	_mm_store_ps(&q[(nb+4)*ldq], _mm256_castps256_ps128(q1));
}
#endif
