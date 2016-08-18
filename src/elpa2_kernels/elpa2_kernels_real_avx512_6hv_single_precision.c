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
//    - Max-Plack-Institut für Mathematik in den Naturwissenschaftrn,
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

#define __forceinline __attribute__((always_inline)) static

#ifdef HAVE_AVX512

#define __ELPA_USE_FMA__
#define _mm512_FMA_ps(a,b,c) _mm512_fmadd_ps(a,b,c)
#define _mm512_NFMA_ps(a,b,c) _mm512_fnmadd_ps(a,b,c)
#define _mm512_FMSUB_ps(a,b,c) _mm512_fmsub_ps(a,b,c)

#endif


//Forward declaration
//static void hh_trafo_kernel_4_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);

//static void hh_trafo_kernel_8_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);
static void hh_trafo_kernel_16_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);
//static void hh_trafo_kernel_24_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);
static void hh_trafo_kernel_32_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);
static void hh_trafo_kernel_48_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);
static void hh_trafo_kernel_64_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);

void hexa_hh_trafo_real_avx512_6hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);
/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine hexa_hh_trafo_real_avx512_6hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="hexa_hh_trafo_real_avx512_6hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)     :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

void hexa_hh_trafo_real_avx512_6hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;

	// calculating scalar products to compute
	// 6 householder vectors simultaneously
	float scalarprods[15];

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

	// calculate scalar product of first and fourth householder vector
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
	for (i = 0; i < nq-48; i+=64)
	{
		hh_trafo_kernel_64_AVX512_6hv_single(&q[i], hh, nb, ldq, ldh, scalarprods);
	}
	if (nq == i)
	{
		return;
	}
	if (nq-i == 48)
	{
		hh_trafo_kernel_48_AVX512_6hv_single(&q[i], hh, nb, ldq, ldh, scalarprods);
	}
	if (nq-i == 32)
	{
		hh_trafo_kernel_32_AVX512_6hv_single(&q[i], hh, nb, ldq, ldh, scalarprods);
	}
        else
	{
		hh_trafo_kernel_16_AVX512_6hv_single(&q[i], hh, nb, ldq, ldh, scalarprods);
	}

}


/**
 * Unrolled kernel that computes
 * 64 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_64_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [8 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m512 a1_1 = _mm512_load_ps(&q[ldq*5]);
	__m512 a2_1 = _mm512_load_ps(&q[ldq*4]);
	__m512 a3_1 = _mm512_load_ps(&q[ldq*3]);
	__m512 a4_1 = _mm512_load_ps(&q[ldq*2]);
	__m512 a5_1 = _mm512_load_ps(&q[ldq]);
	__m512 a6_1 = _mm512_load_ps(&q[0]);

	__m512 h_6_5 = _mm512_set1_ps(hh[(ldh*5)+1]);
	__m512 h_6_4 = _mm512_set1_ps(hh[(ldh*5)+2]);
	__m512 h_6_3 = _mm512_set1_ps(hh[(ldh*5)+3]);
	__m512 h_6_2 = _mm512_set1_ps(hh[(ldh*5)+4]);
	__m512 h_6_1 = _mm512_set1_ps(hh[(ldh*5)+5]);

//	register __m512d t1 = _mm512_FMA_ps(a5_1, h_6_5, a6_1);
        __m512 t1 = _mm512_FMA_ps(a5_1, h_6_5, a6_1);

	t1 = _mm512_FMA_ps(a4_1, h_6_4, t1);
	t1 = _mm512_FMA_ps(a3_1, h_6_3, t1);
	t1 = _mm512_FMA_ps(a2_1, h_6_2, t1);
	t1 = _mm512_FMA_ps(a1_1, h_6_1, t1);

	__m512 h_5_4 = _mm512_set1_ps(hh[(ldh*4)+1]);
	__m512 h_5_3 = _mm512_set1_ps(hh[(ldh*4)+2]);
	__m512 h_5_2 = _mm512_set1_ps(hh[(ldh*4)+3]);
	__m512 h_5_1 = _mm512_set1_ps(hh[(ldh*4)+4]);

//	register __m512d v1 = _mm512_FMA_ps(a4_1, h_5_4, a5_1);
        __m512 v1 = _mm512_FMA_ps(a4_1, h_5_4, a5_1);

	v1 = _mm512_FMA_ps(a3_1, h_5_3, v1);
	v1 = _mm512_FMA_ps(a2_1, h_5_2, v1);
	v1 = _mm512_FMA_ps(a1_1, h_5_1, v1);

	__m512 h_4_3 = _mm512_set1_ps(hh[(ldh*3)+1]);
	__m512 h_4_2 = _mm512_set1_ps(hh[(ldh*3)+2]);
	__m512 h_4_1 = _mm512_set1_ps(hh[(ldh*3)+3]);

//	register __m512d w1 = _mm512_FMA_ps(a3_1, h_4_3, a4_1);
        __m512 w1 = _mm512_FMA_ps(a3_1, h_4_3, a4_1);

	w1 = _mm512_FMA_ps(a2_1, h_4_2, w1);
	w1 = _mm512_FMA_ps(a1_1, h_4_1, w1);

	__m512 h_2_1 = _mm512_set1_ps(hh[ldh+1]);
	__m512 h_3_2 = _mm512_set1_ps(hh[(ldh*2)+1]);
	__m512 h_3_1 = _mm512_set1_ps(hh[(ldh*2)+2]);

//	register __m512d z1 = _mm512_FMA_ps(a2_1, h_3_2, a3_1);
        __m512 z1 = _mm512_FMA_ps(a2_1, h_3_2, a3_1);

	z1 = _mm512_FMA_ps(a1_1, h_3_1, z1);
//	register __m512d y1 = _mm512_FMA_ps(a1_1, h_2_1, a2_1);
        __m512 y1 = _mm512_FMA_ps(a1_1, h_2_1, a2_1);


//	register __m512d x1 = a1_1;
        __m512 x1 = a1_1;



	__m512 a1_2 = _mm512_load_ps(&q[(ldq*5)+16]);
	__m512 a2_2 = _mm512_load_ps(&q[(ldq*4)+16]);
	__m512 a3_2 = _mm512_load_ps(&q[(ldq*3)+16]);
	__m512 a4_2 = _mm512_load_ps(&q[(ldq*2)+16]);
	__m512 a5_2 = _mm512_load_ps(&q[(ldq)+16]);
	__m512 a6_2 = _mm512_load_ps(&q[0+16]);

//	register __m512d t2 = _mm512_FMA_ps(a5_2, h_6_5, a6_2);
         __m512 t2 = _mm512_FMA_ps(a5_2, h_6_5, a6_2);

	t2 = _mm512_FMA_ps(a4_2, h_6_4, t2);
	t2 = _mm512_FMA_ps(a3_2, h_6_3, t2);
	t2 = _mm512_FMA_ps(a2_2, h_6_2, t2);
	t2 = _mm512_FMA_ps(a1_2, h_6_1, t2);

//	register __m512d v2 = _mm512_FMA_ps(a4_2, h_5_4, a5_2);
        __m512 v2 = _mm512_FMA_ps(a4_2, h_5_4, a5_2);

	v2 = _mm512_FMA_ps(a3_2, h_5_3, v2);
	v2 = _mm512_FMA_ps(a2_2, h_5_2, v2);
	v2 = _mm512_FMA_ps(a1_2, h_5_1, v2);

//	register __m512d w2 = _mm512_FMA_ps(a3_2, h_4_3, a4_2);
        __m512 w2 = _mm512_FMA_ps(a3_2, h_4_3, a4_2);

	w2 = _mm512_FMA_ps(a2_2, h_4_2, w2);
	w2 = _mm512_FMA_ps(a1_2, h_4_1, w2);

//	register __m512d z2 = _mm512_FMA_ps(a2_2, h_3_2, a3_2);
         __m512 z2 = _mm512_FMA_ps(a2_2, h_3_2, a3_2);

	z2 = _mm512_FMA_ps(a1_2, h_3_1, z2);
//	register __m512d y2 = _mm512_FMA_ps(a1_2, h_2_1, a2_2);
        __m512 y2 = _mm512_FMA_ps(a1_2, h_2_1, a2_2);


//	register __m512d x2 = a1_2;
        __m512 x2 = a1_2;


	__m512 a1_3 = _mm512_load_ps(&q[(ldq*5)+32]);
	__m512 a2_3 = _mm512_load_ps(&q[(ldq*4)+32]);
	__m512 a3_3 = _mm512_load_ps(&q[(ldq*3)+32]);
	__m512 a4_3 = _mm512_load_ps(&q[(ldq*2)+32]);
	__m512 a5_3 = _mm512_load_ps(&q[(ldq)+32]);
	__m512 a6_3 = _mm512_load_ps(&q[0+32]);

//	register __m512d t3 = _mm512_FMA_ps(a5_3, h_6_5, a6_3);
        __m512 t3 = _mm512_FMA_ps(a5_3, h_6_5, a6_3);

	t3 = _mm512_FMA_ps(a4_3, h_6_4, t3);
	t3 = _mm512_FMA_ps(a3_3, h_6_3, t3);
	t3 = _mm512_FMA_ps(a2_3, h_6_2, t3);
	t3 = _mm512_FMA_ps(a1_3, h_6_1, t3);

//	register __m512d v3 = _mm512_FMA_ps(a4_3, h_5_4, a5_3);
        __m512 v3 = _mm512_FMA_ps(a4_3, h_5_4, a5_3);

	v3 = _mm512_FMA_ps(a3_3, h_5_3, v3);
	v3 = _mm512_FMA_ps(a2_3, h_5_2, v3);
	v3 = _mm512_FMA_ps(a1_3, h_5_1, v3);

//	register __m512d w3 = _mm512_FMA_ps(a3_3, h_4_3, a4_3);
        __m512 w3 = _mm512_FMA_ps(a3_3, h_4_3, a4_3);

	w3 = _mm512_FMA_ps(a2_3, h_4_2, w3);
	w3 = _mm512_FMA_ps(a1_3, h_4_1, w3);

//	register __m512d z3 = _mm512_FMA_ps(a2_3, h_3_2, a3_3);
        __m512 z3 = _mm512_FMA_ps(a2_3, h_3_2, a3_3);

	z3 = _mm512_FMA_ps(a1_3, h_3_1, z3);
//	register __m512d y3 = _mm512_FMA_ps(a1_3, h_2_1, a2_3);
        __m512 y3 = _mm512_FMA_ps(a1_3, h_2_1, a2_3);


//	register __m512d x3 = a1_3;
        __m512 x3 = a1_3;


	__m512 a1_4 = _mm512_load_ps(&q[(ldq*5)+48]);
	__m512 a2_4 = _mm512_load_ps(&q[(ldq*4)+48]);
	__m512 a3_4 = _mm512_load_ps(&q[(ldq*3)+48]);
	__m512 a4_4 = _mm512_load_ps(&q[(ldq*2)+48]);
	__m512 a5_4 = _mm512_load_ps(&q[(ldq)+48]);
	__m512 a6_4 = _mm512_load_ps(&q[0+48]);

//	register __m512d t4 = _mm512_FMA_ps(a5_4, h_6_5, a6_4);
        __m512 t4 = _mm512_FMA_ps(a5_4, h_6_5, a6_4);

	t4 = _mm512_FMA_ps(a4_4, h_6_4, t4);
	t4 = _mm512_FMA_ps(a3_4, h_6_3, t4);
	t4 = _mm512_FMA_ps(a2_4, h_6_2, t4);
	t4 = _mm512_FMA_ps(a1_4, h_6_1, t4);

//	register __m512d v4 = _mm512_FMA_ps(a4_4, h_5_4, a5_4);
        __m512 v4 = _mm512_FMA_ps(a4_4, h_5_4, a5_4);

	v4 = _mm512_FMA_ps(a3_4, h_5_3, v4);
	v4 = _mm512_FMA_ps(a2_4, h_5_2, v4);
	v4 = _mm512_FMA_ps(a1_4, h_5_1, v4);

//	register __m512d w4 = _mm512_FMA_ps(a3_4, h_4_3, a4_4);
        __m512 w4 = _mm512_FMA_ps(a3_4, h_4_3, a4_4);

	w4 = _mm512_FMA_ps(a2_4, h_4_2, w4);
	w4 = _mm512_FMA_ps(a1_4, h_4_1, w4);

//	register __m512d z4 = _mm512_FMA_ps(a2_4, h_3_2, a3_4);
        __m512 z4 = _mm512_FMA_ps(a2_4, h_3_2, a3_4);

	z4 = _mm512_FMA_ps(a1_4, h_3_1, z4);
//	register __m512d y4 = _mm512_FMA_ps(a1_4, h_2_1, a2_4);
         __m512 y4 = _mm512_FMA_ps(a1_4, h_2_1, a2_4);


//	register __m512d x4 = a1_4;
        __m512 x4 = a1_4;


	__m512 q1;
	__m512 q2;
	__m512 q3;
	__m512 q4;

	__m512 h1;
	__m512 h2;
	__m512 h3;
	__m512 h4;
	__m512 h5;
	__m512 h6;

	for(i = 6; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-5]);
		q1 = _mm512_load_ps(&q[i*ldq]);
		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
		q3 = _mm512_load_ps(&q[(i*ldq)+32]);
		q4 = _mm512_load_ps(&q[(i*ldq)+48]);

		x1 = _mm512_FMA_ps(q1, h1, x1);
		x2 = _mm512_FMA_ps(q2, h1, x2);
		x3 = _mm512_FMA_ps(q3, h1, x3);
		x4 = _mm512_FMA_ps(q4, h1, x4);

		h2 = _mm512_set1_ps(hh[ldh+i-4]);

		y1 = _mm512_FMA_ps(q1, h2, y1);
		y2 = _mm512_FMA_ps(q2, h2, y2);
		y3 = _mm512_FMA_ps(q3, h2, y3);
		y4 = _mm512_FMA_ps(q4, h2, y4);

		h3 = _mm512_set1_ps(hh[(ldh*2)+i-3]);

		z1 = _mm512_FMA_ps(q1, h3, z1);
		z2 = _mm512_FMA_ps(q2, h3, z2);
		z3 = _mm512_FMA_ps(q3, h3, z3);
		z4 = _mm512_FMA_ps(q4, h3, z4);

		h4 = _mm512_set1_ps(hh[(ldh*3)+i-2]);

		w1 = _mm512_FMA_ps(q1, h4, w1);
		w2 = _mm512_FMA_ps(q2, h4, w2);
		w3 = _mm512_FMA_ps(q3, h4, w3);
		w4 = _mm512_FMA_ps(q4, h4, w4);

		h5 = _mm512_set1_ps(hh[(ldh*4)+i-1]);

		v1 = _mm512_FMA_ps(q1, h5, v1);
		v2 = _mm512_FMA_ps(q2, h5, v2);
		v3 = _mm512_FMA_ps(q3, h5, v3);
		v4 = _mm512_FMA_ps(q4, h5, v4);

		h6 = _mm512_set1_ps(hh[(ldh*5)+i]);

		t1 = _mm512_FMA_ps(q1, h6, t1);
		t2 = _mm512_FMA_ps(q2, h6, t2);
		t3 = _mm512_FMA_ps(q3, h6, t3);
		t4 = _mm512_FMA_ps(q4, h6, t4);
	}

	h1 = _mm512_set1_ps(hh[nb-5]);
	q1 = _mm512_load_ps(&q[nb*ldq]);
	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);
	q4 = _mm512_load_ps(&q[(nb*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
	x3 = _mm512_FMA_ps(q3, h1, x3);
	x4 = _mm512_FMA_ps(q4, h1, x4);

	h2 = _mm512_set1_ps(hh[ldh+nb-4]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
	y2 = _mm512_FMA_ps(q2, h2, y2);
	y3 = _mm512_FMA_ps(q3, h2, y3);
	y4 = _mm512_FMA_ps(q4, h2, y4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-3]);

	z1 = _mm512_FMA_ps(q1, h3, z1);
	z2 = _mm512_FMA_ps(q2, h3, z2);
	z3 = _mm512_FMA_ps(q3, h3, z3);
	z4 = _mm512_FMA_ps(q4, h3, z4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+nb-2]);

	w1 = _mm512_FMA_ps(q1, h4, w1);
	w2 = _mm512_FMA_ps(q2, h4, w2);
	w3 = _mm512_FMA_ps(q3, h4, w3);
	w4 = _mm512_FMA_ps(q4, h4, w4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+nb-1]);

	v1 = _mm512_FMA_ps(q1, h5, v1);
	v2 = _mm512_FMA_ps(q2, h5, v2);
	v3 = _mm512_FMA_ps(q3, h5, v3);
	v4 = _mm512_FMA_ps(q4, h5, v4);

	h1 = _mm512_set1_ps(hh[nb-4]);

	q1 = _mm512_load_ps(&q[(nb+1)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+1)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+1)*ldq)+32]);
	q4 = _mm512_load_ps(&q[((nb+1)*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
	x3 = _mm512_FMA_ps(q3, h1, x3);
	x4 = _mm512_FMA_ps(q4, h1, x4);

	h2 = _mm512_set1_ps(hh[ldh+nb-3]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
	y2 = _mm512_FMA_ps(q2, h2, y2);
	y3 = _mm512_FMA_ps(q3, h2, y3);
	y4 = _mm512_FMA_ps(q4, h2, y4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-2]);

	z1 = _mm512_FMA_ps(q1, h3, z1);
	z2 = _mm512_FMA_ps(q2, h3, z2);
	z3 = _mm512_FMA_ps(q3, h3, z3);
	z4 = _mm512_FMA_ps(q4, h3, z4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+nb-1]);

	w1 = _mm512_FMA_ps(q1, h4, w1);
	w2 = _mm512_FMA_ps(q2, h4, w2);
	w3 = _mm512_FMA_ps(q3, h4, w3);
	w4 = _mm512_FMA_ps(q4, h4, w4);

	h1 = _mm512_set1_ps(hh[nb-3]);
	q1 = _mm512_load_ps(&q[(nb+2)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+2)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+2)*ldq)+32]);
	q4 = _mm512_load_ps(&q[((nb+2)*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
	x3 = _mm512_FMA_ps(q3, h1, x3);
	x4 = _mm512_FMA_ps(q4, h1, x4);

	h2 = _mm512_set1_ps(hh[ldh+nb-2]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
	y2 = _mm512_FMA_ps(q2, h2, y2);
	y3 = _mm512_FMA_ps(q3, h2, y3);
	y4 = _mm512_FMA_ps(q4, h2, y4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-1]);

	z1 = _mm512_FMA_ps(q1, h3, z1);
	z2 = _mm512_FMA_ps(q2, h3, z2);
	z3 = _mm512_FMA_ps(q3, h3, z3);
	z4 = _mm512_FMA_ps(q4, h3, z4);

	h1 = _mm512_set1_ps(hh[nb-2]);
	q1 = _mm512_load_ps(&q[(nb+3)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+3)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+3)*ldq)+32]);
	q4 = _mm512_load_ps(&q[((nb+3)*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
	x3 = _mm512_FMA_ps(q3, h1, x3);
	x4 = _mm512_FMA_ps(q4, h1, x4);

	h2 = _mm512_set1_ps(hh[ldh+nb-1]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
	y2 = _mm512_FMA_ps(q2, h2, y2);
	y3 = _mm512_FMA_ps(q3, h2, y3);
	y4 = _mm512_FMA_ps(q4, h2, y4);

	h1 = _mm512_set1_ps(hh[nb-1]);
	q1 = _mm512_load_ps(&q[(nb+4)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+4)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+4)*ldq)+32]);
	q4 = _mm512_load_ps(&q[((nb+4)*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
	x3 = _mm512_FMA_ps(q3, h1, x3);
	x4 = _mm512_FMA_ps(q4, h1, x4);

	/////////////////////////////////////////////////////
	// Apply tau, correct wrong calculation using pre-calculated scalar products
	/////////////////////////////////////////////////////

	__m512 tau1 = _mm512_set1_ps(hh[0]);
	x1 = _mm512_mul_ps(x1, tau1);
	x2 = _mm512_mul_ps(x2, tau1);
	x3 = _mm512_mul_ps(x3, tau1);
	x4 = _mm512_mul_ps(x4, tau1);

	__m512 tau2 = _mm512_set1_ps(hh[ldh]);
	__m512 vs_1_2 = _mm512_set1_ps(scalarprods[0]);
	h2 = _mm512_mul_ps(tau2, vs_1_2);

	y1 = _mm512_FMSUB_ps(y1, tau2, _mm512_mul_ps(x1,h2));
	y2 = _mm512_FMSUB_ps(y2, tau2, _mm512_mul_ps(x2,h2));
	y3 = _mm512_FMSUB_ps(y3, tau2, _mm512_mul_ps(x3,h2));
	y4 = _mm512_FMSUB_ps(y4, tau2, _mm512_mul_ps(x4,h2));

	__m512 tau3 = _mm512_set1_ps(hh[ldh*2]);
	__m512 vs_1_3 = _mm512_set1_ps(scalarprods[1]);
	__m512 vs_2_3 = _mm512_set1_ps(scalarprods[2]);

	h2 = _mm512_mul_ps(tau3, vs_1_3);
	h3 = _mm512_mul_ps(tau3, vs_2_3);

	z1 = _mm512_FMSUB_ps(z1, tau3, _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2)));
	z2 = _mm512_FMSUB_ps(z2, tau3, _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2)));
	z3 = _mm512_FMSUB_ps(z3, tau3, _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2)));
	z4 = _mm512_FMSUB_ps(z4, tau3, _mm512_FMA_ps(y4, h3, _mm512_mul_ps(x4,h2)));

	__m512 tau4 = _mm512_set1_ps(hh[ldh*3]);
	__m512 vs_1_4 = _mm512_set1_ps(scalarprods[3]);
	__m512 vs_2_4 = _mm512_set1_ps(scalarprods[4]);

	h2 = _mm512_mul_ps(tau4, vs_1_4);
	h3 = _mm512_mul_ps(tau4, vs_2_4);

	__m512 vs_3_4 = _mm512_set1_ps(scalarprods[5]);
	h4 = _mm512_mul_ps(tau4, vs_3_4);

	w1 = _mm512_FMSUB_ps(w1, tau4, _mm512_FMA_ps(z1, h4, _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2))));
	w2 = _mm512_FMSUB_ps(w2, tau4, _mm512_FMA_ps(z2, h4, _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2))));
	w3 = _mm512_FMSUB_ps(w3, tau4, _mm512_FMA_ps(z3, h4, _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2))));
	w4 = _mm512_FMSUB_ps(w4, tau4, _mm512_FMA_ps(z4, h4, _mm512_FMA_ps(y4, h3, _mm512_mul_ps(x4,h2))));

	__m512 tau5 = _mm512_set1_ps(hh[ldh*4]);
	__m512 vs_1_5 = _mm512_set1_ps(scalarprods[6]);
	__m512 vs_2_5 = _mm512_set1_ps(scalarprods[7]);

	h2 = _mm512_mul_ps(tau5, vs_1_5);
	h3 = _mm512_mul_ps(tau5, vs_2_5);

	__m512 vs_3_5 = _mm512_set1_ps(scalarprods[8]);
	__m512 vs_4_5 = _mm512_set1_ps(scalarprods[9]);

	h4 = _mm512_mul_ps(tau5, vs_3_5);
	h5 = _mm512_mul_ps(tau5, vs_4_5);

	v1 = _mm512_FMSUB_ps(v1, tau5, _mm512_add_ps(_mm512_FMA_ps(w1, h5, _mm512_mul_ps(z1,h4)), _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2))));
	v2 = _mm512_FMSUB_ps(v2, tau5, _mm512_add_ps(_mm512_FMA_ps(w2, h5, _mm512_mul_ps(z2,h4)), _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2))));
	v3 = _mm512_FMSUB_ps(v3, tau5, _mm512_add_ps(_mm512_FMA_ps(w3, h5, _mm512_mul_ps(z3,h4)), _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2))));
	v4 = _mm512_FMSUB_ps(v4, tau5, _mm512_add_ps(_mm512_FMA_ps(w4, h5, _mm512_mul_ps(z4,h4)), _mm512_FMA_ps(y4, h3, _mm512_mul_ps(x4,h2))));

	__m512 tau6 = _mm512_set1_ps(hh[ldh*5]);
	__m512 vs_1_6 = _mm512_set1_ps(scalarprods[10]);
	__m512 vs_2_6 = _mm512_set1_ps(scalarprods[11]);
	h2 = _mm512_mul_ps(tau6, vs_1_6);
	h3 = _mm512_mul_ps(tau6, vs_2_6);

	__m512 vs_3_6 = _mm512_set1_ps(scalarprods[12]);
	__m512 vs_4_6 = _mm512_set1_ps(scalarprods[13]);
	__m512 vs_5_6 = _mm512_set1_ps(scalarprods[14]);

	h4 = _mm512_mul_ps(tau6, vs_3_6);
	h5 = _mm512_mul_ps(tau6, vs_4_6);
	h6 = _mm512_mul_ps(tau6, vs_5_6);

	t1 = _mm512_FMSUB_ps(t1, tau6, _mm512_FMA_ps(v1, h6, _mm512_add_ps(_mm512_FMA_ps(w1, h5, _mm512_mul_ps(z1,h4)), _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2)))));
	t2 = _mm512_FMSUB_ps(t2, tau6, _mm512_FMA_ps(v2, h6, _mm512_add_ps(_mm512_FMA_ps(w2, h5, _mm512_mul_ps(z2,h4)), _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2)))));
	t3 = _mm512_FMSUB_ps(t3, tau6, _mm512_FMA_ps(v3, h6, _mm512_add_ps(_mm512_FMA_ps(w3, h5, _mm512_mul_ps(z3,h4)), _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2)))));
	t4 = _mm512_FMSUB_ps(t4, tau6, _mm512_FMA_ps(v4, h6, _mm512_add_ps(_mm512_FMA_ps(w4, h5, _mm512_mul_ps(z4,h4)), _mm512_FMA_ps(y4, h3, _mm512_mul_ps(x4,h2)))));


	/////////////////////////////////////////////////////
	// Rank-1 update of Q [8 x nb+3]
	/////////////////////////////////////////////////////

	q1 = _mm512_load_ps(&q[0]);
	q2 = _mm512_load_ps(&q[0+16]);
	q3 = _mm512_load_ps(&q[0+32]);
	q4 = _mm512_load_ps(&q[0+48]);

	q1 = _mm512_sub_ps(q1, t1);
	q2 = _mm512_sub_ps(q2, t2);
	q3 = _mm512_sub_ps(q3, t3);
	q4 = _mm512_sub_ps(q4, t4);

	_mm512_store_ps(&q[0],q1);
	_mm512_store_ps(&q[0+16],q2);
	_mm512_store_ps(&q[0+32],q3);
	_mm512_store_ps(&q[0+48],q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+1]);
	q1 = _mm512_load_ps(&q[ldq]);
	q2 = _mm512_load_ps(&q[ldq+16]);
	q3 = _mm512_load_ps(&q[ldq+32]);
	q4 = _mm512_load_ps(&q[ldq+48]);

	q1 = _mm512_sub_ps(q1, v1);
	q2 = _mm512_sub_ps(q2, v2);
	q3 = _mm512_sub_ps(q3, v3);
	q4 = _mm512_sub_ps(q4, v4);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
	q2 = _mm512_NFMA_ps(t2, h6, q2);
	q3 = _mm512_NFMA_ps(t3, h6, q3);
	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq],q1);
	_mm512_store_ps(&q[ldq+16],q2);
	_mm512_store_ps(&q[ldq+32],q3);
	_mm512_store_ps(&q[ldq+48],q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+1]);
	q1 = _mm512_load_ps(&q[ldq*2]);
	q2 = _mm512_load_ps(&q[(ldq*2)+16]);
	q3 = _mm512_load_ps(&q[(ldq*2)+32]);
	q4 = _mm512_load_ps(&q[(ldq*2)+48]);

	q1 = _mm512_sub_ps(q1, w1);
	q2 = _mm512_sub_ps(q2, w2);
	q3 = _mm512_sub_ps(q3, w3);
	q4 = _mm512_sub_ps(q4, w4);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
	q2 = _mm512_NFMA_ps(v2, h5, q2);
	q3 = _mm512_NFMA_ps(v3, h5, q3);
	q4 = _mm512_NFMA_ps(v4, h5, q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+2]);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
	q2 = _mm512_NFMA_ps(t2, h6, q2);
	q3 = _mm512_NFMA_ps(t3, h6, q3);
	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq*2],q1);
	_mm512_store_ps(&q[(ldq*2)+16],q2);
	_mm512_store_ps(&q[(ldq*2)+32],q3);
	_mm512_store_ps(&q[(ldq*2)+48],q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+1]);
	q1 = _mm512_load_ps(&q[ldq*3]);
	q2 = _mm512_load_ps(&q[(ldq*3)+16]);
	q3 = _mm512_load_ps(&q[(ldq*3)+32]);
	q4 = _mm512_load_ps(&q[(ldq*3)+48]);

	q1 = _mm512_sub_ps(q1, z1);
	q2 = _mm512_sub_ps(q2, z2);
	q3 = _mm512_sub_ps(q3, z3);
	q4 = _mm512_sub_ps(q4, z4);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
	q2 = _mm512_NFMA_ps(w2, h4, q2);
	q3 = _mm512_NFMA_ps(w3, h4, q3);
	q4 = _mm512_NFMA_ps(w4, h4, q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+2]);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
	q2 = _mm512_NFMA_ps(v2, h5, q2);
	q3 = _mm512_NFMA_ps(v3, h5, q3);
	q4 = _mm512_NFMA_ps(v4, h5, q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+3]);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
	q2 = _mm512_NFMA_ps(t2, h6, q2);
	q3 = _mm512_NFMA_ps(t3, h6, q3);
	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq*3],q1);
	_mm512_store_ps(&q[(ldq*3)+16],q2);
	_mm512_store_ps(&q[(ldq*3)+32],q3);
	_mm512_store_ps(&q[(ldq*3)+48],q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+1]);
	q1 = _mm512_load_ps(&q[ldq*4]);
	q2 = _mm512_load_ps(&q[(ldq*4)+16]);
	q3 = _mm512_load_ps(&q[(ldq*4)+32]);
	q4 = _mm512_load_ps(&q[(ldq*4)+48]);

	q1 = _mm512_sub_ps(q1, y1);
	q2 = _mm512_sub_ps(q2, y2);
	q3 = _mm512_sub_ps(q3, y3);
	q4 = _mm512_sub_ps(q4, y4);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
	q3 = _mm512_NFMA_ps(z3, h3, q3);
	q4 = _mm512_NFMA_ps(z4, h3, q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+2]);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
	q2 = _mm512_NFMA_ps(w2, h4, q2);
	q3 = _mm512_NFMA_ps(w3, h4, q3);
	q4 = _mm512_NFMA_ps(w4, h4, q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+3]);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
	q2 = _mm512_NFMA_ps(v2, h5, q2);
	q3 = _mm512_NFMA_ps(v3, h5, q3);
	q4 = _mm512_NFMA_ps(v4, h5, q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+4]);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
	q2 = _mm512_NFMA_ps(t2, h6, q2);
	q3 = _mm512_NFMA_ps(t3, h6, q3);
	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq*4],q1);
	_mm512_store_ps(&q[(ldq*4)+16],q2);
	_mm512_store_ps(&q[(ldq*4)+32],q3);
	_mm512_store_ps(&q[(ldq*4)+48],q4);

	h2 = _mm512_set1_ps(hh[(ldh)+1]);
	q1 = _mm512_load_ps(&q[ldq*5]);
	q2 = _mm512_load_ps(&q[(ldq*5)+16]);
	q3 = _mm512_load_ps(&q[(ldq*5)+32]);
	q4 = _mm512_load_ps(&q[(ldq*5)+48]);

	q1 = _mm512_sub_ps(q1, x1);
	q2 = _mm512_sub_ps(q2, x2);
	q3 = _mm512_sub_ps(q3, x3);
	q4 = _mm512_sub_ps(q4, x4);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
	q3 = _mm512_NFMA_ps(y3, h2, q3);
	q4 = _mm512_NFMA_ps(y4, h2, q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+2]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
	q3 = _mm512_NFMA_ps(z3, h3, q3);
	q4 = _mm512_NFMA_ps(z4, h3, q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+3]);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
	q2 = _mm512_NFMA_ps(w2, h4, q2);
	q3 = _mm512_NFMA_ps(w3, h4, q3);
	q4 = _mm512_NFMA_ps(w4, h4, q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+4]);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
	q2 = _mm512_NFMA_ps(v2, h5, q2);
	q3 = _mm512_NFMA_ps(v3, h5, q3);
	q4 = _mm512_NFMA_ps(v4, h5, q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+5]);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
	q2 = _mm512_NFMA_ps(t2, h6, q2);
	q3 = _mm512_NFMA_ps(t3, h6, q3);
	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq*5],q1);
	_mm512_store_ps(&q[(ldq*5)+16],q2);
	_mm512_store_ps(&q[(ldq*5)+32],q3);
	_mm512_store_ps(&q[(ldq*5)+48],q4);

	for (i = 6; i < nb; i++)
	{
		q1 = _mm512_load_ps(&q[i*ldq]);
		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
		q3 = _mm512_load_ps(&q[(i*ldq)+32]);
		q4 = _mm512_load_ps(&q[(i*ldq)+48]);

		h1 = _mm512_set1_ps(hh[i-5]);

		q1 = _mm512_NFMA_ps(x1, h1, q1);
		q2 = _mm512_NFMA_ps(x2, h1, q2);
		q3 = _mm512_NFMA_ps(x3, h1, q3);
		q4 = _mm512_NFMA_ps(x4, h1, q4);

		h2 = _mm512_set1_ps(hh[ldh+i-4]);

		q1 = _mm512_NFMA_ps(y1, h2, q1);
		q2 = _mm512_NFMA_ps(y2, h2, q2);
		q3 = _mm512_NFMA_ps(y3, h2, q3);
        	q4 = _mm512_NFMA_ps(y4, h2, q4);

		h3 = _mm512_set1_ps(hh[(ldh*2)+i-3]);

		q1 = _mm512_NFMA_ps(z1, h3, q1);
		q2 = _mm512_NFMA_ps(z2, h3, q2);
		q3 = _mm512_NFMA_ps(z3, h3, q3);
		q4 = _mm512_NFMA_ps(z4, h3, q4);

		h4 = _mm512_set1_ps(hh[(ldh*3)+i-2]);

		q1 = _mm512_NFMA_ps(w1, h4, q1);
		q2 = _mm512_NFMA_ps(w2, h4, q2);
		q3 = _mm512_NFMA_ps(w3, h4, q3);
		q4 = _mm512_NFMA_ps(w4, h4, q4);

		h5 = _mm512_set1_ps(hh[(ldh*4)+i-1]);

		q1 = _mm512_NFMA_ps(v1, h5, q1);
		q2 = _mm512_NFMA_ps(v2, h5, q2);
		q3 = _mm512_NFMA_ps(v3, h5, q3);
		q4 = _mm512_NFMA_ps(v4, h5, q4);

		h6 = _mm512_set1_ps(hh[(ldh*5)+i]);

		q1 = _mm512_NFMA_ps(t1, h6, q1);
		q2 = _mm512_NFMA_ps(t2, h6, q2);
		q3 = _mm512_NFMA_ps(t3, h6, q3);
		q4 = _mm512_NFMA_ps(t4, h6, q4);

		_mm512_store_ps(&q[i*ldq],q1);
		_mm512_store_ps(&q[(i*ldq)+16],q2);
		_mm512_store_ps(&q[(i*ldq)+32],q3);
		_mm512_store_ps(&q[(i*ldq)+48],q4);

	}

	h1 = _mm512_set1_ps(hh[nb-5]);
	q1 = _mm512_load_ps(&q[nb*ldq]);
	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);
	q4 = _mm512_load_ps(&q[(nb*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
	q3 = _mm512_NFMA_ps(x3, h1, q3);
	q4 = _mm512_NFMA_ps(x4, h1, q4);

	h2 = _mm512_set1_ps(hh[ldh+nb-4]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
	q3 = _mm512_NFMA_ps(y3, h2, q3);
	q4 = _mm512_NFMA_ps(y4, h2, q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-3]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
	q3 = _mm512_NFMA_ps(z3, h3, q3);
	q4 = _mm512_NFMA_ps(z4, h3, q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+nb-2]);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
	q2 = _mm512_NFMA_ps(w2, h4, q2);
	q3 = _mm512_NFMA_ps(w3, h4, q3);
	q4 = _mm512_NFMA_ps(w4, h4, q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+nb-1]);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
	q2 = _mm512_NFMA_ps(v2, h5, q2);
	q3 = _mm512_NFMA_ps(v3, h5, q3);
	q4 = _mm512_NFMA_ps(v4, h5, q4);

	_mm512_store_ps(&q[nb*ldq],q1);
	_mm512_store_ps(&q[(nb*ldq)+16],q2);
	_mm512_store_ps(&q[(nb*ldq)+32],q3);
	_mm512_store_ps(&q[(nb*ldq)+48],q4);

	h1 = _mm512_set1_ps(hh[nb-4]);
	q1 = _mm512_load_ps(&q[(nb+1)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+1)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+1)*ldq)+32]);
	q4 = _mm512_load_ps(&q[((nb+1)*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
	q3 = _mm512_NFMA_ps(x3, h1, q3);
	q4 = _mm512_NFMA_ps(x4, h1, q4);

	h2 = _mm512_set1_ps(hh[ldh+nb-3]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
	q3 = _mm512_NFMA_ps(y3, h2, q3);
	q4 = _mm512_NFMA_ps(y4, h2, q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-2]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
	q3 = _mm512_NFMA_ps(z3, h3, q3);
	q4 = _mm512_NFMA_ps(z4, h3, q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+nb-1]);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
	q2 = _mm512_NFMA_ps(w2, h4, q2);
	q3 = _mm512_NFMA_ps(w3, h4, q3);
	q4 = _mm512_NFMA_ps(w4, h4, q4);

	_mm512_store_ps(&q[(nb+1)*ldq],q1);
	_mm512_store_ps(&q[((nb+1)*ldq)+16],q2);
	_mm512_store_ps(&q[((nb+1)*ldq)+32],q3);
	_mm512_store_ps(&q[((nb+1)*ldq)+48],q4);

	h1 = _mm512_set1_ps(hh[nb-3]);
	q1 = _mm512_load_ps(&q[(nb+2)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+2)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+2)*ldq)+32]);
	q4 = _mm512_load_ps(&q[((nb+2)*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
	q3 = _mm512_NFMA_ps(x3, h1, q3);
	q4 = _mm512_NFMA_ps(x4, h1, q4);

	h2 = _mm512_set1_ps(hh[ldh+nb-2]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
	q3 = _mm512_NFMA_ps(y3, h2, q3);
	q4 = _mm512_NFMA_ps(y4, h2, q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-1]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
	q3 = _mm512_NFMA_ps(z3, h3, q3);
	q4 = _mm512_NFMA_ps(z4, h3, q4);

	_mm512_store_ps(&q[(nb+2)*ldq],q1);
	_mm512_store_ps(&q[((nb+2)*ldq)+16],q2);
	_mm512_store_ps(&q[((nb+2)*ldq)+32],q3);
	_mm512_store_ps(&q[((nb+2)*ldq)+48],q4);

	h1 = _mm512_set1_ps(hh[nb-2]);
	q1 = _mm512_load_ps(&q[(nb+3)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+3)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+3)*ldq)+32]);
	q4 = _mm512_load_ps(&q[((nb+3)*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
	q3 = _mm512_NFMA_ps(x3, h1, q3);
	q4 = _mm512_NFMA_ps(x4, h1, q4);

	h2 = _mm512_set1_ps(hh[ldh+nb-1]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
	q3 = _mm512_NFMA_ps(y3, h2, q3);
	q4 = _mm512_NFMA_ps(y4, h2, q4);

	_mm512_store_ps(&q[(nb+3)*ldq],q1);
	_mm512_store_ps(&q[((nb+3)*ldq)+16],q2);
	_mm512_store_ps(&q[((nb+3)*ldq)+32],q3);
	_mm512_store_ps(&q[((nb+3)*ldq)+48],q4);

	h1 = _mm512_set1_ps(hh[nb-1]);
	q1 = _mm512_load_ps(&q[(nb+4)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+4)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+4)*ldq)+32]);
	q4 = _mm512_load_ps(&q[((nb+4)*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
	q3 = _mm512_NFMA_ps(x3, h1, q3);
	q4 = _mm512_NFMA_ps(x4, h1, q4);

	_mm512_store_ps(&q[(nb+4)*ldq],q1);
	_mm512_store_ps(&q[((nb+4)*ldq)+16],q2);
	_mm512_store_ps(&q[((nb+4)*ldq)+32],q3);
	_mm512_store_ps(&q[((nb+4)*ldq)+48],q4);

}



/**
 * Unrolled kernel that computes
 * 48 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_48_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [8 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m512 a1_1 = _mm512_load_ps(&q[ldq*5]);
	__m512 a2_1 = _mm512_load_ps(&q[ldq*4]);
	__m512 a3_1 = _mm512_load_ps(&q[ldq*3]);
	__m512 a4_1 = _mm512_load_ps(&q[ldq*2]);
	__m512 a5_1 = _mm512_load_ps(&q[ldq]);
	__m512 a6_1 = _mm512_load_ps(&q[0]);

	__m512 h_6_5 = _mm512_set1_ps(hh[(ldh*5)+1]);
	__m512 h_6_4 = _mm512_set1_ps(hh[(ldh*5)+2]);
	__m512 h_6_3 = _mm512_set1_ps(hh[(ldh*5)+3]);
	__m512 h_6_2 = _mm512_set1_ps(hh[(ldh*5)+4]);
	__m512 h_6_1 = _mm512_set1_ps(hh[(ldh*5)+5]);

//	register __m512d t1 = _mm512_FMA_ps(a5_1, h_6_5, a6_1);
        __m512 t1 = _mm512_FMA_ps(a5_1, h_6_5, a6_1);

	t1 = _mm512_FMA_ps(a4_1, h_6_4, t1);
	t1 = _mm512_FMA_ps(a3_1, h_6_3, t1);
	t1 = _mm512_FMA_ps(a2_1, h_6_2, t1);
	t1 = _mm512_FMA_ps(a1_1, h_6_1, t1);

	__m512 h_5_4 = _mm512_set1_ps(hh[(ldh*4)+1]);
	__m512 h_5_3 = _mm512_set1_ps(hh[(ldh*4)+2]);
	__m512 h_5_2 = _mm512_set1_ps(hh[(ldh*4)+3]);
	__m512 h_5_1 = _mm512_set1_ps(hh[(ldh*4)+4]);

//	register __m512d v1 = _mm512_FMA_ps(a4_1, h_5_4, a5_1);
        __m512 v1 = _mm512_FMA_ps(a4_1, h_5_4, a5_1);

	v1 = _mm512_FMA_ps(a3_1, h_5_3, v1);
	v1 = _mm512_FMA_ps(a2_1, h_5_2, v1);
	v1 = _mm512_FMA_ps(a1_1, h_5_1, v1);

	__m512 h_4_3 = _mm512_set1_ps(hh[(ldh*3)+1]);
	__m512 h_4_2 = _mm512_set1_ps(hh[(ldh*3)+2]);
	__m512 h_4_1 = _mm512_set1_ps(hh[(ldh*3)+3]);

//	register __m512d w1 = _mm512_FMA_ps(a3_1, h_4_3, a4_1);
        __m512 w1 = _mm512_FMA_ps(a3_1, h_4_3, a4_1);

	w1 = _mm512_FMA_ps(a2_1, h_4_2, w1);
	w1 = _mm512_FMA_ps(a1_1, h_4_1, w1);

	__m512 h_2_1 = _mm512_set1_ps(hh[ldh+1]);
	__m512 h_3_2 = _mm512_set1_ps(hh[(ldh*2)+1]);
	__m512 h_3_1 = _mm512_set1_ps(hh[(ldh*2)+2]);

//	register __m512d z1 = _mm512_FMA_ps(a2_1, h_3_2, a3_1);
        __m512 z1 = _mm512_FMA_ps(a2_1, h_3_2, a3_1);

	z1 = _mm512_FMA_ps(a1_1, h_3_1, z1);
//	register __m512d y1 = _mm512_FMA_ps(a1_1, h_2_1, a2_1);
        __m512 y1 = _mm512_FMA_ps(a1_1, h_2_1, a2_1);


//	register __m512d x1 = a1_1;
        __m512 x1 = a1_1;

	__m512 a1_2 = _mm512_load_ps(&q[(ldq*5)+16]);
	__m512 a2_2 = _mm512_load_ps(&q[(ldq*4)+16]);
	__m512 a3_2 = _mm512_load_ps(&q[(ldq*3)+16]);
	__m512 a4_2 = _mm512_load_ps(&q[(ldq*2)+16]);
	__m512 a5_2 = _mm512_load_ps(&q[(ldq)+16]);
	__m512 a6_2 = _mm512_load_ps(&q[0+16]);

//	register __m512d t2 = _mm512_FMA_ps(a5_2, h_6_5, a6_2);
         __m512 t2 = _mm512_FMA_ps(a5_2, h_6_5, a6_2);

	t2 = _mm512_FMA_ps(a4_2, h_6_4, t2);
	t2 = _mm512_FMA_ps(a3_2, h_6_3, t2);
	t2 = _mm512_FMA_ps(a2_2, h_6_2, t2);
	t2 = _mm512_FMA_ps(a1_2, h_6_1, t2);

//	register __m512d v2 = _mm512_FMA_ps(a4_2, h_5_4, a5_2);
        __m512 v2 = _mm512_FMA_ps(a4_2, h_5_4, a5_2);

	v2 = _mm512_FMA_ps(a3_2, h_5_3, v2);
	v2 = _mm512_FMA_ps(a2_2, h_5_2, v2);
	v2 = _mm512_FMA_ps(a1_2, h_5_1, v2);

//	register __m512d w2 = _mm512_FMA_ps(a3_2, h_4_3, a4_2);
        __m512 w2 = _mm512_FMA_ps(a3_2, h_4_3, a4_2);

	w2 = _mm512_FMA_ps(a2_2, h_4_2, w2);
	w2 = _mm512_FMA_ps(a1_2, h_4_1, w2);

//	register __m512d z2 = _mm512_FMA_ps(a2_2, h_3_2, a3_2);
         __m512 z2 = _mm512_FMA_ps(a2_2, h_3_2, a3_2);

	z2 = _mm512_FMA_ps(a1_2, h_3_1, z2);
//	register __m512d y2 = _mm512_FMA_ps(a1_2, h_2_1, a2_2);
        __m512 y2 = _mm512_FMA_ps(a1_2, h_2_1, a2_2);


//	register __m512d x2 = a1_2;
        __m512 x2 = a1_2;

	__m512 a1_3 = _mm512_load_ps(&q[(ldq*5)+32]);
	__m512 a2_3 = _mm512_load_ps(&q[(ldq*4)+32]);
	__m512 a3_3 = _mm512_load_ps(&q[(ldq*3)+32]);
	__m512 a4_3 = _mm512_load_ps(&q[(ldq*2)+32]);
	__m512 a5_3 = _mm512_load_ps(&q[(ldq)+32]);
	__m512 a6_3 = _mm512_load_ps(&q[0+32]);

//	register __m512d t3 = _mm512_FMA_ps(a5_3, h_6_5, a6_3);
        __m512 t3 = _mm512_FMA_ps(a5_3, h_6_5, a6_3);

	t3 = _mm512_FMA_ps(a4_3, h_6_4, t3);
	t3 = _mm512_FMA_ps(a3_3, h_6_3, t3);
	t3 = _mm512_FMA_ps(a2_3, h_6_2, t3);
	t3 = _mm512_FMA_ps(a1_3, h_6_1, t3);

//	register __m512d v3 = _mm512_FMA_ps(a4_3, h_5_4, a5_3);
        __m512 v3 = _mm512_FMA_ps(a4_3, h_5_4, a5_3);

	v3 = _mm512_FMA_ps(a3_3, h_5_3, v3);
	v3 = _mm512_FMA_ps(a2_3, h_5_2, v3);
	v3 = _mm512_FMA_ps(a1_3, h_5_1, v3);

//	register __m512d w3 = _mm512_FMA_ps(a3_3, h_4_3, a4_3);
        __m512 w3 = _mm512_FMA_ps(a3_3, h_4_3, a4_3);

	w3 = _mm512_FMA_ps(a2_3, h_4_2, w3);
	w3 = _mm512_FMA_ps(a1_3, h_4_1, w3);

//	register __m512d z3 = _mm512_FMA_ps(a2_3, h_3_2, a3_3);
        __m512 z3 = _mm512_FMA_ps(a2_3, h_3_2, a3_3);

	z3 = _mm512_FMA_ps(a1_3, h_3_1, z3);
//	register __m512d y3 = _mm512_FMA_ps(a1_3, h_2_1, a2_3);
        __m512 y3 = _mm512_FMA_ps(a1_3, h_2_1, a2_3);


//	register __m512d x3 = a1_3;
        __m512 x3 = a1_3;


//	__m512 a1_4 = _mm512_load_ps(&q[(ldq*5)+48]);
//	__m512 a2_4 = _mm512_load_ps(&q[(ldq*4)+48]);
//	__m512 a3_4 = _mm512_load_ps(&q[(ldq*3)+48]);
//	__m512 a4_4 = _mm512_load_ps(&q[(ldq*2)+48]);
//	__m512 a5_4 = _mm512_load_ps(&q[(ldq)+48]);
//	__m512 a6_4 = _mm512_load_ps(&q[0+48]);
//
////	register __m512d t4 = _mm512_FMA_ps(a5_4, h_6_5, a6_4);
 //       __m512 t4 = _mm512_FMA_ps(a5_4, h_6_5, a6_4);
//
//	t4 = _mm512_FMA_ps(a4_4, h_6_4, t4);
//	t4 = _mm512_FMA_ps(a3_4, h_6_3, t4);
//	t4 = _mm512_FMA_ps(a2_4, h_6_2, t4);
//	t4 = _mm512_FMA_ps(a1_4, h_6_1, t4);
//
////	register __m512d v4 = _mm512_FMA_ps(a4_4, h_5_4, a5_4);
//        __m512 v4 = _mm512_FMA_ps(a4_4, h_5_4, a5_4);
//
//	v4 = _mm512_FMA_ps(a3_4, h_5_3, v4);
//	v4 = _mm512_FMA_ps(a2_4, h_5_2, v4);
//	v4 = _mm512_FMA_ps(a1_4, h_5_1, v4);
//
////	register __m512d w4 = _mm512_FMA_ps(a3_4, h_4_3, a4_4);
  //      __m512 w4 = _mm512_FMA_ps(a3_4, h_4_3, a4_4);
//
//	w4 = _mm512_FMA_ps(a2_4, h_4_2, w4);
//	w4 = _mm512_FMA_ps(a1_4, h_4_1, w4);
//
////	register __m512d z4 = _mm512_FMA_ps(a2_4, h_3_2, a3_4);
//        __m512 z4 = _mm512_FMA_ps(a2_4, h_3_2, a3_4);
//
//	z4 = _mm512_FMA_ps(a1_4, h_3_1, z4);
////	register __m512d y4 = _mm512_FMA_ps(a1_4, h_2_1, a2_4);
//         __m512 y4 = _mm512_FMA_ps(a1_4, h_2_1, a2_4);
//
//
////	register __m512d x4 = a1_4;
//        __m512 x4 = a1_4;


	__m512 q1;
	__m512 q2;
	__m512 q3;
//	__m512 q4;

	__m512 h1;
	__m512 h2;
	__m512 h3;
	__m512 h4;
	__m512 h5;
	__m512 h6;

	for(i = 6; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-5]);
		q1 = _mm512_load_ps(&q[i*ldq]);
		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
		q3 = _mm512_load_ps(&q[(i*ldq)+32]);
//		q4 = _mm512_load_ps(&q[(i*ldq)+48]);

		x1 = _mm512_FMA_ps(q1, h1, x1);
		x2 = _mm512_FMA_ps(q2, h1, x2);
		x3 = _mm512_FMA_ps(q3, h1, x3);
//		x4 = _mm512_FMA_ps(q4, h1, x4);

		h2 = _mm512_set1_ps(hh[ldh+i-4]);

		y1 = _mm512_FMA_ps(q1, h2, y1);
		y2 = _mm512_FMA_ps(q2, h2, y2);
		y3 = _mm512_FMA_ps(q3, h2, y3);
//		y4 = _mm512_FMA_ps(q4, h2, y4);

		h3 = _mm512_set1_ps(hh[(ldh*2)+i-3]);

		z1 = _mm512_FMA_ps(q1, h3, z1);
		z2 = _mm512_FMA_ps(q2, h3, z2);
		z3 = _mm512_FMA_ps(q3, h3, z3);
//		z4 = _mm512_FMA_ps(q4, h3, z4);

		h4 = _mm512_set1_ps(hh[(ldh*3)+i-2]);

		w1 = _mm512_FMA_ps(q1, h4, w1);
		w2 = _mm512_FMA_ps(q2, h4, w2);
		w3 = _mm512_FMA_ps(q3, h4, w3);
//		w4 = _mm512_FMA_ps(q4, h4, w4);

		h5 = _mm512_set1_ps(hh[(ldh*4)+i-1]);

		v1 = _mm512_FMA_ps(q1, h5, v1);
		v2 = _mm512_FMA_ps(q2, h5, v2);
		v3 = _mm512_FMA_ps(q3, h5, v3);
//		v4 = _mm512_FMA_ps(q4, h5, v4);

		h6 = _mm512_set1_ps(hh[(ldh*5)+i]);

		t1 = _mm512_FMA_ps(q1, h6, t1);
		t2 = _mm512_FMA_ps(q2, h6, t2);
		t3 = _mm512_FMA_ps(q3, h6, t3);
//		t4 = _mm512_FMA_ps(q4, h6, t4);
	}

	h1 = _mm512_set1_ps(hh[nb-5]);
	q1 = _mm512_load_ps(&q[nb*ldq]);
	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);
//	q4 = _mm512_load_ps(&q[(nb*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
	x3 = _mm512_FMA_ps(q3, h1, x3);
//	x4 = _mm512_FMA_ps(q4, h1, x4);

	h2 = _mm512_set1_ps(hh[ldh+nb-4]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
	y2 = _mm512_FMA_ps(q2, h2, y2);
	y3 = _mm512_FMA_ps(q3, h2, y3);
//	y4 = _mm512_FMA_ps(q4, h2, y4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-3]);

	z1 = _mm512_FMA_ps(q1, h3, z1);
	z2 = _mm512_FMA_ps(q2, h3, z2);
	z3 = _mm512_FMA_ps(q3, h3, z3);
//	z4 = _mm512_FMA_ps(q4, h3, z4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+nb-2]);

	w1 = _mm512_FMA_ps(q1, h4, w1);
	w2 = _mm512_FMA_ps(q2, h4, w2);
	w3 = _mm512_FMA_ps(q3, h4, w3);
//	w4 = _mm512_FMA_ps(q4, h4, w4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+nb-1]);

	v1 = _mm512_FMA_ps(q1, h5, v1);
	v2 = _mm512_FMA_ps(q2, h5, v2);
	v3 = _mm512_FMA_ps(q3, h5, v3);
//	v4 = _mm512_FMA_ps(q4, h5, v4);

	h1 = _mm512_set1_ps(hh[nb-4]);

	q1 = _mm512_load_ps(&q[(nb+1)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+1)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+1)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+1)*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
	x3 = _mm512_FMA_ps(q3, h1, x3);
//	x4 = _mm512_FMA_ps(q4, h1, x4);

	h2 = _mm512_set1_ps(hh[ldh+nb-3]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
	y2 = _mm512_FMA_ps(q2, h2, y2);
	y3 = _mm512_FMA_ps(q3, h2, y3);
//	y4 = _mm512_FMA_ps(q4, h2, y4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-2]);

	z1 = _mm512_FMA_ps(q1, h3, z1);
	z2 = _mm512_FMA_ps(q2, h3, z2);
	z3 = _mm512_FMA_ps(q3, h3, z3);
//	z4 = _mm512_FMA_ps(q4, h3, z4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+nb-1]);

	w1 = _mm512_FMA_ps(q1, h4, w1);
	w2 = _mm512_FMA_ps(q2, h4, w2);
	w3 = _mm512_FMA_ps(q3, h4, w3);
//	w4 = _mm512_FMA_ps(q4, h4, w4);

	h1 = _mm512_set1_ps(hh[nb-3]);
	q1 = _mm512_load_ps(&q[(nb+2)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+2)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+2)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+2)*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
	x3 = _mm512_FMA_ps(q3, h1, x3);
//	x4 = _mm512_FMA_ps(q4, h1, x4);

	h2 = _mm512_set1_ps(hh[ldh+nb-2]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
	y2 = _mm512_FMA_ps(q2, h2, y2);
	y3 = _mm512_FMA_ps(q3, h2, y3);
//	y4 = _mm512_FMA_ps(q4, h2, y4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-1]);

	z1 = _mm512_FMA_ps(q1, h3, z1);
	z2 = _mm512_FMA_ps(q2, h3, z2);
	z3 = _mm512_FMA_ps(q3, h3, z3);
//	z4 = _mm512_FMA_ps(q4, h3, z4);

	h1 = _mm512_set1_ps(hh[nb-2]);
	q1 = _mm512_load_ps(&q[(nb+3)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+3)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+3)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+3)*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
	x3 = _mm512_FMA_ps(q3, h1, x3);
//	x4 = _mm512_FMA_ps(q4, h1, x4);

	h2 = _mm512_set1_ps(hh[ldh+nb-1]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
	y2 = _mm512_FMA_ps(q2, h2, y2);
	y3 = _mm512_FMA_ps(q3, h2, y3);
//	y4 = _mm512_FMA_ps(q4, h2, y4);

	h1 = _mm512_set1_ps(hh[nb-1]);
	q1 = _mm512_load_ps(&q[(nb+4)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+4)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+4)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+4)*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
	x3 = _mm512_FMA_ps(q3, h1, x3);
//	x4 = _mm512_FMA_ps(q4, h1, x4);

	/////////////////////////////////////////////////////
	// Apply tau, correct wrong calculation using pre-calculated scalar products
	/////////////////////////////////////////////////////

	__m512 tau1 = _mm512_set1_ps(hh[0]);
	x1 = _mm512_mul_ps(x1, tau1);
	x2 = _mm512_mul_ps(x2, tau1);
	x3 = _mm512_mul_ps(x3, tau1);
//	x4 = _mm512_mul_ps(x4, tau1);

	__m512 tau2 = _mm512_set1_ps(hh[ldh]);
	__m512 vs_1_2 = _mm512_set1_ps(scalarprods[0]);
	h2 = _mm512_mul_ps(tau2, vs_1_2);

	y1 = _mm512_FMSUB_ps(y1, tau2, _mm512_mul_ps(x1,h2));
	y2 = _mm512_FMSUB_ps(y2, tau2, _mm512_mul_ps(x2,h2));
	y3 = _mm512_FMSUB_ps(y3, tau2, _mm512_mul_ps(x3,h2));
//	y4 = _mm512_FMSUB_ps(y4, tau2, _mm512_mul_ps(x4,h2));

	__m512 tau3 = _mm512_set1_ps(hh[ldh*2]);
	__m512 vs_1_3 = _mm512_set1_ps(scalarprods[1]);
	__m512 vs_2_3 = _mm512_set1_ps(scalarprods[2]);

	h2 = _mm512_mul_ps(tau3, vs_1_3);
	h3 = _mm512_mul_ps(tau3, vs_2_3);

	z1 = _mm512_FMSUB_ps(z1, tau3, _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2)));
	z2 = _mm512_FMSUB_ps(z2, tau3, _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2)));
	z3 = _mm512_FMSUB_ps(z3, tau3, _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2)));
//	z4 = _mm512_FMSUB_ps(z4, tau3, _mm512_FMA_ps(y4, h3, _mm512_mul_ps(x4,h2)));

	__m512 tau4 = _mm512_set1_ps(hh[ldh*3]);
	__m512 vs_1_4 = _mm512_set1_ps(scalarprods[3]);
	__m512 vs_2_4 = _mm512_set1_ps(scalarprods[4]);

	h2 = _mm512_mul_ps(tau4, vs_1_4);
	h3 = _mm512_mul_ps(tau4, vs_2_4);

	__m512 vs_3_4 = _mm512_set1_ps(scalarprods[5]);
	h4 = _mm512_mul_ps(tau4, vs_3_4);

	w1 = _mm512_FMSUB_ps(w1, tau4, _mm512_FMA_ps(z1, h4, _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2))));
	w2 = _mm512_FMSUB_ps(w2, tau4, _mm512_FMA_ps(z2, h4, _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2))));
	w3 = _mm512_FMSUB_ps(w3, tau4, _mm512_FMA_ps(z3, h4, _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2))));
//	w4 = _mm512_FMSUB_ps(w4, tau4, _mm512_FMA_ps(z4, h4, _mm512_FMA_ps(y4, h3, _mm512_mul_ps(x4,h2))));

	__m512 tau5 = _mm512_set1_ps(hh[ldh*4]);
	__m512 vs_1_5 = _mm512_set1_ps(scalarprods[6]);
	__m512 vs_2_5 = _mm512_set1_ps(scalarprods[7]);

	h2 = _mm512_mul_ps(tau5, vs_1_5);
	h3 = _mm512_mul_ps(tau5, vs_2_5);

	__m512 vs_3_5 = _mm512_set1_ps(scalarprods[8]);
	__m512 vs_4_5 = _mm512_set1_ps(scalarprods[9]);

	h4 = _mm512_mul_ps(tau5, vs_3_5);
	h5 = _mm512_mul_ps(tau5, vs_4_5);

	v1 = _mm512_FMSUB_ps(v1, tau5, _mm512_add_ps(_mm512_FMA_ps(w1, h5, _mm512_mul_ps(z1,h4)), _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2))));
	v2 = _mm512_FMSUB_ps(v2, tau5, _mm512_add_ps(_mm512_FMA_ps(w2, h5, _mm512_mul_ps(z2,h4)), _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2))));
	v3 = _mm512_FMSUB_ps(v3, tau5, _mm512_add_ps(_mm512_FMA_ps(w3, h5, _mm512_mul_ps(z3,h4)), _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2))));
//	v4 = _mm512_FMSUB_ps(v4, tau5, _mm512_add_ps(_mm512_FMA_ps(w4, h5, _mm512_mul_ps(z4,h4)), _mm512_FMA_ps(y4, h3, _mm512_mul_ps(x4,h2))));

	__m512 tau6 = _mm512_set1_ps(hh[ldh*5]);
	__m512 vs_1_6 = _mm512_set1_ps(scalarprods[10]);
	__m512 vs_2_6 = _mm512_set1_ps(scalarprods[11]);
	h2 = _mm512_mul_ps(tau6, vs_1_6);
	h3 = _mm512_mul_ps(tau6, vs_2_6);

	__m512 vs_3_6 = _mm512_set1_ps(scalarprods[12]);
	__m512 vs_4_6 = _mm512_set1_ps(scalarprods[13]);
	__m512 vs_5_6 = _mm512_set1_ps(scalarprods[14]);

	h4 = _mm512_mul_ps(tau6, vs_3_6);
	h5 = _mm512_mul_ps(tau6, vs_4_6);
	h6 = _mm512_mul_ps(tau6, vs_5_6);

	t1 = _mm512_FMSUB_ps(t1, tau6, _mm512_FMA_ps(v1, h6, _mm512_add_ps(_mm512_FMA_ps(w1, h5, _mm512_mul_ps(z1,h4)), _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2)))));
	t2 = _mm512_FMSUB_ps(t2, tau6, _mm512_FMA_ps(v2, h6, _mm512_add_ps(_mm512_FMA_ps(w2, h5, _mm512_mul_ps(z2,h4)), _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2)))));
	t3 = _mm512_FMSUB_ps(t3, tau6, _mm512_FMA_ps(v3, h6, _mm512_add_ps(_mm512_FMA_ps(w3, h5, _mm512_mul_ps(z3,h4)), _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2)))));
//	t4 = _mm512_FMSUB_ps(t4, tau6, _mm512_FMA_ps(v4, h6, _mm512_add_ps(_mm512_FMA_ps(w4, h5, _mm512_mul_ps(z4,h4)), _mm512_FMA_ps(y4, h3, _mm512_mul_ps(x4,h2)))));


	/////////////////////////////////////////////////////
	// Rank-1 update of Q [8 x nb+3]
	/////////////////////////////////////////////////////

	q1 = _mm512_load_ps(&q[0]);
	q2 = _mm512_load_ps(&q[0+16]);
	q3 = _mm512_load_ps(&q[0+32]);
//	q4 = _mm512_load_ps(&q[0+48]);

	q1 = _mm512_sub_ps(q1, t1);
	q2 = _mm512_sub_ps(q2, t2);
	q3 = _mm512_sub_ps(q3, t3);
//	q4 = _mm512_sub_ps(q4, t4);

	_mm512_store_ps(&q[0],q1);
	_mm512_store_ps(&q[0+16],q2);
	_mm512_store_ps(&q[0+32],q3);
//	_mm512_store_ps(&q[0+48],q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+1]);
	q1 = _mm512_load_ps(&q[ldq]);
	q2 = _mm512_load_ps(&q[ldq+16]);
	q3 = _mm512_load_ps(&q[ldq+32]);
//	q4 = _mm512_load_ps(&q[ldq+48]);

	q1 = _mm512_sub_ps(q1, v1);
	q2 = _mm512_sub_ps(q2, v2);
	q3 = _mm512_sub_ps(q3, v3);
//	q4 = _mm512_sub_ps(q4, v4);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
	q2 = _mm512_NFMA_ps(t2, h6, q2);
	q3 = _mm512_NFMA_ps(t3, h6, q3);
//	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq],q1);
	_mm512_store_ps(&q[ldq+16],q2);
	_mm512_store_ps(&q[ldq+32],q3);
//	_mm512_store_ps(&q[ldq+48],q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+1]);
	q1 = _mm512_load_ps(&q[ldq*2]);
	q2 = _mm512_load_ps(&q[(ldq*2)+16]);
	q3 = _mm512_load_ps(&q[(ldq*2)+32]);
//	q4 = _mm512_load_ps(&q[(ldq*2)+48]);

	q1 = _mm512_sub_ps(q1, w1);
	q2 = _mm512_sub_ps(q2, w2);
	q3 = _mm512_sub_ps(q3, w3);
//	q4 = _mm512_sub_ps(q4, w4);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
	q2 = _mm512_NFMA_ps(v2, h5, q2);
	q3 = _mm512_NFMA_ps(v3, h5, q3);
//	q4 = _mm512_NFMA_ps(v4, h5, q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+2]);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
	q2 = _mm512_NFMA_ps(t2, h6, q2);
	q3 = _mm512_NFMA_ps(t3, h6, q3);
//	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq*2],q1);
	_mm512_store_ps(&q[(ldq*2)+16],q2);
	_mm512_store_ps(&q[(ldq*2)+32],q3);
//	_mm512_store_ps(&q[(ldq*2)+48],q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+1]);
	q1 = _mm512_load_ps(&q[ldq*3]);
	q2 = _mm512_load_ps(&q[(ldq*3)+16]);
	q3 = _mm512_load_ps(&q[(ldq*3)+32]);
//	q4 = _mm512_load_ps(&q[(ldq*3)+48]);

	q1 = _mm512_sub_ps(q1, z1);
	q2 = _mm512_sub_ps(q2, z2);
	q3 = _mm512_sub_ps(q3, z3);
//	q4 = _mm512_sub_ps(q4, z4);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
	q2 = _mm512_NFMA_ps(w2, h4, q2);
	q3 = _mm512_NFMA_ps(w3, h4, q3);
//	q4 = _mm512_NFMA_ps(w4, h4, q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+2]);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
	q2 = _mm512_NFMA_ps(v2, h5, q2);
	q3 = _mm512_NFMA_ps(v3, h5, q3);
//	q4 = _mm512_NFMA_ps(v4, h5, q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+3]);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
	q2 = _mm512_NFMA_ps(t2, h6, q2);
	q3 = _mm512_NFMA_ps(t3, h6, q3);
//	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq*3],q1);
	_mm512_store_ps(&q[(ldq*3)+16],q2);
	_mm512_store_ps(&q[(ldq*3)+32],q3);
//	_mm512_store_ps(&q[(ldq*3)+48],q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+1]);
	q1 = _mm512_load_ps(&q[ldq*4]);
	q2 = _mm512_load_ps(&q[(ldq*4)+16]);
	q3 = _mm512_load_ps(&q[(ldq*4)+32]);
//	q4 = _mm512_load_ps(&q[(ldq*4)+48]);

	q1 = _mm512_sub_ps(q1, y1);
	q2 = _mm512_sub_ps(q2, y2);
	q3 = _mm512_sub_ps(q3, y3);
//	q4 = _mm512_sub_ps(q4, y4);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
	q3 = _mm512_NFMA_ps(z3, h3, q3);
//	q4 = _mm512_NFMA_ps(z4, h3, q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+2]);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
	q2 = _mm512_NFMA_ps(w2, h4, q2);
	q3 = _mm512_NFMA_ps(w3, h4, q3);
//	q4 = _mm512_NFMA_ps(w4, h4, q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+3]);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
	q2 = _mm512_NFMA_ps(v2, h5, q2);
	q3 = _mm512_NFMA_ps(v3, h5, q3);
//	q4 = _mm512_NFMA_ps(v4, h5, q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+4]);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
	q2 = _mm512_NFMA_ps(t2, h6, q2);
	q3 = _mm512_NFMA_ps(t3, h6, q3);
//	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq*4],q1);
	_mm512_store_ps(&q[(ldq*4)+16],q2);
	_mm512_store_ps(&q[(ldq*4)+32],q3);
//	_mm512_store_ps(&q[(ldq*4)+48],q4);

	h2 = _mm512_set1_ps(hh[(ldh)+1]);
	q1 = _mm512_load_ps(&q[ldq*5]);
	q2 = _mm512_load_ps(&q[(ldq*5)+16]);
	q3 = _mm512_load_ps(&q[(ldq*5)+32]);
//	q4 = _mm512_load_ps(&q[(ldq*5)+48]);

	q1 = _mm512_sub_ps(q1, x1);
	q2 = _mm512_sub_ps(q2, x2);
	q3 = _mm512_sub_ps(q3, x3);
//	q4 = _mm512_sub_ps(q4, x4);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
	q3 = _mm512_NFMA_ps(y3, h2, q3);
//	q4 = _mm512_NFMA_ps(y4, h2, q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+2]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
	q3 = _mm512_NFMA_ps(z3, h3, q3);
//	q4 = _mm512_NFMA_ps(z4, h3, q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+3]);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
	q2 = _mm512_NFMA_ps(w2, h4, q2);
	q3 = _mm512_NFMA_ps(w3, h4, q3);
//	q4 = _mm512_NFMA_ps(w4, h4, q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+4]);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
	q2 = _mm512_NFMA_ps(v2, h5, q2);
	q3 = _mm512_NFMA_ps(v3, h5, q3);
//	q4 = _mm512_NFMA_ps(v4, h5, q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+5]);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
	q2 = _mm512_NFMA_ps(t2, h6, q2);
	q3 = _mm512_NFMA_ps(t3, h6, q3);
//	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq*5],q1);
	_mm512_store_ps(&q[(ldq*5)+16],q2);
	_mm512_store_ps(&q[(ldq*5)+32],q3);
//	_mm512_store_ps(&q[(ldq*5)+48],q4);

	for (i = 6; i < nb; i++)
	{
		q1 = _mm512_load_ps(&q[i*ldq]);
		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
		q3 = _mm512_load_ps(&q[(i*ldq)+32]);
//		q4 = _mm512_load_ps(&q[(i*ldq)+48]);

		h1 = _mm512_set1_ps(hh[i-5]);

		q1 = _mm512_NFMA_ps(x1, h1, q1);
		q2 = _mm512_NFMA_ps(x2, h1, q2);
		q3 = _mm512_NFMA_ps(x3, h1, q3);
//		q4 = _mm512_NFMA_ps(x4, h1, q4);

		h2 = _mm512_set1_ps(hh[ldh+i-4]);

		q1 = _mm512_NFMA_ps(y1, h2, q1);
		q2 = _mm512_NFMA_ps(y2, h2, q2);
		q3 = _mm512_NFMA_ps(y3, h2, q3);
  //      	q4 = _mm512_NFMA_ps(y4, h2, q4);

		h3 = _mm512_set1_ps(hh[(ldh*2)+i-3]);

		q1 = _mm512_NFMA_ps(z1, h3, q1);
		q2 = _mm512_NFMA_ps(z2, h3, q2);
		q3 = _mm512_NFMA_ps(z3, h3, q3);
//		q4 = _mm512_NFMA_ps(z4, h3, q4);

		h4 = _mm512_set1_ps(hh[(ldh*3)+i-2]);

		q1 = _mm512_NFMA_ps(w1, h4, q1);
		q2 = _mm512_NFMA_ps(w2, h4, q2);
		q3 = _mm512_NFMA_ps(w3, h4, q3);
//		q4 = _mm512_NFMA_ps(w4, h4, q4);

		h5 = _mm512_set1_ps(hh[(ldh*4)+i-1]);

		q1 = _mm512_NFMA_ps(v1, h5, q1);
		q2 = _mm512_NFMA_ps(v2, h5, q2);
		q3 = _mm512_NFMA_ps(v3, h5, q3);
//		q4 = _mm512_NFMA_ps(v4, h5, q4);

		h6 = _mm512_set1_ps(hh[(ldh*5)+i]);

		q1 = _mm512_NFMA_ps(t1, h6, q1);
		q2 = _mm512_NFMA_ps(t2, h6, q2);
		q3 = _mm512_NFMA_ps(t3, h6, q3);
//		q4 = _mm512_NFMA_ps(t4, h6, q4);

		_mm512_store_ps(&q[i*ldq],q1);
		_mm512_store_ps(&q[(i*ldq)+16],q2);
		_mm512_store_ps(&q[(i*ldq)+32],q3);
//		_mm512_store_ps(&q[(i*ldq)+48],q4);

	}

	h1 = _mm512_set1_ps(hh[nb-5]);
	q1 = _mm512_load_ps(&q[nb*ldq]);
	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);
//	q4 = _mm512_load_ps(&q[(nb*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
	q3 = _mm512_NFMA_ps(x3, h1, q3);
//	q4 = _mm512_NFMA_ps(x4, h1, q4);

	h2 = _mm512_set1_ps(hh[ldh+nb-4]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
	q3 = _mm512_NFMA_ps(y3, h2, q3);
//	q4 = _mm512_NFMA_ps(y4, h2, q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-3]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
	q3 = _mm512_NFMA_ps(z3, h3, q3);
//	q4 = _mm512_NFMA_ps(z4, h3, q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+nb-2]);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
	q2 = _mm512_NFMA_ps(w2, h4, q2);
	q3 = _mm512_NFMA_ps(w3, h4, q3);
//	q4 = _mm512_NFMA_ps(w4, h4, q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+nb-1]);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
	q2 = _mm512_NFMA_ps(v2, h5, q2);
	q3 = _mm512_NFMA_ps(v3, h5, q3);
//	q4 = _mm512_NFMA_ps(v4, h5, q4);

	_mm512_store_ps(&q[nb*ldq],q1);
	_mm512_store_ps(&q[(nb*ldq)+16],q2);
	_mm512_store_ps(&q[(nb*ldq)+32],q3);
//	_mm512_store_ps(&q[(nb*ldq)+48],q4);

	h1 = _mm512_set1_ps(hh[nb-4]);
	q1 = _mm512_load_ps(&q[(nb+1)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+1)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+1)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+1)*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
	q3 = _mm512_NFMA_ps(x3, h1, q3);
//	q4 = _mm512_NFMA_ps(x4, h1, q4);

	h2 = _mm512_set1_ps(hh[ldh+nb-3]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
	q3 = _mm512_NFMA_ps(y3, h2, q3);
//	q4 = _mm512_NFMA_ps(y4, h2, q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-2]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
	q3 = _mm512_NFMA_ps(z3, h3, q3);
//	q4 = _mm512_NFMA_ps(z4, h3, q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+nb-1]);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
	q2 = _mm512_NFMA_ps(w2, h4, q2);
	q3 = _mm512_NFMA_ps(w3, h4, q3);
//	q4 = _mm512_NFMA_ps(w4, h4, q4);

	_mm512_store_ps(&q[(nb+1)*ldq],q1);
	_mm512_store_ps(&q[((nb+1)*ldq)+16],q2);
	_mm512_store_ps(&q[((nb+1)*ldq)+32],q3);
//	_mm512_store_ps(&q[((nb+1)*ldq)+48],q4);

	h1 = _mm512_set1_ps(hh[nb-3]);
	q1 = _mm512_load_ps(&q[(nb+2)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+2)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+2)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+2)*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
	q3 = _mm512_NFMA_ps(x3, h1, q3);
//	q4 = _mm512_NFMA_ps(x4, h1, q4);

	h2 = _mm512_set1_ps(hh[ldh+nb-2]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
	q3 = _mm512_NFMA_ps(y3, h2, q3);
//	q4 = _mm512_NFMA_ps(y4, h2, q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-1]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
	q3 = _mm512_NFMA_ps(z3, h3, q3);
//	q4 = _mm512_NFMA_ps(z4, h3, q4);

	_mm512_store_ps(&q[(nb+2)*ldq],q1);
	_mm512_store_ps(&q[((nb+2)*ldq)+16],q2);
	_mm512_store_ps(&q[((nb+2)*ldq)+32],q3);
//	_mm512_store_ps(&q[((nb+2)*ldq)+48],q4);

	h1 = _mm512_set1_ps(hh[nb-2]);
	q1 = _mm512_load_ps(&q[(nb+3)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+3)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+3)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+3)*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
	q3 = _mm512_NFMA_ps(x3, h1, q3);
//	q4 = _mm512_NFMA_ps(x4, h1, q4);

	h2 = _mm512_set1_ps(hh[ldh+nb-1]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
	q3 = _mm512_NFMA_ps(y3, h2, q3);
//	q4 = _mm512_NFMA_ps(y4, h2, q4);

	_mm512_store_ps(&q[(nb+3)*ldq],q1);
	_mm512_store_ps(&q[((nb+3)*ldq)+16],q2);
	_mm512_store_ps(&q[((nb+3)*ldq)+32],q3);
//	_mm512_store_ps(&q[((nb+3)*ldq)+48],q4);

	h1 = _mm512_set1_ps(hh[nb-1]);
	q1 = _mm512_load_ps(&q[(nb+4)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+4)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+4)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+4)*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
	q3 = _mm512_NFMA_ps(x3, h1, q3);
//	q4 = _mm512_NFMA_ps(x4, h1, q4);

	_mm512_store_ps(&q[(nb+4)*ldq],q1);
	_mm512_store_ps(&q[((nb+4)*ldq)+16],q2);
	_mm512_store_ps(&q[((nb+4)*ldq)+32],q3);
//	_mm512_store_ps(&q[((nb+4)*ldq)+48],q4);

}


/**
 * Unrolled kernel that computes
 * 32 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_32_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [8 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m512 a1_1 = _mm512_load_ps(&q[ldq*5]);
	__m512 a2_1 = _mm512_load_ps(&q[ldq*4]);
	__m512 a3_1 = _mm512_load_ps(&q[ldq*3]);
	__m512 a4_1 = _mm512_load_ps(&q[ldq*2]);
	__m512 a5_1 = _mm512_load_ps(&q[ldq]);
	__m512 a6_1 = _mm512_load_ps(&q[0]);

	__m512 h_6_5 = _mm512_set1_ps(hh[(ldh*5)+1]);
	__m512 h_6_4 = _mm512_set1_ps(hh[(ldh*5)+2]);
	__m512 h_6_3 = _mm512_set1_ps(hh[(ldh*5)+3]);
	__m512 h_6_2 = _mm512_set1_ps(hh[(ldh*5)+4]);
	__m512 h_6_1 = _mm512_set1_ps(hh[(ldh*5)+5]);

//	register __m512d t1 = _mm512_FMA_ps(a5_1, h_6_5, a6_1);
        __m512 t1 = _mm512_FMA_ps(a5_1, h_6_5, a6_1);

	t1 = _mm512_FMA_ps(a4_1, h_6_4, t1);
	t1 = _mm512_FMA_ps(a3_1, h_6_3, t1);
	t1 = _mm512_FMA_ps(a2_1, h_6_2, t1);
	t1 = _mm512_FMA_ps(a1_1, h_6_1, t1);

	__m512 h_5_4 = _mm512_set1_ps(hh[(ldh*4)+1]);
	__m512 h_5_3 = _mm512_set1_ps(hh[(ldh*4)+2]);
	__m512 h_5_2 = _mm512_set1_ps(hh[(ldh*4)+3]);
	__m512 h_5_1 = _mm512_set1_ps(hh[(ldh*4)+4]);

//	register __m512d v1 = _mm512_FMA_ps(a4_1, h_5_4, a5_1);
        __m512 v1 = _mm512_FMA_ps(a4_1, h_5_4, a5_1);

	v1 = _mm512_FMA_ps(a3_1, h_5_3, v1);
	v1 = _mm512_FMA_ps(a2_1, h_5_2, v1);
	v1 = _mm512_FMA_ps(a1_1, h_5_1, v1);

	__m512 h_4_3 = _mm512_set1_ps(hh[(ldh*3)+1]);
	__m512 h_4_2 = _mm512_set1_ps(hh[(ldh*3)+2]);
	__m512 h_4_1 = _mm512_set1_ps(hh[(ldh*3)+3]);

//	register __m512d w1 = _mm512_FMA_ps(a3_1, h_4_3, a4_1);
        __m512 w1 = _mm512_FMA_ps(a3_1, h_4_3, a4_1);

	w1 = _mm512_FMA_ps(a2_1, h_4_2, w1);
	w1 = _mm512_FMA_ps(a1_1, h_4_1, w1);

	__m512 h_2_1 = _mm512_set1_ps(hh[ldh+1]);
	__m512 h_3_2 = _mm512_set1_ps(hh[(ldh*2)+1]);
	__m512 h_3_1 = _mm512_set1_ps(hh[(ldh*2)+2]);

//	register __m512d z1 = _mm512_FMA_ps(a2_1, h_3_2, a3_1);
        __m512 z1 = _mm512_FMA_ps(a2_1, h_3_2, a3_1);

	z1 = _mm512_FMA_ps(a1_1, h_3_1, z1);
//	register __m512d y1 = _mm512_FMA_ps(a1_1, h_2_1, a2_1);
        __m512 y1 = _mm512_FMA_ps(a1_1, h_2_1, a2_1);


//	register __m512d x1 = a1_1;
        __m512 x1 = a1_1;

	__m512 a1_2 = _mm512_load_ps(&q[(ldq*5)+16]);
	__m512 a2_2 = _mm512_load_ps(&q[(ldq*4)+16]);
	__m512 a3_2 = _mm512_load_ps(&q[(ldq*3)+16]);
	__m512 a4_2 = _mm512_load_ps(&q[(ldq*2)+16]);
	__m512 a5_2 = _mm512_load_ps(&q[(ldq)+16]);
	__m512 a6_2 = _mm512_load_ps(&q[0+16]);

//	register __m512d t2 = _mm512_FMA_ps(a5_2, h_6_5, a6_2);
         __m512 t2 = _mm512_FMA_ps(a5_2, h_6_5, a6_2);

	t2 = _mm512_FMA_ps(a4_2, h_6_4, t2);
	t2 = _mm512_FMA_ps(a3_2, h_6_3, t2);
	t2 = _mm512_FMA_ps(a2_2, h_6_2, t2);
	t2 = _mm512_FMA_ps(a1_2, h_6_1, t2);

//	register __m512d v2 = _mm512_FMA_ps(a4_2, h_5_4, a5_2);
        __m512 v2 = _mm512_FMA_ps(a4_2, h_5_4, a5_2);

	v2 = _mm512_FMA_ps(a3_2, h_5_3, v2);
	v2 = _mm512_FMA_ps(a2_2, h_5_2, v2);
	v2 = _mm512_FMA_ps(a1_2, h_5_1, v2);

//	register __m512d w2 = _mm512_FMA_ps(a3_2, h_4_3, a4_2);
        __m512 w2 = _mm512_FMA_ps(a3_2, h_4_3, a4_2);

	w2 = _mm512_FMA_ps(a2_2, h_4_2, w2);
	w2 = _mm512_FMA_ps(a1_2, h_4_1, w2);

//	register __m512d z2 = _mm512_FMA_ps(a2_2, h_3_2, a3_2);
         __m512 z2 = _mm512_FMA_ps(a2_2, h_3_2, a3_2);

	z2 = _mm512_FMA_ps(a1_2, h_3_1, z2);
//	register __m512d y2 = _mm512_FMA_ps(a1_2, h_2_1, a2_2);
        __m512 y2 = _mm512_FMA_ps(a1_2, h_2_1, a2_2);


//	register __m512d x2 = a1_2;
        __m512 x2 = a1_2;

//	__m512 a1_3 = _mm512_load_ps(&q[(ldq*5)+32]);
//	__m512 a2_3 = _mm512_load_ps(&q[(ldq*4)+32]);
//	__m512 a3_3 = _mm512_load_ps(&q[(ldq*3)+32]);
//	__m512 a4_3 = _mm512_load_ps(&q[(ldq*2)+32]);
//	__m512 a5_3 = _mm512_load_ps(&q[(ldq)+32]);
//	__m512 a6_3 = _mm512_load_ps(&q[0+32]);
//
////	register __m512d t3 = _mm512_FMA_ps(a5_3, h_6_5, a6_3);
//        __m512 t3 = _mm512_FMA_ps(a5_3, h_6_5, a6_3);
//
//	t3 = _mm512_FMA_ps(a4_3, h_6_4, t3);
//	t3 = _mm512_FMA_ps(a3_3, h_6_3, t3);
//	t3 = _mm512_FMA_ps(a2_3, h_6_2, t3);
//	t3 = _mm512_FMA_ps(a1_3, h_6_1, t3);
//
////	register __m512d v3 = _mm512_FMA_ps(a4_3, h_5_4, a5_3);
//        __m512 v3 = _mm512_FMA_ps(a4_3, h_5_4, a5_3);
//
//	v3 = _mm512_FMA_ps(a3_3, h_5_3, v3);
//	v3 = _mm512_FMA_ps(a2_3, h_5_2, v3);
//	v3 = _mm512_FMA_ps(a1_3, h_5_1, v3);
//
////	register __m512d w3 = _mm512_FMA_ps(a3_3, h_4_3, a4_3);
//        __m512 w3 = _mm512_FMA_ps(a3_3, h_4_3, a4_3);
//
//	w3 = _mm512_FMA_ps(a2_3, h_4_2, w3);
//	w3 = _mm512_FMA_ps(a1_3, h_4_1, w3);
//
////	register __m512d z3 = _mm512_FMA_ps(a2_3, h_3_2, a3_3);
//        __m512 z3 = _mm512_FMA_ps(a2_3, h_3_2, a3_3);
//
//	z3 = _mm512_FMA_ps(a1_3, h_3_1, z3);
////	register __m512d y3 = _mm512_FMA_ps(a1_3, h_2_1, a2_3);
//        __m512 y3 = _mm512_FMA_ps(a1_3, h_2_1, a2_3);
//
//
////	register __m512d x3 = a1_3;
//        __m512 x3 = a1_3;


//	__m512 a1_4 = _mm512_load_ps(&q[(ldq*5)+48]);
//	__m512 a2_4 = _mm512_load_ps(&q[(ldq*4)+48]);
//	__m512 a3_4 = _mm512_load_ps(&q[(ldq*3)+48]);
//	__m512 a4_4 = _mm512_load_ps(&q[(ldq*2)+48]);
//	__m512 a5_4 = _mm512_load_ps(&q[(ldq)+48]);
//	__m512 a6_4 = _mm512_load_ps(&q[0+48]);
//
////	register __m512d t4 = _mm512_FMA_ps(a5_4, h_6_5, a6_4);
 //       __m512 t4 = _mm512_FMA_ps(a5_4, h_6_5, a6_4);
//
//	t4 = _mm512_FMA_ps(a4_4, h_6_4, t4);
//	t4 = _mm512_FMA_ps(a3_4, h_6_3, t4);
//	t4 = _mm512_FMA_ps(a2_4, h_6_2, t4);
//	t4 = _mm512_FMA_ps(a1_4, h_6_1, t4);
//
////	register __m512d v4 = _mm512_FMA_ps(a4_4, h_5_4, a5_4);
//        __m512 v4 = _mm512_FMA_ps(a4_4, h_5_4, a5_4);
//
//	v4 = _mm512_FMA_ps(a3_4, h_5_3, v4);
//	v4 = _mm512_FMA_ps(a2_4, h_5_2, v4);
//	v4 = _mm512_FMA_ps(a1_4, h_5_1, v4);
//
////	register __m512d w4 = _mm512_FMA_ps(a3_4, h_4_3, a4_4);
  //      __m512 w4 = _mm512_FMA_ps(a3_4, h_4_3, a4_4);
//
//	w4 = _mm512_FMA_ps(a2_4, h_4_2, w4);
//	w4 = _mm512_FMA_ps(a1_4, h_4_1, w4);
//
////	register __m512d z4 = _mm512_FMA_ps(a2_4, h_3_2, a3_4);
//        __m512 z4 = _mm512_FMA_ps(a2_4, h_3_2, a3_4);
//
//	z4 = _mm512_FMA_ps(a1_4, h_3_1, z4);
////	register __m512d y4 = _mm512_FMA_ps(a1_4, h_2_1, a2_4);
//         __m512 y4 = _mm512_FMA_ps(a1_4, h_2_1, a2_4);
//
//
////	register __m512d x4 = a1_4;
//        __m512 x4 = a1_4;


	__m512 q1;
	__m512 q2;
//	__m512 q3;
//	__m512 q4;

	__m512 h1;
	__m512 h2;
	__m512 h3;
	__m512 h4;
	__m512 h5;
	__m512 h6;

	for(i = 6; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-5]);
		q1 = _mm512_load_ps(&q[i*ldq]);
		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
//		q3 = _mm512_load_ps(&q[(i*ldq)+32]);
//		q4 = _mm512_load_ps(&q[(i*ldq)+48]);

		x1 = _mm512_FMA_ps(q1, h1, x1);
		x2 = _mm512_FMA_ps(q2, h1, x2);
//		x3 = _mm512_FMA_ps(q3, h1, x3);
//		x4 = _mm512_FMA_ps(q4, h1, x4);

		h2 = _mm512_set1_ps(hh[ldh+i-4]);

		y1 = _mm512_FMA_ps(q1, h2, y1);
		y2 = _mm512_FMA_ps(q2, h2, y2);
//		y3 = _mm512_FMA_ps(q3, h2, y3);
//		y4 = _mm512_FMA_ps(q4, h2, y4);

		h3 = _mm512_set1_ps(hh[(ldh*2)+i-3]);

		z1 = _mm512_FMA_ps(q1, h3, z1);
		z2 = _mm512_FMA_ps(q2, h3, z2);
//		z3 = _mm512_FMA_ps(q3, h3, z3);
//		z4 = _mm512_FMA_ps(q4, h3, z4);

		h4 = _mm512_set1_ps(hh[(ldh*3)+i-2]);

		w1 = _mm512_FMA_ps(q1, h4, w1);
		w2 = _mm512_FMA_ps(q2, h4, w2);
//		w3 = _mm512_FMA_ps(q3, h4, w3);
//		w4 = _mm512_FMA_ps(q4, h4, w4);

		h5 = _mm512_set1_ps(hh[(ldh*4)+i-1]);

		v1 = _mm512_FMA_ps(q1, h5, v1);
		v2 = _mm512_FMA_ps(q2, h5, v2);
//		v3 = _mm512_FMA_ps(q3, h5, v3);
//		v4 = _mm512_FMA_ps(q4, h5, v4);

		h6 = _mm512_set1_ps(hh[(ldh*5)+i]);

		t1 = _mm512_FMA_ps(q1, h6, t1);
		t2 = _mm512_FMA_ps(q2, h6, t2);
//		t3 = _mm512_FMA_ps(q3, h6, t3);
//		t4 = _mm512_FMA_ps(q4, h6, t4);
	}

	h1 = _mm512_set1_ps(hh[nb-5]);
	q1 = _mm512_load_ps(&q[nb*ldq]);
	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
//	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);
//	q4 = _mm512_load_ps(&q[(nb*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
//	x3 = _mm512_FMA_ps(q3, h1, x3);
//	x4 = _mm512_FMA_ps(q4, h1, x4);

	h2 = _mm512_set1_ps(hh[ldh+nb-4]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
	y2 = _mm512_FMA_ps(q2, h2, y2);
//	y3 = _mm512_FMA_ps(q3, h2, y3);
//	y4 = _mm512_FMA_ps(q4, h2, y4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-3]);

	z1 = _mm512_FMA_ps(q1, h3, z1);
	z2 = _mm512_FMA_ps(q2, h3, z2);
//	z3 = _mm512_FMA_ps(q3, h3, z3);
//	z4 = _mm512_FMA_ps(q4, h3, z4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+nb-2]);

	w1 = _mm512_FMA_ps(q1, h4, w1);
	w2 = _mm512_FMA_ps(q2, h4, w2);
//	w3 = _mm512_FMA_ps(q3, h4, w3);
//	w4 = _mm512_FMA_ps(q4, h4, w4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+nb-1]);

	v1 = _mm512_FMA_ps(q1, h5, v1);
	v2 = _mm512_FMA_ps(q2, h5, v2);
//	v3 = _mm512_FMA_ps(q3, h5, v3);
//	v4 = _mm512_FMA_ps(q4, h5, v4);

	h1 = _mm512_set1_ps(hh[nb-4]);

	q1 = _mm512_load_ps(&q[(nb+1)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+1)*ldq)+16]);
//	q3 = _mm512_load_ps(&q[((nb+1)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+1)*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
//	x3 = _mm512_FMA_ps(q3, h1, x3);
//	x4 = _mm512_FMA_ps(q4, h1, x4);

	h2 = _mm512_set1_ps(hh[ldh+nb-3]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
	y2 = _mm512_FMA_ps(q2, h2, y2);
//	y3 = _mm512_FMA_ps(q3, h2, y3);
//	y4 = _mm512_FMA_ps(q4, h2, y4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-2]);

	z1 = _mm512_FMA_ps(q1, h3, z1);
	z2 = _mm512_FMA_ps(q2, h3, z2);
//	z3 = _mm512_FMA_ps(q3, h3, z3);
//	z4 = _mm512_FMA_ps(q4, h3, z4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+nb-1]);

	w1 = _mm512_FMA_ps(q1, h4, w1);
	w2 = _mm512_FMA_ps(q2, h4, w2);
//	w3 = _mm512_FMA_ps(q3, h4, w3);
//	w4 = _mm512_FMA_ps(q4, h4, w4);

	h1 = _mm512_set1_ps(hh[nb-3]);
	q1 = _mm512_load_ps(&q[(nb+2)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+2)*ldq)+16]);
//	q3 = _mm512_load_ps(&q[((nb+2)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+2)*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
//	x3 = _mm512_FMA_ps(q3, h1, x3);
//	x4 = _mm512_FMA_ps(q4, h1, x4);

	h2 = _mm512_set1_ps(hh[ldh+nb-2]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
	y2 = _mm512_FMA_ps(q2, h2, y2);
//	y3 = _mm512_FMA_ps(q3, h2, y3);
//	y4 = _mm512_FMA_ps(q4, h2, y4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-1]);

	z1 = _mm512_FMA_ps(q1, h3, z1);
	z2 = _mm512_FMA_ps(q2, h3, z2);
//	z3 = _mm512_FMA_ps(q3, h3, z3);
//	z4 = _mm512_FMA_ps(q4, h3, z4);

	h1 = _mm512_set1_ps(hh[nb-2]);
	q1 = _mm512_load_ps(&q[(nb+3)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+3)*ldq)+16]);
//	q3 = _mm512_load_ps(&q[((nb+3)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+3)*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
//	x3 = _mm512_FMA_ps(q3, h1, x3);
//	x4 = _mm512_FMA_ps(q4, h1, x4);

	h2 = _mm512_set1_ps(hh[ldh+nb-1]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
	y2 = _mm512_FMA_ps(q2, h2, y2);
//	y3 = _mm512_FMA_ps(q3, h2, y3);
//	y4 = _mm512_FMA_ps(q4, h2, y4);

	h1 = _mm512_set1_ps(hh[nb-1]);
	q1 = _mm512_load_ps(&q[(nb+4)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+4)*ldq)+16]);
//	q3 = _mm512_load_ps(&q[((nb+4)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+4)*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
//	x3 = _mm512_FMA_ps(q3, h1, x3);
//	x4 = _mm512_FMA_ps(q4, h1, x4);

	/////////////////////////////////////////////////////
	// Apply tau, correct wrong calculation using pre-calculated scalar products
	/////////////////////////////////////////////////////

	__m512 tau1 = _mm512_set1_ps(hh[0]);
	x1 = _mm512_mul_ps(x1, tau1);
	x2 = _mm512_mul_ps(x2, tau1);
//	x3 = _mm512_mul_ps(x3, tau1);
//	x4 = _mm512_mul_ps(x4, tau1);

	__m512 tau2 = _mm512_set1_ps(hh[ldh]);
	__m512 vs_1_2 = _mm512_set1_ps(scalarprods[0]);
	h2 = _mm512_mul_ps(tau2, vs_1_2);

	y1 = _mm512_FMSUB_ps(y1, tau2, _mm512_mul_ps(x1,h2));
	y2 = _mm512_FMSUB_ps(y2, tau2, _mm512_mul_ps(x2,h2));
//	y3 = _mm512_FMSUB_ps(y3, tau2, _mm512_mul_ps(x3,h2));
//	y4 = _mm512_FMSUB_ps(y4, tau2, _mm512_mul_ps(x4,h2));

	__m512 tau3 = _mm512_set1_ps(hh[ldh*2]);
	__m512 vs_1_3 = _mm512_set1_ps(scalarprods[1]);
	__m512 vs_2_3 = _mm512_set1_ps(scalarprods[2]);

	h2 = _mm512_mul_ps(tau3, vs_1_3);
	h3 = _mm512_mul_ps(tau3, vs_2_3);

	z1 = _mm512_FMSUB_ps(z1, tau3, _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2)));
	z2 = _mm512_FMSUB_ps(z2, tau3, _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2)));
//	z3 = _mm512_FMSUB_ps(z3, tau3, _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2)));
//	z4 = _mm512_FMSUB_ps(z4, tau3, _mm512_FMA_ps(y4, h3, _mm512_mul_ps(x4,h2)));

	__m512 tau4 = _mm512_set1_ps(hh[ldh*3]);
	__m512 vs_1_4 = _mm512_set1_ps(scalarprods[3]);
	__m512 vs_2_4 = _mm512_set1_ps(scalarprods[4]);

	h2 = _mm512_mul_ps(tau4, vs_1_4);
	h3 = _mm512_mul_ps(tau4, vs_2_4);

	__m512 vs_3_4 = _mm512_set1_ps(scalarprods[5]);
	h4 = _mm512_mul_ps(tau4, vs_3_4);

	w1 = _mm512_FMSUB_ps(w1, tau4, _mm512_FMA_ps(z1, h4, _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2))));
	w2 = _mm512_FMSUB_ps(w2, tau4, _mm512_FMA_ps(z2, h4, _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2))));
//	w3 = _mm512_FMSUB_ps(w3, tau4, _mm512_FMA_ps(z3, h4, _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2))));
//	w4 = _mm512_FMSUB_ps(w4, tau4, _mm512_FMA_ps(z4, h4, _mm512_FMA_ps(y4, h3, _mm512_mul_ps(x4,h2))));

	__m512 tau5 = _mm512_set1_ps(hh[ldh*4]);
	__m512 vs_1_5 = _mm512_set1_ps(scalarprods[6]);
	__m512 vs_2_5 = _mm512_set1_ps(scalarprods[7]);

	h2 = _mm512_mul_ps(tau5, vs_1_5);
	h3 = _mm512_mul_ps(tau5, vs_2_5);

	__m512 vs_3_5 = _mm512_set1_ps(scalarprods[8]);
	__m512 vs_4_5 = _mm512_set1_ps(scalarprods[9]);

	h4 = _mm512_mul_ps(tau5, vs_3_5);
	h5 = _mm512_mul_ps(tau5, vs_4_5);

	v1 = _mm512_FMSUB_ps(v1, tau5, _mm512_add_ps(_mm512_FMA_ps(w1, h5, _mm512_mul_ps(z1,h4)), _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2))));
	v2 = _mm512_FMSUB_ps(v2, tau5, _mm512_add_ps(_mm512_FMA_ps(w2, h5, _mm512_mul_ps(z2,h4)), _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2))));
//	v3 = _mm512_FMSUB_ps(v3, tau5, _mm512_add_ps(_mm512_FMA_ps(w3, h5, _mm512_mul_ps(z3,h4)), _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2))));
//	v4 = _mm512_FMSUB_ps(v4, tau5, _mm512_add_ps(_mm512_FMA_ps(w4, h5, _mm512_mul_ps(z4,h4)), _mm512_FMA_ps(y4, h3, _mm512_mul_ps(x4,h2))));

	__m512 tau6 = _mm512_set1_ps(hh[ldh*5]);
	__m512 vs_1_6 = _mm512_set1_ps(scalarprods[10]);
	__m512 vs_2_6 = _mm512_set1_ps(scalarprods[11]);
	h2 = _mm512_mul_ps(tau6, vs_1_6);
	h3 = _mm512_mul_ps(tau6, vs_2_6);

	__m512 vs_3_6 = _mm512_set1_ps(scalarprods[12]);
	__m512 vs_4_6 = _mm512_set1_ps(scalarprods[13]);
	__m512 vs_5_6 = _mm512_set1_ps(scalarprods[14]);

	h4 = _mm512_mul_ps(tau6, vs_3_6);
	h5 = _mm512_mul_ps(tau6, vs_4_6);
	h6 = _mm512_mul_ps(tau6, vs_5_6);

	t1 = _mm512_FMSUB_ps(t1, tau6, _mm512_FMA_ps(v1, h6, _mm512_add_ps(_mm512_FMA_ps(w1, h5, _mm512_mul_ps(z1,h4)), _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2)))));
	t2 = _mm512_FMSUB_ps(t2, tau6, _mm512_FMA_ps(v2, h6, _mm512_add_ps(_mm512_FMA_ps(w2, h5, _mm512_mul_ps(z2,h4)), _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2)))));
//	t3 = _mm512_FMSUB_ps(t3, tau6, _mm512_FMA_ps(v3, h6, _mm512_add_ps(_mm512_FMA_ps(w3, h5, _mm512_mul_ps(z3,h4)), _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2)))));
//	t4 = _mm512_FMSUB_ps(t4, tau6, _mm512_FMA_ps(v4, h6, _mm512_add_ps(_mm512_FMA_ps(w4, h5, _mm512_mul_ps(z4,h4)), _mm512_FMA_ps(y4, h3, _mm512_mul_ps(x4,h2)))));


	/////////////////////////////////////////////////////
	// Rank-1 update of Q [8 x nb+3]
	/////////////////////////////////////////////////////

	q1 = _mm512_load_ps(&q[0]);
	q2 = _mm512_load_ps(&q[0+16]);
//	q3 = _mm512_load_ps(&q[0+32]);
//	q4 = _mm512_load_ps(&q[0+48]);

	q1 = _mm512_sub_ps(q1, t1);
	q2 = _mm512_sub_ps(q2, t2);
//	q3 = _mm512_sub_ps(q3, t3);
//	q4 = _mm512_sub_ps(q4, t4);

	_mm512_store_ps(&q[0],q1);
	_mm512_store_ps(&q[0+16],q2);
//	_mm512_store_ps(&q[0+32],q3);
//	_mm512_store_ps(&q[0+48],q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+1]);
	q1 = _mm512_load_ps(&q[ldq]);
	q2 = _mm512_load_ps(&q[ldq+16]);
//	q3 = _mm512_load_ps(&q[ldq+32]);
//	q4 = _mm512_load_ps(&q[ldq+48]);

	q1 = _mm512_sub_ps(q1, v1);
	q2 = _mm512_sub_ps(q2, v2);
//	q3 = _mm512_sub_ps(q3, v3);
//	q4 = _mm512_sub_ps(q4, v4);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
	q2 = _mm512_NFMA_ps(t2, h6, q2);
//	q3 = _mm512_NFMA_ps(t3, h6, q3);
//	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq],q1);
	_mm512_store_ps(&q[ldq+16],q2);
//	_mm512_store_ps(&q[ldq+32],q3);
//	_mm512_store_ps(&q[ldq+48],q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+1]);
	q1 = _mm512_load_ps(&q[ldq*2]);
	q2 = _mm512_load_ps(&q[(ldq*2)+16]);
//	q3 = _mm512_load_ps(&q[(ldq*2)+32]);
//	q4 = _mm512_load_ps(&q[(ldq*2)+48]);

	q1 = _mm512_sub_ps(q1, w1);
	q2 = _mm512_sub_ps(q2, w2);
//	q3 = _mm512_sub_ps(q3, w3);
//	q4 = _mm512_sub_ps(q4, w4);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
	q2 = _mm512_NFMA_ps(v2, h5, q2);
//	q3 = _mm512_NFMA_ps(v3, h5, q3);
//	q4 = _mm512_NFMA_ps(v4, h5, q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+2]);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
	q2 = _mm512_NFMA_ps(t2, h6, q2);
//	q3 = _mm512_NFMA_ps(t3, h6, q3);
//	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq*2],q1);
	_mm512_store_ps(&q[(ldq*2)+16],q2);
//	_mm512_store_ps(&q[(ldq*2)+32],q3);
//	_mm512_store_ps(&q[(ldq*2)+48],q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+1]);
	q1 = _mm512_load_ps(&q[ldq*3]);
	q2 = _mm512_load_ps(&q[(ldq*3)+16]);
//	q3 = _mm512_load_ps(&q[(ldq*3)+32]);
//	q4 = _mm512_load_ps(&q[(ldq*3)+48]);

	q1 = _mm512_sub_ps(q1, z1);
	q2 = _mm512_sub_ps(q2, z2);
//	q3 = _mm512_sub_ps(q3, z3);
//	q4 = _mm512_sub_ps(q4, z4);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
	q2 = _mm512_NFMA_ps(w2, h4, q2);
//	q3 = _mm512_NFMA_ps(w3, h4, q3);
//	q4 = _mm512_NFMA_ps(w4, h4, q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+2]);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
	q2 = _mm512_NFMA_ps(v2, h5, q2);
//	q3 = _mm512_NFMA_ps(v3, h5, q3);
//	q4 = _mm512_NFMA_ps(v4, h5, q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+3]);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
	q2 = _mm512_NFMA_ps(t2, h6, q2);
//	q3 = _mm512_NFMA_ps(t3, h6, q3);
//	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq*3],q1);
	_mm512_store_ps(&q[(ldq*3)+16],q2);
//	_mm512_store_ps(&q[(ldq*3)+32],q3);
//	_mm512_store_ps(&q[(ldq*3)+48],q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+1]);
	q1 = _mm512_load_ps(&q[ldq*4]);
	q2 = _mm512_load_ps(&q[(ldq*4)+16]);
//	q3 = _mm512_load_ps(&q[(ldq*4)+32]);
//	q4 = _mm512_load_ps(&q[(ldq*4)+48]);

	q1 = _mm512_sub_ps(q1, y1);
	q2 = _mm512_sub_ps(q2, y2);
//	q3 = _mm512_sub_ps(q3, y3);
//	q4 = _mm512_sub_ps(q4, y4);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
//	q3 = _mm512_NFMA_ps(z3, h3, q3);
//	q4 = _mm512_NFMA_ps(z4, h3, q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+2]);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
	q2 = _mm512_NFMA_ps(w2, h4, q2);
//	q3 = _mm512_NFMA_ps(w3, h4, q3);
//	q4 = _mm512_NFMA_ps(w4, h4, q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+3]);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
	q2 = _mm512_NFMA_ps(v2, h5, q2);
//	q3 = _mm512_NFMA_ps(v3, h5, q3);
//	q4 = _mm512_NFMA_ps(v4, h5, q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+4]);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
	q2 = _mm512_NFMA_ps(t2, h6, q2);
//	q3 = _mm512_NFMA_ps(t3, h6, q3);
//	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq*4],q1);
	_mm512_store_ps(&q[(ldq*4)+16],q2);
//	_mm512_store_ps(&q[(ldq*4)+32],q3);
//	_mm512_store_ps(&q[(ldq*4)+48],q4);

	h2 = _mm512_set1_ps(hh[(ldh)+1]);
	q1 = _mm512_load_ps(&q[ldq*5]);
	q2 = _mm512_load_ps(&q[(ldq*5)+16]);
//	q3 = _mm512_load_ps(&q[(ldq*5)+32]);
//	q4 = _mm512_load_ps(&q[(ldq*5)+48]);

	q1 = _mm512_sub_ps(q1, x1);
	q2 = _mm512_sub_ps(q2, x2);
//	q3 = _mm512_sub_ps(q3, x3);
//	q4 = _mm512_sub_ps(q4, x4);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
//	q3 = _mm512_NFMA_ps(y3, h2, q3);
//	q4 = _mm512_NFMA_ps(y4, h2, q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+2]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
//	q3 = _mm512_NFMA_ps(z3, h3, q3);
//	q4 = _mm512_NFMA_ps(z4, h3, q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+3]);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
	q2 = _mm512_NFMA_ps(w2, h4, q2);
//	q3 = _mm512_NFMA_ps(w3, h4, q3);
//	q4 = _mm512_NFMA_ps(w4, h4, q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+4]);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
	q2 = _mm512_NFMA_ps(v2, h5, q2);
//	q3 = _mm512_NFMA_ps(v3, h5, q3);
//	q4 = _mm512_NFMA_ps(v4, h5, q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+5]);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
	q2 = _mm512_NFMA_ps(t2, h6, q2);
//	q3 = _mm512_NFMA_ps(t3, h6, q3);
//	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq*5],q1);
	_mm512_store_ps(&q[(ldq*5)+16],q2);
//	_mm512_store_ps(&q[(ldq*5)+32],q3);
//	_mm512_store_ps(&q[(ldq*5)+48],q4);

	for (i = 6; i < nb; i++)
	{
		q1 = _mm512_load_ps(&q[i*ldq]);
		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
//		q3 = _mm512_load_ps(&q[(i*ldq)+32]);
//		q4 = _mm512_load_ps(&q[(i*ldq)+48]);

		h1 = _mm512_set1_ps(hh[i-5]);

		q1 = _mm512_NFMA_ps(x1, h1, q1);
		q2 = _mm512_NFMA_ps(x2, h1, q2);
//		q3 = _mm512_NFMA_ps(x3, h1, q3);
//		q4 = _mm512_NFMA_ps(x4, h1, q4);

		h2 = _mm512_set1_ps(hh[ldh+i-4]);

		q1 = _mm512_NFMA_ps(y1, h2, q1);
		q2 = _mm512_NFMA_ps(y2, h2, q2);
//		q3 = _mm512_NFMA_ps(y3, h2, q3);
  //      	q4 = _mm512_NFMA_ps(y4, h2, q4);

		h3 = _mm512_set1_ps(hh[(ldh*2)+i-3]);

		q1 = _mm512_NFMA_ps(z1, h3, q1);
		q2 = _mm512_NFMA_ps(z2, h3, q2);
//		q3 = _mm512_NFMA_ps(z3, h3, q3);
//		q4 = _mm512_NFMA_ps(z4, h3, q4);

		h4 = _mm512_set1_ps(hh[(ldh*3)+i-2]);

		q1 = _mm512_NFMA_ps(w1, h4, q1);
		q2 = _mm512_NFMA_ps(w2, h4, q2);
//		q3 = _mm512_NFMA_ps(w3, h4, q3);
//		q4 = _mm512_NFMA_ps(w4, h4, q4);

		h5 = _mm512_set1_ps(hh[(ldh*4)+i-1]);

		q1 = _mm512_NFMA_ps(v1, h5, q1);
		q2 = _mm512_NFMA_ps(v2, h5, q2);
//		q3 = _mm512_NFMA_ps(v3, h5, q3);
//		q4 = _mm512_NFMA_ps(v4, h5, q4);

		h6 = _mm512_set1_ps(hh[(ldh*5)+i]);

		q1 = _mm512_NFMA_ps(t1, h6, q1);
		q2 = _mm512_NFMA_ps(t2, h6, q2);
//		q3 = _mm512_NFMA_ps(t3, h6, q3);
//		q4 = _mm512_NFMA_ps(t4, h6, q4);

		_mm512_store_ps(&q[i*ldq],q1);
		_mm512_store_ps(&q[(i*ldq)+16],q2);
//		_mm512_store_ps(&q[(i*ldq)+32],q3);
//		_mm512_store_ps(&q[(i*ldq)+48],q4);

	}

	h1 = _mm512_set1_ps(hh[nb-5]);
	q1 = _mm512_load_ps(&q[nb*ldq]);
	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
//	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);
//	q4 = _mm512_load_ps(&q[(nb*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
//	q3 = _mm512_NFMA_ps(x3, h1, q3);
//	q4 = _mm512_NFMA_ps(x4, h1, q4);

	h2 = _mm512_set1_ps(hh[ldh+nb-4]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
//	q3 = _mm512_NFMA_ps(y3, h2, q3);
//	q4 = _mm512_NFMA_ps(y4, h2, q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-3]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
//	q3 = _mm512_NFMA_ps(z3, h3, q3);
//	q4 = _mm512_NFMA_ps(z4, h3, q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+nb-2]);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
	q2 = _mm512_NFMA_ps(w2, h4, q2);
//	q3 = _mm512_NFMA_ps(w3, h4, q3);
//	q4 = _mm512_NFMA_ps(w4, h4, q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+nb-1]);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
	q2 = _mm512_NFMA_ps(v2, h5, q2);
//	q3 = _mm512_NFMA_ps(v3, h5, q3);
//	q4 = _mm512_NFMA_ps(v4, h5, q4);

	_mm512_store_ps(&q[nb*ldq],q1);
	_mm512_store_ps(&q[(nb*ldq)+16],q2);
//	_mm512_store_ps(&q[(nb*ldq)+32],q3);
//	_mm512_store_ps(&q[(nb*ldq)+48],q4);

	h1 = _mm512_set1_ps(hh[nb-4]);
	q1 = _mm512_load_ps(&q[(nb+1)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+1)*ldq)+16]);
//	q3 = _mm512_load_ps(&q[((nb+1)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+1)*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
//	q3 = _mm512_NFMA_ps(x3, h1, q3);
//	q4 = _mm512_NFMA_ps(x4, h1, q4);

	h2 = _mm512_set1_ps(hh[ldh+nb-3]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
//	q3 = _mm512_NFMA_ps(y3, h2, q3);
//	q4 = _mm512_NFMA_ps(y4, h2, q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-2]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
//	q3 = _mm512_NFMA_ps(z3, h3, q3);
//	q4 = _mm512_NFMA_ps(z4, h3, q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+nb-1]);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
	q2 = _mm512_NFMA_ps(w2, h4, q2);
//	q3 = _mm512_NFMA_ps(w3, h4, q3);
//	q4 = _mm512_NFMA_ps(w4, h4, q4);

	_mm512_store_ps(&q[(nb+1)*ldq],q1);
	_mm512_store_ps(&q[((nb+1)*ldq)+16],q2);
//	_mm512_store_ps(&q[((nb+1)*ldq)+32],q3);
//	_mm512_store_ps(&q[((nb+1)*ldq)+48],q4);

	h1 = _mm512_set1_ps(hh[nb-3]);
	q1 = _mm512_load_ps(&q[(nb+2)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+2)*ldq)+16]);
//	q3 = _mm512_load_ps(&q[((nb+2)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+2)*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
//	q3 = _mm512_NFMA_ps(x3, h1, q3);
//	q4 = _mm512_NFMA_ps(x4, h1, q4);

	h2 = _mm512_set1_ps(hh[ldh+nb-2]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
//	q3 = _mm512_NFMA_ps(y3, h2, q3);
//	q4 = _mm512_NFMA_ps(y4, h2, q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-1]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
//	q3 = _mm512_NFMA_ps(z3, h3, q3);
//	q4 = _mm512_NFMA_ps(z4, h3, q4);

	_mm512_store_ps(&q[(nb+2)*ldq],q1);
	_mm512_store_ps(&q[((nb+2)*ldq)+16],q2);
//	_mm512_store_ps(&q[((nb+2)*ldq)+32],q3);
//	_mm512_store_ps(&q[((nb+2)*ldq)+48],q4);

	h1 = _mm512_set1_ps(hh[nb-2]);
	q1 = _mm512_load_ps(&q[(nb+3)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+3)*ldq)+16]);
//	q3 = _mm512_load_ps(&q[((nb+3)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+3)*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
//	q3 = _mm512_NFMA_ps(x3, h1, q3);
//	q4 = _mm512_NFMA_ps(x4, h1, q4);

	h2 = _mm512_set1_ps(hh[ldh+nb-1]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
//	q3 = _mm512_NFMA_ps(y3, h2, q3);
//	q4 = _mm512_NFMA_ps(y4, h2, q4);

	_mm512_store_ps(&q[(nb+3)*ldq],q1);
	_mm512_store_ps(&q[((nb+3)*ldq)+16],q2);
//	_mm512_store_ps(&q[((nb+3)*ldq)+32],q3);
//	_mm512_store_ps(&q[((nb+3)*ldq)+48],q4);

	h1 = _mm512_set1_ps(hh[nb-1]);
	q1 = _mm512_load_ps(&q[(nb+4)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+4)*ldq)+16]);
//	q3 = _mm512_load_ps(&q[((nb+4)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+4)*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
//	q3 = _mm512_NFMA_ps(x3, h1, q3);
//	q4 = _mm512_NFMA_ps(x4, h1, q4);

	_mm512_store_ps(&q[(nb+4)*ldq],q1);
	_mm512_store_ps(&q[((nb+4)*ldq)+16],q2);
//	_mm512_store_ps(&q[((nb+4)*ldq)+32],q3);
//	_mm512_store_ps(&q[((nb+4)*ldq)+48],q4);

}


/**
 * Unrolled kernel that computes
 * 16 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_16_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [8 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m512 a1_1 = _mm512_load_ps(&q[ldq*5]);
	__m512 a2_1 = _mm512_load_ps(&q[ldq*4]);
	__m512 a3_1 = _mm512_load_ps(&q[ldq*3]);
	__m512 a4_1 = _mm512_load_ps(&q[ldq*2]);
	__m512 a5_1 = _mm512_load_ps(&q[ldq]);
	__m512 a6_1 = _mm512_load_ps(&q[0]);

	__m512 h_6_5 = _mm512_set1_ps(hh[(ldh*5)+1]);
	__m512 h_6_4 = _mm512_set1_ps(hh[(ldh*5)+2]);
	__m512 h_6_3 = _mm512_set1_ps(hh[(ldh*5)+3]);
	__m512 h_6_2 = _mm512_set1_ps(hh[(ldh*5)+4]);
	__m512 h_6_1 = _mm512_set1_ps(hh[(ldh*5)+5]);

//	register __m512d t1 = _mm512_FMA_ps(a5_1, h_6_5, a6_1);
        __m512 t1 = _mm512_FMA_ps(a5_1, h_6_5, a6_1);

	t1 = _mm512_FMA_ps(a4_1, h_6_4, t1);
	t1 = _mm512_FMA_ps(a3_1, h_6_3, t1);
	t1 = _mm512_FMA_ps(a2_1, h_6_2, t1);
	t1 = _mm512_FMA_ps(a1_1, h_6_1, t1);

	__m512 h_5_4 = _mm512_set1_ps(hh[(ldh*4)+1]);
	__m512 h_5_3 = _mm512_set1_ps(hh[(ldh*4)+2]);
	__m512 h_5_2 = _mm512_set1_ps(hh[(ldh*4)+3]);
	__m512 h_5_1 = _mm512_set1_ps(hh[(ldh*4)+4]);

//	register __m512d v1 = _mm512_FMA_ps(a4_1, h_5_4, a5_1);
        __m512 v1 = _mm512_FMA_ps(a4_1, h_5_4, a5_1);

	v1 = _mm512_FMA_ps(a3_1, h_5_3, v1);
	v1 = _mm512_FMA_ps(a2_1, h_5_2, v1);
	v1 = _mm512_FMA_ps(a1_1, h_5_1, v1);

	__m512 h_4_3 = _mm512_set1_ps(hh[(ldh*3)+1]);
	__m512 h_4_2 = _mm512_set1_ps(hh[(ldh*3)+2]);
	__m512 h_4_1 = _mm512_set1_ps(hh[(ldh*3)+3]);

//	register __m512d w1 = _mm512_FMA_ps(a3_1, h_4_3, a4_1);
        __m512 w1 = _mm512_FMA_ps(a3_1, h_4_3, a4_1);

	w1 = _mm512_FMA_ps(a2_1, h_4_2, w1);
	w1 = _mm512_FMA_ps(a1_1, h_4_1, w1);

	__m512 h_2_1 = _mm512_set1_ps(hh[ldh+1]);
	__m512 h_3_2 = _mm512_set1_ps(hh[(ldh*2)+1]);
	__m512 h_3_1 = _mm512_set1_ps(hh[(ldh*2)+2]);

//	register __m512d z1 = _mm512_FMA_ps(a2_1, h_3_2, a3_1);
        __m512 z1 = _mm512_FMA_ps(a2_1, h_3_2, a3_1);

	z1 = _mm512_FMA_ps(a1_1, h_3_1, z1);
//	register __m512d y1 = _mm512_FMA_ps(a1_1, h_2_1, a2_1);
        __m512 y1 = _mm512_FMA_ps(a1_1, h_2_1, a2_1);


//	register __m512d x1 = a1_1;
        __m512 x1 = a1_1;

//	__m512 a1_2 = _mm512_load_ps(&q[(ldq*5)+16]);
//	__m512 a2_2 = _mm512_load_ps(&q[(ldq*4)+16]);
//	__m512 a3_2 = _mm512_load_ps(&q[(ldq*3)+16]);
//	__m512 a4_2 = _mm512_load_ps(&q[(ldq*2)+16]);
//	__m512 a5_2 = _mm512_load_ps(&q[(ldq)+16]);
//	__m512 a6_2 = _mm512_load_ps(&q[0+16]);
//
////	register __m512d t2 = _mm512_FMA_ps(a5_2, h_6_5, a6_2);
//         __m512 t2 = _mm512_FMA_ps(a5_2, h_6_5, a6_2);
//
//	t2 = _mm512_FMA_ps(a4_2, h_6_4, t2);
//	t2 = _mm512_FMA_ps(a3_2, h_6_3, t2);
//	t2 = _mm512_FMA_ps(a2_2, h_6_2, t2);
//	t2 = _mm512_FMA_ps(a1_2, h_6_1, t2);
//
////	register __m512d v2 = _mm512_FMA_ps(a4_2, h_5_4, a5_2);
//        __m512 v2 = _mm512_FMA_ps(a4_2, h_5_4, a5_2);
//
//	v2 = _mm512_FMA_ps(a3_2, h_5_3, v2);
//	v2 = _mm512_FMA_ps(a2_2, h_5_2, v2);
//	v2 = _mm512_FMA_ps(a1_2, h_5_1, v2);
//
////	register __m512d w2 = _mm512_FMA_ps(a3_2, h_4_3, a4_2);
//        __m512 w2 = _mm512_FMA_ps(a3_2, h_4_3, a4_2);
//
//	w2 = _mm512_FMA_ps(a2_2, h_4_2, w2);
//	w2 = _mm512_FMA_ps(a1_2, h_4_1, w2);
//
////	register __m512d z2 = _mm512_FMA_ps(a2_2, h_3_2, a3_2);
//         __m512 z2 = _mm512_FMA_ps(a2_2, h_3_2, a3_2);
//
//	z2 = _mm512_FMA_ps(a1_2, h_3_1, z2);
////	register __m512d y2 = _mm512_FMA_ps(a1_2, h_2_1, a2_2);
//        __m512 y2 = _mm512_FMA_ps(a1_2, h_2_1, a2_2);
//
//
////	register __m512d x2 = a1_2;
//        __m512 x2 = a1_2;

//	__m512 a1_3 = _mm512_load_ps(&q[(ldq*5)+32]);
//	__m512 a2_3 = _mm512_load_ps(&q[(ldq*4)+32]);
//	__m512 a3_3 = _mm512_load_ps(&q[(ldq*3)+32]);
//	__m512 a4_3 = _mm512_load_ps(&q[(ldq*2)+32]);
//	__m512 a5_3 = _mm512_load_ps(&q[(ldq)+32]);
//	__m512 a6_3 = _mm512_load_ps(&q[0+32]);
//
////	register __m512d t3 = _mm512_FMA_ps(a5_3, h_6_5, a6_3);
//        __m512 t3 = _mm512_FMA_ps(a5_3, h_6_5, a6_3);
//
//	t3 = _mm512_FMA_ps(a4_3, h_6_4, t3);
//	t3 = _mm512_FMA_ps(a3_3, h_6_3, t3);
//	t3 = _mm512_FMA_ps(a2_3, h_6_2, t3);
//	t3 = _mm512_FMA_ps(a1_3, h_6_1, t3);
//
////	register __m512d v3 = _mm512_FMA_ps(a4_3, h_5_4, a5_3);
//        __m512 v3 = _mm512_FMA_ps(a4_3, h_5_4, a5_3);
//
//	v3 = _mm512_FMA_ps(a3_3, h_5_3, v3);
//	v3 = _mm512_FMA_ps(a2_3, h_5_2, v3);
//	v3 = _mm512_FMA_ps(a1_3, h_5_1, v3);
//
////	register __m512d w3 = _mm512_FMA_ps(a3_3, h_4_3, a4_3);
//        __m512 w3 = _mm512_FMA_ps(a3_3, h_4_3, a4_3);
//
//	w3 = _mm512_FMA_ps(a2_3, h_4_2, w3);
//	w3 = _mm512_FMA_ps(a1_3, h_4_1, w3);
//
////	register __m512d z3 = _mm512_FMA_ps(a2_3, h_3_2, a3_3);
//        __m512 z3 = _mm512_FMA_ps(a2_3, h_3_2, a3_3);
//
//	z3 = _mm512_FMA_ps(a1_3, h_3_1, z3);
////	register __m512d y3 = _mm512_FMA_ps(a1_3, h_2_1, a2_3);
//        __m512 y3 = _mm512_FMA_ps(a1_3, h_2_1, a2_3);
//
//
////	register __m512d x3 = a1_3;
//        __m512 x3 = a1_3;


//	__m512 a1_4 = _mm512_load_ps(&q[(ldq*5)+48]);
//	__m512 a2_4 = _mm512_load_ps(&q[(ldq*4)+48]);
//	__m512 a3_4 = _mm512_load_ps(&q[(ldq*3)+48]);
//	__m512 a4_4 = _mm512_load_ps(&q[(ldq*2)+48]);
//	__m512 a5_4 = _mm512_load_ps(&q[(ldq)+48]);
//	__m512 a6_4 = _mm512_load_ps(&q[0+48]);
//
////	register __m512d t4 = _mm512_FMA_ps(a5_4, h_6_5, a6_4);
 //       __m512 t4 = _mm512_FMA_ps(a5_4, h_6_5, a6_4);
//
//	t4 = _mm512_FMA_ps(a4_4, h_6_4, t4);
//	t4 = _mm512_FMA_ps(a3_4, h_6_3, t4);
//	t4 = _mm512_FMA_ps(a2_4, h_6_2, t4);
//	t4 = _mm512_FMA_ps(a1_4, h_6_1, t4);
//
////	register __m512d v4 = _mm512_FMA_ps(a4_4, h_5_4, a5_4);
//        __m512 v4 = _mm512_FMA_ps(a4_4, h_5_4, a5_4);
//
//	v4 = _mm512_FMA_ps(a3_4, h_5_3, v4);
//	v4 = _mm512_FMA_ps(a2_4, h_5_2, v4);
//	v4 = _mm512_FMA_ps(a1_4, h_5_1, v4);
//
////	register __m512d w4 = _mm512_FMA_ps(a3_4, h_4_3, a4_4);
  //      __m512 w4 = _mm512_FMA_ps(a3_4, h_4_3, a4_4);
//
//	w4 = _mm512_FMA_ps(a2_4, h_4_2, w4);
//	w4 = _mm512_FMA_ps(a1_4, h_4_1, w4);
//
////	register __m512d z4 = _mm512_FMA_ps(a2_4, h_3_2, a3_4);
//        __m512 z4 = _mm512_FMA_ps(a2_4, h_3_2, a3_4);
//
//	z4 = _mm512_FMA_ps(a1_4, h_3_1, z4);
////	register __m512d y4 = _mm512_FMA_ps(a1_4, h_2_1, a2_4);
//         __m512 y4 = _mm512_FMA_ps(a1_4, h_2_1, a2_4);
//
//
////	register __m512d x4 = a1_4;
//        __m512 x4 = a1_4;


	__m512 q1;
//	__m512 q2;
//	__m512 q3;
//	__m512 q4;

	__m512 h1;
	__m512 h2;
	__m512 h3;
	__m512 h4;
	__m512 h5;
	__m512 h6;

	for(i = 6; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-5]);
		q1 = _mm512_load_ps(&q[i*ldq]);
//		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
//		q3 = _mm512_load_ps(&q[(i*ldq)+32]);
//		q4 = _mm512_load_ps(&q[(i*ldq)+48]);

		x1 = _mm512_FMA_ps(q1, h1, x1);
//		x2 = _mm512_FMA_ps(q2, h1, x2);
//		x3 = _mm512_FMA_ps(q3, h1, x3);
//		x4 = _mm512_FMA_ps(q4, h1, x4);

		h2 = _mm512_set1_ps(hh[ldh+i-4]);

		y1 = _mm512_FMA_ps(q1, h2, y1);
//		y2 = _mm512_FMA_ps(q2, h2, y2);
//		y3 = _mm512_FMA_ps(q3, h2, y3);
//		y4 = _mm512_FMA_ps(q4, h2, y4);

		h3 = _mm512_set1_ps(hh[(ldh*2)+i-3]);

		z1 = _mm512_FMA_ps(q1, h3, z1);
//		z2 = _mm512_FMA_ps(q2, h3, z2);
//		z3 = _mm512_FMA_ps(q3, h3, z3);
//		z4 = _mm512_FMA_ps(q4, h3, z4);

		h4 = _mm512_set1_ps(hh[(ldh*3)+i-2]);

		w1 = _mm512_FMA_ps(q1, h4, w1);
//		w2 = _mm512_FMA_ps(q2, h4, w2);
//		w3 = _mm512_FMA_ps(q3, h4, w3);
//		w4 = _mm512_FMA_ps(q4, h4, w4);

		h5 = _mm512_set1_ps(hh[(ldh*4)+i-1]);

		v1 = _mm512_FMA_ps(q1, h5, v1);
//		v2 = _mm512_FMA_ps(q2, h5, v2);
//		v3 = _mm512_FMA_ps(q3, h5, v3);
//		v4 = _mm512_FMA_ps(q4, h5, v4);

		h6 = _mm512_set1_ps(hh[(ldh*5)+i]);

		t1 = _mm512_FMA_ps(q1, h6, t1);
//		t2 = _mm512_FMA_ps(q2, h6, t2);
//		t3 = _mm512_FMA_ps(q3, h6, t3);
//		t4 = _mm512_FMA_ps(q4, h6, t4);
	}

	h1 = _mm512_set1_ps(hh[nb-5]);
	q1 = _mm512_load_ps(&q[nb*ldq]);
//	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
//	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);
//	q4 = _mm512_load_ps(&q[(nb*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
//	x2 = _mm512_FMA_ps(q2, h1, x2);
//	x3 = _mm512_FMA_ps(q3, h1, x3);
//	x4 = _mm512_FMA_ps(q4, h1, x4);

	h2 = _mm512_set1_ps(hh[ldh+nb-4]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
//	y2 = _mm512_FMA_ps(q2, h2, y2);
//	y3 = _mm512_FMA_ps(q3, h2, y3);
//	y4 = _mm512_FMA_ps(q4, h2, y4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-3]);

	z1 = _mm512_FMA_ps(q1, h3, z1);
//	z2 = _mm512_FMA_ps(q2, h3, z2);
//	z3 = _mm512_FMA_ps(q3, h3, z3);
//	z4 = _mm512_FMA_ps(q4, h3, z4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+nb-2]);

	w1 = _mm512_FMA_ps(q1, h4, w1);
//	w2 = _mm512_FMA_ps(q2, h4, w2);
//	w3 = _mm512_FMA_ps(q3, h4, w3);
//	w4 = _mm512_FMA_ps(q4, h4, w4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+nb-1]);

	v1 = _mm512_FMA_ps(q1, h5, v1);
//	v2 = _mm512_FMA_ps(q2, h5, v2);
//	v3 = _mm512_FMA_ps(q3, h5, v3);
//	v4 = _mm512_FMA_ps(q4, h5, v4);

	h1 = _mm512_set1_ps(hh[nb-4]);

	q1 = _mm512_load_ps(&q[(nb+1)*ldq]);
//	q2 = _mm512_load_ps(&q[((nb+1)*ldq)+16]);
//	q3 = _mm512_load_ps(&q[((nb+1)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+1)*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
//	x2 = _mm512_FMA_ps(q2, h1, x2);
//	x3 = _mm512_FMA_ps(q3, h1, x3);
//	x4 = _mm512_FMA_ps(q4, h1, x4);

	h2 = _mm512_set1_ps(hh[ldh+nb-3]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
//	y2 = _mm512_FMA_ps(q2, h2, y2);
//	y3 = _mm512_FMA_ps(q3, h2, y3);
//	y4 = _mm512_FMA_ps(q4, h2, y4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-2]);

	z1 = _mm512_FMA_ps(q1, h3, z1);
//	z2 = _mm512_FMA_ps(q2, h3, z2);
//	z3 = _mm512_FMA_ps(q3, h3, z3);
//	z4 = _mm512_FMA_ps(q4, h3, z4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+nb-1]);

	w1 = _mm512_FMA_ps(q1, h4, w1);
//	w2 = _mm512_FMA_ps(q2, h4, w2);
//	w3 = _mm512_FMA_ps(q3, h4, w3);
//	w4 = _mm512_FMA_ps(q4, h4, w4);

	h1 = _mm512_set1_ps(hh[nb-3]);
	q1 = _mm512_load_ps(&q[(nb+2)*ldq]);
//	q2 = _mm512_load_ps(&q[((nb+2)*ldq)+16]);
//	q3 = _mm512_load_ps(&q[((nb+2)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+2)*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
//	x2 = _mm512_FMA_ps(q2, h1, x2);
//	x3 = _mm512_FMA_ps(q3, h1, x3);
//	x4 = _mm512_FMA_ps(q4, h1, x4);

	h2 = _mm512_set1_ps(hh[ldh+nb-2]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
//	y2 = _mm512_FMA_ps(q2, h2, y2);
//	y3 = _mm512_FMA_ps(q3, h2, y3);
//	y4 = _mm512_FMA_ps(q4, h2, y4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-1]);

	z1 = _mm512_FMA_ps(q1, h3, z1);
//	z2 = _mm512_FMA_ps(q2, h3, z2);
//	z3 = _mm512_FMA_ps(q3, h3, z3);
//	z4 = _mm512_FMA_ps(q4, h3, z4);

	h1 = _mm512_set1_ps(hh[nb-2]);
	q1 = _mm512_load_ps(&q[(nb+3)*ldq]);
//	q2 = _mm512_load_ps(&q[((nb+3)*ldq)+16]);
//	q3 = _mm512_load_ps(&q[((nb+3)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+3)*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
//	x2 = _mm512_FMA_ps(q2, h1, x2);
//	x3 = _mm512_FMA_ps(q3, h1, x3);
//	x4 = _mm512_FMA_ps(q4, h1, x4);

	h2 = _mm512_set1_ps(hh[ldh+nb-1]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
//	y2 = _mm512_FMA_ps(q2, h2, y2);
//	y3 = _mm512_FMA_ps(q3, h2, y3);
//	y4 = _mm512_FMA_ps(q4, h2, y4);

	h1 = _mm512_set1_ps(hh[nb-1]);
	q1 = _mm512_load_ps(&q[(nb+4)*ldq]);
//	q2 = _mm512_load_ps(&q[((nb+4)*ldq)+16]);
//	q3 = _mm512_load_ps(&q[((nb+4)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+4)*ldq)+48]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
//	x2 = _mm512_FMA_ps(q2, h1, x2);
//	x3 = _mm512_FMA_ps(q3, h1, x3);
//	x4 = _mm512_FMA_ps(q4, h1, x4);

	/////////////////////////////////////////////////////
	// Apply tau, correct wrong calculation using pre-calculated scalar products
	/////////////////////////////////////////////////////

	__m512 tau1 = _mm512_set1_ps(hh[0]);
	x1 = _mm512_mul_ps(x1, tau1);
//	x2 = _mm512_mul_ps(x2, tau1);
//	x3 = _mm512_mul_ps(x3, tau1);
//	x4 = _mm512_mul_ps(x4, tau1);

	__m512 tau2 = _mm512_set1_ps(hh[ldh]);
	__m512 vs_1_2 = _mm512_set1_ps(scalarprods[0]);
	h2 = _mm512_mul_ps(tau2, vs_1_2);

	y1 = _mm512_FMSUB_ps(y1, tau2, _mm512_mul_ps(x1,h2));
//	y2 = _mm512_FMSUB_ps(y2, tau2, _mm512_mul_ps(x2,h2));
//	y3 = _mm512_FMSUB_ps(y3, tau2, _mm512_mul_ps(x3,h2));
//	y4 = _mm512_FMSUB_ps(y4, tau2, _mm512_mul_ps(x4,h2));

	__m512 tau3 = _mm512_set1_ps(hh[ldh*2]);
	__m512 vs_1_3 = _mm512_set1_ps(scalarprods[1]);
	__m512 vs_2_3 = _mm512_set1_ps(scalarprods[2]);

	h2 = _mm512_mul_ps(tau3, vs_1_3);
	h3 = _mm512_mul_ps(tau3, vs_2_3);

	z1 = _mm512_FMSUB_ps(z1, tau3, _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2)));
//	z2 = _mm512_FMSUB_ps(z2, tau3, _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2)));
//	z3 = _mm512_FMSUB_ps(z3, tau3, _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2)));
//	z4 = _mm512_FMSUB_ps(z4, tau3, _mm512_FMA_ps(y4, h3, _mm512_mul_ps(x4,h2)));

	__m512 tau4 = _mm512_set1_ps(hh[ldh*3]);
	__m512 vs_1_4 = _mm512_set1_ps(scalarprods[3]);
	__m512 vs_2_4 = _mm512_set1_ps(scalarprods[4]);

	h2 = _mm512_mul_ps(tau4, vs_1_4);
	h3 = _mm512_mul_ps(tau4, vs_2_4);

	__m512 vs_3_4 = _mm512_set1_ps(scalarprods[5]);
	h4 = _mm512_mul_ps(tau4, vs_3_4);

	w1 = _mm512_FMSUB_ps(w1, tau4, _mm512_FMA_ps(z1, h4, _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2))));
//	w2 = _mm512_FMSUB_ps(w2, tau4, _mm512_FMA_ps(z2, h4, _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2))));
//	w3 = _mm512_FMSUB_ps(w3, tau4, _mm512_FMA_ps(z3, h4, _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2))));
//	w4 = _mm512_FMSUB_ps(w4, tau4, _mm512_FMA_ps(z4, h4, _mm512_FMA_ps(y4, h3, _mm512_mul_ps(x4,h2))));

	__m512 tau5 = _mm512_set1_ps(hh[ldh*4]);
	__m512 vs_1_5 = _mm512_set1_ps(scalarprods[6]);
	__m512 vs_2_5 = _mm512_set1_ps(scalarprods[7]);

	h2 = _mm512_mul_ps(tau5, vs_1_5);
	h3 = _mm512_mul_ps(tau5, vs_2_5);

	__m512 vs_3_5 = _mm512_set1_ps(scalarprods[8]);
	__m512 vs_4_5 = _mm512_set1_ps(scalarprods[9]);

	h4 = _mm512_mul_ps(tau5, vs_3_5);
	h5 = _mm512_mul_ps(tau5, vs_4_5);

	v1 = _mm512_FMSUB_ps(v1, tau5, _mm512_add_ps(_mm512_FMA_ps(w1, h5, _mm512_mul_ps(z1,h4)), _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2))));
//	v2 = _mm512_FMSUB_ps(v2, tau5, _mm512_add_ps(_mm512_FMA_ps(w2, h5, _mm512_mul_ps(z2,h4)), _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2))));
//	v3 = _mm512_FMSUB_ps(v3, tau5, _mm512_add_ps(_mm512_FMA_ps(w3, h5, _mm512_mul_ps(z3,h4)), _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2))));
//	v4 = _mm512_FMSUB_ps(v4, tau5, _mm512_add_ps(_mm512_FMA_ps(w4, h5, _mm512_mul_ps(z4,h4)), _mm512_FMA_ps(y4, h3, _mm512_mul_ps(x4,h2))));

	__m512 tau6 = _mm512_set1_ps(hh[ldh*5]);
	__m512 vs_1_6 = _mm512_set1_ps(scalarprods[10]);
	__m512 vs_2_6 = _mm512_set1_ps(scalarprods[11]);
	h2 = _mm512_mul_ps(tau6, vs_1_6);
	h3 = _mm512_mul_ps(tau6, vs_2_6);

	__m512 vs_3_6 = _mm512_set1_ps(scalarprods[12]);
	__m512 vs_4_6 = _mm512_set1_ps(scalarprods[13]);
	__m512 vs_5_6 = _mm512_set1_ps(scalarprods[14]);

	h4 = _mm512_mul_ps(tau6, vs_3_6);
	h5 = _mm512_mul_ps(tau6, vs_4_6);
	h6 = _mm512_mul_ps(tau6, vs_5_6);

	t1 = _mm512_FMSUB_ps(t1, tau6, _mm512_FMA_ps(v1, h6, _mm512_add_ps(_mm512_FMA_ps(w1, h5, _mm512_mul_ps(z1,h4)), _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2)))));
//	t2 = _mm512_FMSUB_ps(t2, tau6, _mm512_FMA_ps(v2, h6, _mm512_add_ps(_mm512_FMA_ps(w2, h5, _mm512_mul_ps(z2,h4)), _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2)))));
//	t3 = _mm512_FMSUB_ps(t3, tau6, _mm512_FMA_ps(v3, h6, _mm512_add_ps(_mm512_FMA_ps(w3, h5, _mm512_mul_ps(z3,h4)), _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2)))));
//	t4 = _mm512_FMSUB_ps(t4, tau6, _mm512_FMA_ps(v4, h6, _mm512_add_ps(_mm512_FMA_ps(w4, h5, _mm512_mul_ps(z4,h4)), _mm512_FMA_ps(y4, h3, _mm512_mul_ps(x4,h2)))));


	/////////////////////////////////////////////////////
	// Rank-1 update of Q [8 x nb+3]
	/////////////////////////////////////////////////////

	q1 = _mm512_load_ps(&q[0]);
//	q2 = _mm512_load_ps(&q[0+16]);
//	q3 = _mm512_load_ps(&q[0+32]);
//	q4 = _mm512_load_ps(&q[0+48]);

	q1 = _mm512_sub_ps(q1, t1);
//	q2 = _mm512_sub_ps(q2, t2);
//	q3 = _mm512_sub_ps(q3, t3);
//	q4 = _mm512_sub_ps(q4, t4);

	_mm512_store_ps(&q[0],q1);
//	_mm512_store_ps(&q[0+16],q2);
//	_mm512_store_ps(&q[0+32],q3);
//	_mm512_store_ps(&q[0+48],q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+1]);
	q1 = _mm512_load_ps(&q[ldq]);
//	q2 = _mm512_load_ps(&q[ldq+16]);
//	q3 = _mm512_load_ps(&q[ldq+32]);
//	q4 = _mm512_load_ps(&q[ldq+48]);

	q1 = _mm512_sub_ps(q1, v1);
//	q2 = _mm512_sub_ps(q2, v2);
//	q3 = _mm512_sub_ps(q3, v3);
//	q4 = _mm512_sub_ps(q4, v4);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
//	q2 = _mm512_NFMA_ps(t2, h6, q2);
//	q3 = _mm512_NFMA_ps(t3, h6, q3);
//	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq],q1);
//	_mm512_store_ps(&q[ldq+16],q2);
//	_mm512_store_ps(&q[ldq+32],q3);
//	_mm512_store_ps(&q[ldq+48],q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+1]);
	q1 = _mm512_load_ps(&q[ldq*2]);
//	q2 = _mm512_load_ps(&q[(ldq*2)+16]);
//	q3 = _mm512_load_ps(&q[(ldq*2)+32]);
//	q4 = _mm512_load_ps(&q[(ldq*2)+48]);

	q1 = _mm512_sub_ps(q1, w1);
//	q2 = _mm512_sub_ps(q2, w2);
//	q3 = _mm512_sub_ps(q3, w3);
//	q4 = _mm512_sub_ps(q4, w4);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
//	q2 = _mm512_NFMA_ps(v2, h5, q2);
//	q3 = _mm512_NFMA_ps(v3, h5, q3);
//	q4 = _mm512_NFMA_ps(v4, h5, q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+2]);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
//	q2 = _mm512_NFMA_ps(t2, h6, q2);
//	q3 = _mm512_NFMA_ps(t3, h6, q3);
//	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq*2],q1);
//	_mm512_store_ps(&q[(ldq*2)+16],q2);
//	_mm512_store_ps(&q[(ldq*2)+32],q3);
//	_mm512_store_ps(&q[(ldq*2)+48],q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+1]);
	q1 = _mm512_load_ps(&q[ldq*3]);
//	q2 = _mm512_load_ps(&q[(ldq*3)+16]);
//	q3 = _mm512_load_ps(&q[(ldq*3)+32]);
//	q4 = _mm512_load_ps(&q[(ldq*3)+48]);

	q1 = _mm512_sub_ps(q1, z1);
//	q2 = _mm512_sub_ps(q2, z2);
//	q3 = _mm512_sub_ps(q3, z3);
//	q4 = _mm512_sub_ps(q4, z4);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
//	q2 = _mm512_NFMA_ps(w2, h4, q2);
//	q3 = _mm512_NFMA_ps(w3, h4, q3);
//	q4 = _mm512_NFMA_ps(w4, h4, q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+2]);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
//	q2 = _mm512_NFMA_ps(v2, h5, q2);
//	q3 = _mm512_NFMA_ps(v3, h5, q3);
//	q4 = _mm512_NFMA_ps(v4, h5, q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+3]);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
//	q2 = _mm512_NFMA_ps(t2, h6, q2);
//	q3 = _mm512_NFMA_ps(t3, h6, q3);
//	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq*3],q1);
//	_mm512_store_ps(&q[(ldq*3)+16],q2);
//	_mm512_store_ps(&q[(ldq*3)+32],q3);
//	_mm512_store_ps(&q[(ldq*3)+48],q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+1]);
	q1 = _mm512_load_ps(&q[ldq*4]);
//	q2 = _mm512_load_ps(&q[(ldq*4)+16]);
//	q3 = _mm512_load_ps(&q[(ldq*4)+32]);
//	q4 = _mm512_load_ps(&q[(ldq*4)+48]);

	q1 = _mm512_sub_ps(q1, y1);
//	q2 = _mm512_sub_ps(q2, y2);
//	q3 = _mm512_sub_ps(q3, y3);
//	q4 = _mm512_sub_ps(q4, y4);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
//	q2 = _mm512_NFMA_ps(z2, h3, q2);
//	q3 = _mm512_NFMA_ps(z3, h3, q3);
//	q4 = _mm512_NFMA_ps(z4, h3, q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+2]);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
//	q2 = _mm512_NFMA_ps(w2, h4, q2);
//	q3 = _mm512_NFMA_ps(w3, h4, q3);
//	q4 = _mm512_NFMA_ps(w4, h4, q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+3]);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
//	q2 = _mm512_NFMA_ps(v2, h5, q2);
//	q3 = _mm512_NFMA_ps(v3, h5, q3);
//	q4 = _mm512_NFMA_ps(v4, h5, q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+4]);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
//	q2 = _mm512_NFMA_ps(t2, h6, q2);
//	q3 = _mm512_NFMA_ps(t3, h6, q3);
//	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq*4],q1);
//	_mm512_store_ps(&q[(ldq*4)+16],q2);
//	_mm512_store_ps(&q[(ldq*4)+32],q3);
//	_mm512_store_ps(&q[(ldq*4)+48],q4);

	h2 = _mm512_set1_ps(hh[(ldh)+1]);
	q1 = _mm512_load_ps(&q[ldq*5]);
//	q2 = _mm512_load_ps(&q[(ldq*5)+16]);
//	q3 = _mm512_load_ps(&q[(ldq*5)+32]);
//	q4 = _mm512_load_ps(&q[(ldq*5)+48]);

	q1 = _mm512_sub_ps(q1, x1);
//	q2 = _mm512_sub_ps(q2, x2);
//	q3 = _mm512_sub_ps(q3, x3);
//	q4 = _mm512_sub_ps(q4, x4);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
//	q2 = _mm512_NFMA_ps(y2, h2, q2);
//	q3 = _mm512_NFMA_ps(y3, h2, q3);
//	q4 = _mm512_NFMA_ps(y4, h2, q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+2]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
//	q2 = _mm512_NFMA_ps(z2, h3, q2);
//	q3 = _mm512_NFMA_ps(z3, h3, q3);
//	q4 = _mm512_NFMA_ps(z4, h3, q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+3]);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
//	q2 = _mm512_NFMA_ps(w2, h4, q2);
//	q3 = _mm512_NFMA_ps(w3, h4, q3);
//	q4 = _mm512_NFMA_ps(w4, h4, q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+4]);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
//	q2 = _mm512_NFMA_ps(v2, h5, q2);
//	q3 = _mm512_NFMA_ps(v3, h5, q3);
//	q4 = _mm512_NFMA_ps(v4, h5, q4);

	h6 = _mm512_set1_ps(hh[(ldh*5)+5]);

	q1 = _mm512_NFMA_ps(t1, h6, q1);
//	q2 = _mm512_NFMA_ps(t2, h6, q2);
//	q3 = _mm512_NFMA_ps(t3, h6, q3);
//	q4 = _mm512_NFMA_ps(t4, h6, q4);

	_mm512_store_ps(&q[ldq*5],q1);
//	_mm512_store_ps(&q[(ldq*5)+16],q2);
//	_mm512_store_ps(&q[(ldq*5)+32],q3);
//	_mm512_store_ps(&q[(ldq*5)+48],q4);

	for (i = 6; i < nb; i++)
	{
		q1 = _mm512_load_ps(&q[i*ldq]);
//		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
//		q3 = _mm512_load_ps(&q[(i*ldq)+32]);
//		q4 = _mm512_load_ps(&q[(i*ldq)+48]);

		h1 = _mm512_set1_ps(hh[i-5]);

		q1 = _mm512_NFMA_ps(x1, h1, q1);
//		q2 = _mm512_NFMA_ps(x2, h1, q2);
//		q3 = _mm512_NFMA_ps(x3, h1, q3);
//		q4 = _mm512_NFMA_ps(x4, h1, q4);

		h2 = _mm512_set1_ps(hh[ldh+i-4]);

		q1 = _mm512_NFMA_ps(y1, h2, q1);
//		q2 = _mm512_NFMA_ps(y2, h2, q2);
//		q3 = _mm512_NFMA_ps(y3, h2, q3);
  //      	q4 = _mm512_NFMA_ps(y4, h2, q4);

		h3 = _mm512_set1_ps(hh[(ldh*2)+i-3]);

		q1 = _mm512_NFMA_ps(z1, h3, q1);
//		q2 = _mm512_NFMA_ps(z2, h3, q2);
//		q3 = _mm512_NFMA_ps(z3, h3, q3);
//		q4 = _mm512_NFMA_ps(z4, h3, q4);

		h4 = _mm512_set1_ps(hh[(ldh*3)+i-2]);

		q1 = _mm512_NFMA_ps(w1, h4, q1);
//		q2 = _mm512_NFMA_ps(w2, h4, q2);
//		q3 = _mm512_NFMA_ps(w3, h4, q3);
//		q4 = _mm512_NFMA_ps(w4, h4, q4);

		h5 = _mm512_set1_ps(hh[(ldh*4)+i-1]);

		q1 = _mm512_NFMA_ps(v1, h5, q1);
//		q2 = _mm512_NFMA_ps(v2, h5, q2);
//		q3 = _mm512_NFMA_ps(v3, h5, q3);
//		q4 = _mm512_NFMA_ps(v4, h5, q4);

		h6 = _mm512_set1_ps(hh[(ldh*5)+i]);

		q1 = _mm512_NFMA_ps(t1, h6, q1);
//		q2 = _mm512_NFMA_ps(t2, h6, q2);
//		q3 = _mm512_NFMA_ps(t3, h6, q3);
//		q4 = _mm512_NFMA_ps(t4, h6, q4);

		_mm512_store_ps(&q[i*ldq],q1);
//		_mm512_store_ps(&q[(i*ldq)+16],q2);
//		_mm512_store_ps(&q[(i*ldq)+32],q3);
//		_mm512_store_ps(&q[(i*ldq)+48],q4);

	}

	h1 = _mm512_set1_ps(hh[nb-5]);
	q1 = _mm512_load_ps(&q[nb*ldq]);
//	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
//	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);
//	q4 = _mm512_load_ps(&q[(nb*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
//	q2 = _mm512_NFMA_ps(x2, h1, q2);
//	q3 = _mm512_NFMA_ps(x3, h1, q3);
//	q4 = _mm512_NFMA_ps(x4, h1, q4);

	h2 = _mm512_set1_ps(hh[ldh+nb-4]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
//	q2 = _mm512_NFMA_ps(y2, h2, q2);
//	q3 = _mm512_NFMA_ps(y3, h2, q3);
//	q4 = _mm512_NFMA_ps(y4, h2, q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-3]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
//	q2 = _mm512_NFMA_ps(z2, h3, q2);
//	q3 = _mm512_NFMA_ps(z3, h3, q3);
//	q4 = _mm512_NFMA_ps(z4, h3, q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+nb-2]);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
//	q2 = _mm512_NFMA_ps(w2, h4, q2);
//	q3 = _mm512_NFMA_ps(w3, h4, q3);
//	q4 = _mm512_NFMA_ps(w4, h4, q4);

	h5 = _mm512_set1_ps(hh[(ldh*4)+nb-1]);

	q1 = _mm512_NFMA_ps(v1, h5, q1);
//	q2 = _mm512_NFMA_ps(v2, h5, q2);
//	q3 = _mm512_NFMA_ps(v3, h5, q3);
//	q4 = _mm512_NFMA_ps(v4, h5, q4);

	_mm512_store_ps(&q[nb*ldq],q1);
//	_mm512_store_ps(&q[(nb*ldq)+16],q2);
//	_mm512_store_ps(&q[(nb*ldq)+32],q3);
//	_mm512_store_ps(&q[(nb*ldq)+48],q4);

	h1 = _mm512_set1_ps(hh[nb-4]);
	q1 = _mm512_load_ps(&q[(nb+1)*ldq]);
//	q2 = _mm512_load_ps(&q[((nb+1)*ldq)+16]);
//	q3 = _mm512_load_ps(&q[((nb+1)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+1)*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
//	q2 = _mm512_NFMA_ps(x2, h1, q2);
//	q3 = _mm512_NFMA_ps(x3, h1, q3);
//	q4 = _mm512_NFMA_ps(x4, h1, q4);

	h2 = _mm512_set1_ps(hh[ldh+nb-3]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
//	q2 = _mm512_NFMA_ps(y2, h2, q2);
//	q3 = _mm512_NFMA_ps(y3, h2, q3);
//	q4 = _mm512_NFMA_ps(y4, h2, q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-2]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
//	q2 = _mm512_NFMA_ps(z2, h3, q2);
//	q3 = _mm512_NFMA_ps(z3, h3, q3);
//	q4 = _mm512_NFMA_ps(z4, h3, q4);

	h4 = _mm512_set1_ps(hh[(ldh*3)+nb-1]);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
//	q2 = _mm512_NFMA_ps(w2, h4, q2);
//	q3 = _mm512_NFMA_ps(w3, h4, q3);
//	q4 = _mm512_NFMA_ps(w4, h4, q4);

	_mm512_store_ps(&q[(nb+1)*ldq],q1);
//	_mm512_store_ps(&q[((nb+1)*ldq)+16],q2);
//	_mm512_store_ps(&q[((nb+1)*ldq)+32],q3);
//	_mm512_store_ps(&q[((nb+1)*ldq)+48],q4);

	h1 = _mm512_set1_ps(hh[nb-3]);
	q1 = _mm512_load_ps(&q[(nb+2)*ldq]);
//	q2 = _mm512_load_ps(&q[((nb+2)*ldq)+16]);
//	q3 = _mm512_load_ps(&q[((nb+2)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+2)*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
//	q2 = _mm512_NFMA_ps(x2, h1, q2);
//	q3 = _mm512_NFMA_ps(x3, h1, q3);
//	q4 = _mm512_NFMA_ps(x4, h1, q4);

	h2 = _mm512_set1_ps(hh[ldh+nb-2]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
//	q2 = _mm512_NFMA_ps(y2, h2, q2);
//	q3 = _mm512_NFMA_ps(y3, h2, q3);
//	q4 = _mm512_NFMA_ps(y4, h2, q4);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-1]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
//	q2 = _mm512_NFMA_ps(z2, h3, q2);
//	q3 = _mm512_NFMA_ps(z3, h3, q3);
//	q4 = _mm512_NFMA_ps(z4, h3, q4);

	_mm512_store_ps(&q[(nb+2)*ldq],q1);
//	_mm512_store_ps(&q[((nb+2)*ldq)+16],q2);
//	_mm512_store_ps(&q[((nb+2)*ldq)+32],q3);
//	_mm512_store_ps(&q[((nb+2)*ldq)+48],q4);

	h1 = _mm512_set1_ps(hh[nb-2]);
	q1 = _mm512_load_ps(&q[(nb+3)*ldq]);
//	q2 = _mm512_load_ps(&q[((nb+3)*ldq)+16]);
//	q3 = _mm512_load_ps(&q[((nb+3)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+3)*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
//	q2 = _mm512_NFMA_ps(x2, h1, q2);
//	q3 = _mm512_NFMA_ps(x3, h1, q3);
//	q4 = _mm512_NFMA_ps(x4, h1, q4);

	h2 = _mm512_set1_ps(hh[ldh+nb-1]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
//	q2 = _mm512_NFMA_ps(y2, h2, q2);
//	q3 = _mm512_NFMA_ps(y3, h2, q3);
//	q4 = _mm512_NFMA_ps(y4, h2, q4);

	_mm512_store_ps(&q[(nb+3)*ldq],q1);
//	_mm512_store_ps(&q[((nb+3)*ldq)+16],q2);
//	_mm512_store_ps(&q[((nb+3)*ldq)+32],q3);
//	_mm512_store_ps(&q[((nb+3)*ldq)+48],q4);

	h1 = _mm512_set1_ps(hh[nb-1]);
	q1 = _mm512_load_ps(&q[(nb+4)*ldq]);
//	q2 = _mm512_load_ps(&q[((nb+4)*ldq)+16]);
//	q3 = _mm512_load_ps(&q[((nb+4)*ldq)+32]);
//	q4 = _mm512_load_ps(&q[((nb+4)*ldq)+48]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
//	q2 = _mm512_NFMA_ps(x2, h1, q2);
//	q3 = _mm512_NFMA_ps(x3, h1, q3);
//	q4 = _mm512_NFMA_ps(x4, h1, q4);

	_mm512_store_ps(&q[(nb+4)*ldq],q1);
//	_mm512_store_ps(&q[((nb+4)*ldq)+16],q2);
//	_mm512_store_ps(&q[((nb+4)*ldq)+32],q3);
//	_mm512_store_ps(&q[((nb+4)*ldq)+48],q4);

}

