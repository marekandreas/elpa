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


#define __forceinline __attribute__((always_inline)) static

#ifdef HAVE_AVX512
#define __ELPA_USE_FMA__
#define _mm512_FMA_pd(a,b,c) _mm512_fmadd_pd(a,b,c)
#define _mm512_NFMA_pd(a,b,c) _mm512_fnmadd_pd(a,b,c)
#define _mm512_FMSUB_pd(a,b,c) _mm512_fmsub_pd(a,b,c)
#endif

//Forward declaration
__forceinline void hh_trafo_kernel_8_AVX512_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4);
__forceinline void hh_trafo_kernel_16_AVX512_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4);
__forceinline void hh_trafo_kernel_24_AVX512_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4);
__forceinline void hh_trafo_kernel_32_AVX512_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4);


void quad_hh_trafo_real_avx512_4hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
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


void quad_hh_trafo_real_avx512_4hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;

	// calculating scalar products to compute
	// 4 householder vectors simultaneously
	double s_1_2 = hh[(ldh)+1];
	double s_1_3 = hh[(ldh*2)+2];
	double s_2_3 = hh[(ldh*2)+1];
	double s_1_4 = hh[(ldh*3)+3];
	double s_2_4 = hh[(ldh*3)+2];
	double s_3_4 = hh[(ldh*3)+1];

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

	for (i = 0; i < nq-24; i+=32)
	{
		hh_trafo_kernel_32_AVX512_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
	}
	if (nq == i)
	{
		return;
	}
	if (nq-i == 24)
	{
		hh_trafo_kernel_24_AVX512_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
	}
	if (nq-i == 16)
	{
		hh_trafo_kernel_16_AVX512_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
	}
        else
	{
		hh_trafo_kernel_8_AVX512_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
	}

//	else
//
//	{
//
//		if (nq-i > 4)
//		{
//			hh_trafo_kernel_8_AVX512_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
//		}
//		else
//		{
//			hh_trafo_kernel_4_AVX512_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
//		}
//	}
}

/**
 * Unrolled kernel that computes
 * 32 rows of Q simultaneously, a
 * matrix Vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_32_AVX512_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m512d a1_1 = _mm512_load_pd(&q[ldq*3]);
	__m512d a2_1 = _mm512_load_pd(&q[ldq*2]);
	__m512d a3_1 = _mm512_load_pd(&q[ldq]);
	__m512d a4_1 = _mm512_load_pd(&q[0]);

	__m512d a1_2 = _mm512_load_pd(&q[(ldq*3)+8]);
	__m512d a2_2 = _mm512_load_pd(&q[(ldq*2)+8]);
	__m512d a3_2 = _mm512_load_pd(&q[ldq+8]);
	__m512d a4_2 = _mm512_load_pd(&q[0+8]);

	__m512d a1_3 = _mm512_load_pd(&q[(ldq*3)+16]);
	__m512d a2_3 = _mm512_load_pd(&q[(ldq*2)+16]);
	__m512d a3_3 = _mm512_load_pd(&q[ldq+16]);
	__m512d a4_3 = _mm512_load_pd(&q[0+16]);

	__m512d a1_4 = _mm512_load_pd(&q[(ldq*3)+24]);
	__m512d a2_4 = _mm512_load_pd(&q[(ldq*2)+24]);
	__m512d a3_4 = _mm512_load_pd(&q[ldq+24]);
	__m512d a4_4 = _mm512_load_pd(&q[0+24]);


	__m512d h_2_1 = _mm512_set1_pd(hh[ldh+1]);
	__m512d h_3_2 = _mm512_set1_pd(hh[(ldh*2)+1]);
	__m512d h_3_1 = _mm512_set1_pd(hh[(ldh*2)+2]);
	__m512d h_4_3 = _mm512_set1_pd(hh[(ldh*3)+1]);
	__m512d h_4_2 = _mm512_set1_pd(hh[(ldh*3)+2]);
	__m512d h_4_1 = _mm512_set1_pd(hh[(ldh*3)+3]);

	__m512d w1 = _mm512_FMA_pd(a3_1, h_4_3, a4_1);
	w1 = _mm512_FMA_pd(a2_1, h_4_2, w1);
	w1 = _mm512_FMA_pd(a1_1, h_4_1, w1);
	__m512d z1 = _mm512_FMA_pd(a2_1, h_3_2, a3_1);
	z1 = _mm512_FMA_pd(a1_1, h_3_1, z1);
	__m512d y1 = _mm512_FMA_pd(a1_1, h_2_1, a2_1);
	__m512d x1 = a1_1;

	__m512d w2 = _mm512_FMA_pd(a3_2, h_4_3, a4_2);
	w2 = _mm512_FMA_pd(a2_2, h_4_2, w2);
	w2 = _mm512_FMA_pd(a1_2, h_4_1, w2);
	__m512d z2 = _mm512_FMA_pd(a2_2, h_3_2, a3_2);
	z2 = _mm512_FMA_pd(a1_2, h_3_1, z2);
	__m512d y2 = _mm512_FMA_pd(a1_2, h_2_1, a2_2);
	__m512d x2 = a1_2;

	__m512d w3 = _mm512_FMA_pd(a3_3, h_4_3, a4_3);
	w3 = _mm512_FMA_pd(a2_3, h_4_2, w3);
	w3 = _mm512_FMA_pd(a1_3, h_4_1, w3);
	__m512d z3 = _mm512_FMA_pd(a2_3, h_3_2, a3_3);
	z3 = _mm512_FMA_pd(a1_3, h_3_1, z3);
	__m512d y3 = _mm512_FMA_pd(a1_3, h_2_1, a2_3);
	__m512d x3 = a1_3;

	__m512d w4 = _mm512_FMA_pd(a3_4, h_4_3, a4_4);
	w4 = _mm512_FMA_pd(a2_4, h_4_2, w4);
	w4 = _mm512_FMA_pd(a1_4, h_4_1, w4);
	__m512d z4 = _mm512_FMA_pd(a2_4, h_3_2, a3_4);
	z4 = _mm512_FMA_pd(a1_4, h_3_1, z4);
	__m512d y4 = _mm512_FMA_pd(a1_4, h_2_1, a2_4);
	__m512d x4 = a1_4;


	__m512d q1;
	__m512d q2;
	__m512d q3;
	__m512d q4;

	__m512d h1;
	__m512d h2;
	__m512d h3;
	__m512d h4;

	for(i = 4; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-3]);
		h2 = _mm512_set1_pd(hh[ldh+i-2]);
		h3 = _mm512_set1_pd(hh[(ldh*2)+i-1]);
		h4 = _mm512_set1_pd(hh[(ldh*3)+i]);

		q1 = _mm512_load_pd(&q[i*ldq]);
		q2 = _mm512_load_pd(&q[(i*ldq)+8]);
		q3 = _mm512_load_pd(&q[(i*ldq)+16]);
		q4 = _mm512_load_pd(&q[(i*ldq)+24]);

		x1 = _mm512_FMA_pd(q1, h1, x1);
		y1 = _mm512_FMA_pd(q1, h2, y1);
		z1 = _mm512_FMA_pd(q1, h3, z1);
		w1 = _mm512_FMA_pd(q1, h4, w1);

		x2 = _mm512_FMA_pd(q2, h1, x2);
		y2 = _mm512_FMA_pd(q2, h2, y2);
		z2 = _mm512_FMA_pd(q2, h3, z2);
		w2 = _mm512_FMA_pd(q2, h4, w2);

		x3 = _mm512_FMA_pd(q3, h1, x3);
		y3 = _mm512_FMA_pd(q3, h2, y3);
		z3 = _mm512_FMA_pd(q3, h3, z3);
		w3 = _mm512_FMA_pd(q3, h4, w3);

		x4 = _mm512_FMA_pd(q4, h1, x4);
		y4 = _mm512_FMA_pd(q4, h2, y4);
		z4 = _mm512_FMA_pd(q4, h3, z4);
		w4 = _mm512_FMA_pd(q4, h4, w4);

	}

	h1 = _mm512_set1_pd(hh[nb-3]);
	h2 = _mm512_set1_pd(hh[ldh+nb-2]);
	h3 = _mm512_set1_pd(hh[(ldh*2)+nb-1]);

	q1 = _mm512_load_pd(&q[nb*ldq]);
	q2 = _mm512_load_pd(&q[(nb*ldq)+8]);
	q3 = _mm512_load_pd(&q[(nb*ldq)+16]);
	q4 = _mm512_load_pd(&q[(nb*ldq)+24]);

	x1 = _mm512_FMA_pd(q1, h1, x1);
	y1 = _mm512_FMA_pd(q1, h2, y1);
	z1 = _mm512_FMA_pd(q1, h3, z1);

	x2 = _mm512_FMA_pd(q2, h1, x2);
	y2 = _mm512_FMA_pd(q2, h2, y2);
	z2 = _mm512_FMA_pd(q2, h3, z2);

	x3 = _mm512_FMA_pd(q3, h1, x3);
	y3 = _mm512_FMA_pd(q3, h2, y3);
	z3 = _mm512_FMA_pd(q3, h3, z3);

	x4 = _mm512_FMA_pd(q4, h1, x4);
	y4 = _mm512_FMA_pd(q4, h2, y4);
	z4 = _mm512_FMA_pd(q4, h3, z4);


	h1 = _mm512_set1_pd(hh[nb-2]);
	h2 = _mm512_set1_pd(hh[(ldh*1)+nb-1]);

	q1 = _mm512_load_pd(&q[(nb+1)*ldq]);
	q2 = _mm512_load_pd(&q[((nb+1)*ldq)+8]);
	q3 = _mm512_load_pd(&q[((nb+1)*ldq)+16]);
	q4 = _mm512_load_pd(&q[((nb+1)*ldq)+24]);

	x1 = _mm512_FMA_pd(q1, h1, x1);
	y1 = _mm512_FMA_pd(q1, h2, y1);
	x2 = _mm512_FMA_pd(q2, h1, x2);
	y2 = _mm512_FMA_pd(q2, h2, y2);
	x3 = _mm512_FMA_pd(q3, h1, x3);
	y3 = _mm512_FMA_pd(q3, h2, y3);
	x4 = _mm512_FMA_pd(q4, h1, x4);
	y4 = _mm512_FMA_pd(q4, h2, y4);

	h1 = _mm512_set1_pd(hh[nb-1]);

	q1 = _mm512_load_pd(&q[(nb+2)*ldq]);
	q2 = _mm512_load_pd(&q[((nb+2)*ldq)+8]);
	q3 = _mm512_load_pd(&q[((nb+2)*ldq)+16]);
	q4 = _mm512_load_pd(&q[((nb+2)*ldq)+24]);

	x1 = _mm512_FMA_pd(q1, h1, x1);
	x2 = _mm512_FMA_pd(q2, h1, x2);
	x3 = _mm512_FMA_pd(q3, h1, x3);
	x4 = _mm512_FMA_pd(q4, h1, x4);

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [8 x nb+3]
	/////////////////////////////////////////////////////

	__m512d tau1 = _mm512_set1_pd(hh[0]);
	__m512d tau2 = _mm512_set1_pd(hh[ldh]);
	__m512d tau3 = _mm512_set1_pd(hh[ldh*2]);
	__m512d tau4 = _mm512_set1_pd(hh[ldh*3]);

	__m512d vs_1_2 = _mm512_set1_pd(s_1_2);
	__m512d vs_1_3 = _mm512_set1_pd(s_1_3);
	__m512d vs_2_3 = _mm512_set1_pd(s_2_3);
	__m512d vs_1_4 = _mm512_set1_pd(s_1_4);
	__m512d vs_2_4 = _mm512_set1_pd(s_2_4);
	__m512d vs_3_4 = _mm512_set1_pd(s_3_4);

	h1 = tau1;
	x1 = _mm512_mul_pd(x1, h1);
	x2 = _mm512_mul_pd(x2, h1);
	x3 = _mm512_mul_pd(x3, h1);
	x4 = _mm512_mul_pd(x4, h1);

	h1 = tau2;
	h2 = _mm512_mul_pd(h1, vs_1_2);

	y1 = _mm512_FMSUB_pd(y1, h1, _mm512_mul_pd(x1,h2));
	y2 = _mm512_FMSUB_pd(y2, h1, _mm512_mul_pd(x2,h2));
	y3 = _mm512_FMSUB_pd(y3, h1, _mm512_mul_pd(x3,h2));
	y4 = _mm512_FMSUB_pd(y4, h1, _mm512_mul_pd(x4,h2));

	h1 = tau3;
	h2 = _mm512_mul_pd(h1, vs_1_3);
	h3 = _mm512_mul_pd(h1, vs_2_3);

	z1 = _mm512_FMSUB_pd(z1, h1, _mm512_FMA_pd(y1, h3, _mm512_mul_pd(x1,h2)));
	z2 = _mm512_FMSUB_pd(z2, h1, _mm512_FMA_pd(y2, h3, _mm512_mul_pd(x2,h2)));
	z3 = _mm512_FMSUB_pd(z3, h1, _mm512_FMA_pd(y3, h3, _mm512_mul_pd(x3,h2)));
	z4 = _mm512_FMSUB_pd(z4, h1, _mm512_FMA_pd(y4, h3, _mm512_mul_pd(x4,h2)));

	h1 = tau4;
	h2 = _mm512_mul_pd(h1, vs_1_4);
	h3 = _mm512_mul_pd(h1, vs_2_4);
	h4 = _mm512_mul_pd(h1, vs_3_4);

	w1 = _mm512_FMSUB_pd(w1, h1, _mm512_FMA_pd(z1, h4, _mm512_FMA_pd(y1, h3, _mm512_mul_pd(x1,h2))));
	w2 = _mm512_FMSUB_pd(w2, h1, _mm512_FMA_pd(z2, h4, _mm512_FMA_pd(y2, h3, _mm512_mul_pd(x2,h2))));
	w3 = _mm512_FMSUB_pd(w3, h1, _mm512_FMA_pd(z3, h4, _mm512_FMA_pd(y3, h3, _mm512_mul_pd(x3,h2))));
	w4 = _mm512_FMSUB_pd(w4, h1, _mm512_FMA_pd(z4, h4, _mm512_FMA_pd(y4, h3, _mm512_mul_pd(x4,h2))));

	q1 = _mm512_load_pd(&q[0]);
	q1 = _mm512_sub_pd(q1, w1);
	_mm512_store_pd(&q[0],q1);

	q2 = _mm512_load_pd(&q[0+8]);
	q2 = _mm512_sub_pd(q2, w2);
	_mm512_store_pd(&q[0+8],q2);

	q3 = _mm512_load_pd(&q[0+16]);
	q3 = _mm512_sub_pd(q3, w3);
	_mm512_store_pd(&q[0+16],q3);

	q4 = _mm512_load_pd(&q[0+24]);
	q4 = _mm512_sub_pd(q4, w4);
	_mm512_store_pd(&q[0+24],q4);


	h4 = _mm512_set1_pd(hh[(ldh*3)+1]);
	q1 = _mm512_load_pd(&q[ldq]);
	q2 = _mm512_load_pd(&q[ldq+8]);
	q3 = _mm512_load_pd(&q[ldq+16]);
	q4 = _mm512_load_pd(&q[ldq+24]);

	q1 = _mm512_sub_pd(q1, _mm512_FMA_pd(w1, h4, z1));
	_mm512_store_pd(&q[ldq],q1);
	q2 = _mm512_sub_pd(q2, _mm512_FMA_pd(w2, h4, z2));
	_mm512_store_pd(&q[ldq+8],q2);
	q3 = _mm512_sub_pd(q3, _mm512_FMA_pd(w3, h4, z3));
	_mm512_store_pd(&q[ldq+16],q3);
	q4 = _mm512_sub_pd(q4, _mm512_FMA_pd(w4, h4, z4));
	_mm512_store_pd(&q[ldq+24],q4);


	h3 = _mm512_set1_pd(hh[(ldh*2)+1]);
	h4 = _mm512_set1_pd(hh[(ldh*3)+2]);
	q1 = _mm512_load_pd(&q[ldq*2]);
	q2 = _mm512_load_pd(&q[(ldq*2)+8]);
	q3 = _mm512_load_pd(&q[(ldq*2)+16]);
	q4 = _mm512_load_pd(&q[(ldq*2)+24]);

        q1 = _mm512_sub_pd(q1, y1);
        q1 = _mm512_NFMA_pd(z1, h3, q1);
        q1 = _mm512_NFMA_pd(w1, h4, q1);
	_mm512_store_pd(&q[ldq*2],q1);

        q2 = _mm512_sub_pd(q2, y2);
        q2 = _mm512_NFMA_pd(z2, h3, q2);
        q2 = _mm512_NFMA_pd(w2, h4, q2);
	_mm512_store_pd(&q[(ldq*2)+8],q2);

        q3 = _mm512_sub_pd(q3, y3);
        q3 = _mm512_NFMA_pd(z3, h3, q3);
        q3 = _mm512_NFMA_pd(w3, h4, q3);
	_mm512_store_pd(&q[(ldq*2)+16],q3);

        q4 = _mm512_sub_pd(q4, y4);
        q4 = _mm512_NFMA_pd(z4, h3, q4);
        q4 = _mm512_NFMA_pd(w4, h4, q4);
	_mm512_store_pd(&q[(ldq*2)+24],q4);


	h2 = _mm512_set1_pd(hh[ldh+1]);
	h3 = _mm512_set1_pd(hh[(ldh*2)+2]);
	h4 = _mm512_set1_pd(hh[(ldh*3)+3]);

	q1 = _mm512_load_pd(&q[ldq*3]);

	q1 = _mm512_sub_pd(q1, x1);
        q1 = _mm512_NFMA_pd(y1, h2, q1);
        q1 = _mm512_NFMA_pd(z1, h3, q1);
        q1 = _mm512_NFMA_pd(w1, h4, q1);
	_mm512_store_pd(&q[ldq*3], q1);

	q2 = _mm512_load_pd(&q[(ldq*3)+8]);

	q2 = _mm512_sub_pd(q2, x2);
        q2 = _mm512_NFMA_pd(y2, h2, q2);
        q2 = _mm512_NFMA_pd(z2, h3, q2);
        q2 = _mm512_NFMA_pd(w2, h4, q2);
	_mm512_store_pd(&q[(ldq*3)+8], q2);

	q3 = _mm512_load_pd(&q[(ldq*3)+16]);

	q3 = _mm512_sub_pd(q3, x3);
        q3 = _mm512_NFMA_pd(y3, h2, q3);
        q3 = _mm512_NFMA_pd(z3, h3, q3);
        q3 = _mm512_NFMA_pd(w3, h4, q3);
	_mm512_store_pd(&q[(ldq*3)+16], q3);

	q4 = _mm512_load_pd(&q[(ldq*3)+24]);

	q4 = _mm512_sub_pd(q4, x4);
        q4 = _mm512_NFMA_pd(y4, h2, q4);
        q4 = _mm512_NFMA_pd(z4, h3, q4);
        q4 = _mm512_NFMA_pd(w4, h4, q4);
	_mm512_store_pd(&q[(ldq*3)+24], q4);

	for (i = 4; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-3]);
		h2 = _mm512_set1_pd(hh[ldh+i-2]);
		h3 = _mm512_set1_pd(hh[(ldh*2)+i-1]);
		h4 = _mm512_set1_pd(hh[(ldh*3)+i]);

		q1 = _mm512_load_pd(&q[i*ldq]);
                q1 = _mm512_NFMA_pd(x1, h1, q1);
                q1 = _mm512_NFMA_pd(y1, h2, q1);
                q1 = _mm512_NFMA_pd(z1, h3, q1);
                q1 = _mm512_NFMA_pd(w1, h4, q1);
		_mm512_store_pd(&q[i*ldq],q1);

		q2 = _mm512_load_pd(&q[(i*ldq)+8]);
                q2 = _mm512_NFMA_pd(x2, h1, q2);
                q2 = _mm512_NFMA_pd(y2, h2, q2);
                q2 = _mm512_NFMA_pd(z2, h3, q2);
                q2 = _mm512_NFMA_pd(w2, h4, q2);
		_mm512_store_pd(&q[(i*ldq)+8],q2);

		q3 = _mm512_load_pd(&q[(i*ldq)+16]);
                q3 = _mm512_NFMA_pd(x3, h1, q3);
                q3 = _mm512_NFMA_pd(y3, h2, q3);
                q3 = _mm512_NFMA_pd(z3, h3, q3);
                q3 = _mm512_NFMA_pd(w3, h4, q3);
		_mm512_store_pd(&q[(i*ldq)+16],q3);

		q4 = _mm512_load_pd(&q[(i*ldq)+24]);
                q4 = _mm512_NFMA_pd(x4, h1, q4);
                q4 = _mm512_NFMA_pd(y4, h2, q4);
                q4 = _mm512_NFMA_pd(z4, h3, q4);
                q4 = _mm512_NFMA_pd(w4, h4, q4);
		_mm512_store_pd(&q[(i*ldq)+24],q4);

	}

	h1 = _mm512_set1_pd(hh[nb-3]);
	h2 = _mm512_set1_pd(hh[ldh+nb-2]);
	h3 = _mm512_set1_pd(hh[(ldh*2)+nb-1]);
	q1 = _mm512_load_pd(&q[nb*ldq]);

	q1 = _mm512_NFMA_pd(x1, h1, q1);
        q1 = _mm512_NFMA_pd(y1, h2, q1);
        q1 = _mm512_NFMA_pd(z1, h3, q1);
	_mm512_store_pd(&q[nb*ldq],q1);

	q2 = _mm512_load_pd(&q[(nb*ldq)+8]);

	q2 = _mm512_NFMA_pd(x2, h1, q2);
        q2 = _mm512_NFMA_pd(y2, h2, q2);
        q2 = _mm512_NFMA_pd(z2, h3, q2);
	_mm512_store_pd(&q[(nb*ldq)+8],q2);

	q3 = _mm512_load_pd(&q[(nb*ldq)+16]);

	q3 = _mm512_NFMA_pd(x3, h1, q3);
        q3 = _mm512_NFMA_pd(y3, h2, q3);
        q3 = _mm512_NFMA_pd(z3, h3, q3);
	_mm512_store_pd(&q[(nb*ldq)+16],q3);

	q4 = _mm512_load_pd(&q[(nb*ldq)+24]);

	q4 = _mm512_NFMA_pd(x4, h1, q4);
        q4 = _mm512_NFMA_pd(y4, h2, q4);
        q4 = _mm512_NFMA_pd(z4, h3, q4);
	_mm512_store_pd(&q[(nb*ldq)+24],q4);

	h1 = _mm512_set1_pd(hh[nb-2]);
	h2 = _mm512_set1_pd(hh[ldh+nb-1]);
	q1 = _mm512_load_pd(&q[(nb+1)*ldq]);

	q1 = _mm512_NFMA_pd(x1, h1, q1);
	q1 = _mm512_NFMA_pd(y1, h2, q1);
	_mm512_store_pd(&q[(nb+1)*ldq],q1);

	q2 = _mm512_load_pd(&q[((nb+1)*ldq)+8]);

	q2 = _mm512_NFMA_pd(x2, h1, q2);
	q2 = _mm512_NFMA_pd(y2, h2, q2);
	_mm512_store_pd(&q[((nb+1)*ldq)+8],q2);

	q3 = _mm512_load_pd(&q[((nb+1)*ldq)+16]);

	q3 = _mm512_NFMA_pd(x3, h1, q3);
	q3 = _mm512_NFMA_pd(y3, h2, q3);
	_mm512_store_pd(&q[((nb+1)*ldq)+16],q3);

	q4 = _mm512_load_pd(&q[((nb+1)*ldq)+24]);

	q4 = _mm512_NFMA_pd(x4, h1, q4);
	q4 = _mm512_NFMA_pd(y4, h2, q4);
	_mm512_store_pd(&q[((nb+1)*ldq)+24],q4);


	h1 = _mm512_set1_pd(hh[nb-1]);
	q1 = _mm512_load_pd(&q[(nb+2)*ldq]);

	q1 = _mm512_NFMA_pd(x1, h1, q1);
	_mm512_store_pd(&q[(nb+2)*ldq],q1);

	q2 = _mm512_load_pd(&q[((nb+2)*ldq)+8]);

	q2 = _mm512_NFMA_pd(x2, h1, q2);
	_mm512_store_pd(&q[((nb+2)*ldq)+8],q2);

	q3 = _mm512_load_pd(&q[((nb+2)*ldq)+16]);

	q3 = _mm512_NFMA_pd(x3, h1, q3);
	_mm512_store_pd(&q[((nb+2)*ldq)+16],q3);

	q4 = _mm512_load_pd(&q[((nb+2)*ldq)+24]);

	q4 = _mm512_NFMA_pd(x4, h1, q4);
	_mm512_store_pd(&q[((nb+2)*ldq)+24],q4);


}


/**
 * Unrolled kernel that computes
 * 24 rows of Q simultaneously, a
 * matrix Vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_24_AVX512_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m512d a1_1 = _mm512_load_pd(&q[ldq*3]);
	__m512d a2_1 = _mm512_load_pd(&q[ldq*2]);
	__m512d a3_1 = _mm512_load_pd(&q[ldq]);
	__m512d a4_1 = _mm512_load_pd(&q[0]);

	__m512d a1_2 = _mm512_load_pd(&q[(ldq*3)+8]);
	__m512d a2_2 = _mm512_load_pd(&q[(ldq*2)+8]);
	__m512d a3_2 = _mm512_load_pd(&q[ldq+8]);
	__m512d a4_2 = _mm512_load_pd(&q[0+8]);

	__m512d a1_3 = _mm512_load_pd(&q[(ldq*3)+16]);
	__m512d a2_3 = _mm512_load_pd(&q[(ldq*2)+16]);
	__m512d a3_3 = _mm512_load_pd(&q[ldq+16]);
	__m512d a4_3 = _mm512_load_pd(&q[0+16]);

	__m512d h_2_1 = _mm512_set1_pd(hh[ldh+1]);
	__m512d h_3_2 = _mm512_set1_pd(hh[(ldh*2)+1]);
	__m512d h_3_1 = _mm512_set1_pd(hh[(ldh*2)+2]);
	__m512d h_4_3 = _mm512_set1_pd(hh[(ldh*3)+1]);
	__m512d h_4_2 = _mm512_set1_pd(hh[(ldh*3)+2]);
	__m512d h_4_1 = _mm512_set1_pd(hh[(ldh*3)+3]);

	__m512d w1 = _mm512_FMA_pd(a3_1, h_4_3, a4_1);
	w1 = _mm512_FMA_pd(a2_1, h_4_2, w1);
	w1 = _mm512_FMA_pd(a1_1, h_4_1, w1);
	__m512d z1 = _mm512_FMA_pd(a2_1, h_3_2, a3_1);
	z1 = _mm512_FMA_pd(a1_1, h_3_1, z1);
	__m512d y1 = _mm512_FMA_pd(a1_1, h_2_1, a2_1);
	__m512d x1 = a1_1;

	__m512d w2 = _mm512_FMA_pd(a3_2, h_4_3, a4_2);
	w2 = _mm512_FMA_pd(a2_2, h_4_2, w2);
	w2 = _mm512_FMA_pd(a1_2, h_4_1, w2);
	__m512d z2 = _mm512_FMA_pd(a2_2, h_3_2, a3_2);
	z2 = _mm512_FMA_pd(a1_2, h_3_1, z2);
	__m512d y2 = _mm512_FMA_pd(a1_2, h_2_1, a2_2);
	__m512d x2 = a1_2;

	__m512d w3 = _mm512_FMA_pd(a3_3, h_4_3, a4_3);
	w3 = _mm512_FMA_pd(a2_3, h_4_2, w3);
	w3 = _mm512_FMA_pd(a1_3, h_4_1, w3);
	__m512d z3 = _mm512_FMA_pd(a2_3, h_3_2, a3_3);
	z3 = _mm512_FMA_pd(a1_3, h_3_1, z3);
	__m512d y3 = _mm512_FMA_pd(a1_3, h_2_1, a2_3);
	__m512d x3 = a1_3;

	__m512d q1;
	__m512d q2;
	__m512d q3;

	__m512d h1;
	__m512d h2;
	__m512d h3;
	__m512d h4;

	for(i = 4; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-3]);
		h2 = _mm512_set1_pd(hh[ldh+i-2]);
		h3 = _mm512_set1_pd(hh[(ldh*2)+i-1]);
		h4 = _mm512_set1_pd(hh[(ldh*3)+i]);

		q1 = _mm512_load_pd(&q[i*ldq]);
		q2 = _mm512_load_pd(&q[(i*ldq)+8]);
		q3 = _mm512_load_pd(&q[(i*ldq)+16]);

		x1 = _mm512_FMA_pd(q1, h1, x1);
		y1 = _mm512_FMA_pd(q1, h2, y1);
		z1 = _mm512_FMA_pd(q1, h3, z1);
		w1 = _mm512_FMA_pd(q1, h4, w1);

		x2 = _mm512_FMA_pd(q2, h1, x2);
		y2 = _mm512_FMA_pd(q2, h2, y2);
		z2 = _mm512_FMA_pd(q2, h3, z2);
		w2 = _mm512_FMA_pd(q2, h4, w2);

		x3 = _mm512_FMA_pd(q3, h1, x3);
		y3 = _mm512_FMA_pd(q3, h2, y3);
		z3 = _mm512_FMA_pd(q3, h3, z3);
		w3 = _mm512_FMA_pd(q3, h4, w3);

	}

	h1 = _mm512_set1_pd(hh[nb-3]);
	h2 = _mm512_set1_pd(hh[ldh+nb-2]);
	h3 = _mm512_set1_pd(hh[(ldh*2)+nb-1]);

	q1 = _mm512_load_pd(&q[nb*ldq]);
	q2 = _mm512_load_pd(&q[(nb*ldq)+8]);
	q3 = _mm512_load_pd(&q[(nb*ldq)+16]);

	x1 = _mm512_FMA_pd(q1, h1, x1);
	y1 = _mm512_FMA_pd(q1, h2, y1);
	z1 = _mm512_FMA_pd(q1, h3, z1);

	x2 = _mm512_FMA_pd(q2, h1, x2);
	y2 = _mm512_FMA_pd(q2, h2, y2);
	z2 = _mm512_FMA_pd(q2, h3, z2);

	x3 = _mm512_FMA_pd(q3, h1, x3);
	y3 = _mm512_FMA_pd(q3, h2, y3);
	z3 = _mm512_FMA_pd(q3, h3, z3);

	h1 = _mm512_set1_pd(hh[nb-2]);
	h2 = _mm512_set1_pd(hh[(ldh*1)+nb-1]);

	q1 = _mm512_load_pd(&q[(nb+1)*ldq]);
	q2 = _mm512_load_pd(&q[((nb+1)*ldq)+8]);
	q3 = _mm512_load_pd(&q[((nb+1)*ldq)+16]);

	x1 = _mm512_FMA_pd(q1, h1, x1);
	y1 = _mm512_FMA_pd(q1, h2, y1);
	x2 = _mm512_FMA_pd(q2, h1, x2);
	y2 = _mm512_FMA_pd(q2, h2, y2);
	x3 = _mm512_FMA_pd(q3, h1, x3);
	y3 = _mm512_FMA_pd(q3, h2, y3);

	h1 = _mm512_set1_pd(hh[nb-1]);

	q1 = _mm512_load_pd(&q[(nb+2)*ldq]);
	q2 = _mm512_load_pd(&q[((nb+2)*ldq)+8]);
	q3 = _mm512_load_pd(&q[((nb+2)*ldq)+16]);

	x1 = _mm512_FMA_pd(q1, h1, x1);
	x2 = _mm512_FMA_pd(q2, h1, x2);
	x3 = _mm512_FMA_pd(q3, h1, x3);

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [8 x nb+3]
	/////////////////////////////////////////////////////

	__m512d tau1 = _mm512_set1_pd(hh[0]);
	__m512d tau2 = _mm512_set1_pd(hh[ldh]);
	__m512d tau3 = _mm512_set1_pd(hh[ldh*2]);
	__m512d tau4 = _mm512_set1_pd(hh[ldh*3]);

	__m512d vs_1_2 = _mm512_set1_pd(s_1_2);
	__m512d vs_1_3 = _mm512_set1_pd(s_1_3);
	__m512d vs_2_3 = _mm512_set1_pd(s_2_3);
	__m512d vs_1_4 = _mm512_set1_pd(s_1_4);
	__m512d vs_2_4 = _mm512_set1_pd(s_2_4);
	__m512d vs_3_4 = _mm512_set1_pd(s_3_4);

	h1 = tau1;
	x1 = _mm512_mul_pd(x1, h1);
	x2 = _mm512_mul_pd(x2, h1);
	x3 = _mm512_mul_pd(x3, h1);

	h1 = tau2;
	h2 = _mm512_mul_pd(h1, vs_1_2);

	y1 = _mm512_FMSUB_pd(y1, h1, _mm512_mul_pd(x1,h2));
	y2 = _mm512_FMSUB_pd(y2, h1, _mm512_mul_pd(x2,h2));
	y3 = _mm512_FMSUB_pd(y3, h1, _mm512_mul_pd(x3,h2));

	h1 = tau3;
	h2 = _mm512_mul_pd(h1, vs_1_3);
	h3 = _mm512_mul_pd(h1, vs_2_3);

	z1 = _mm512_FMSUB_pd(z1, h1, _mm512_FMA_pd(y1, h3, _mm512_mul_pd(x1,h2)));
	z2 = _mm512_FMSUB_pd(z2, h1, _mm512_FMA_pd(y2, h3, _mm512_mul_pd(x2,h2)));
	z3 = _mm512_FMSUB_pd(z3, h1, _mm512_FMA_pd(y3, h3, _mm512_mul_pd(x3,h2)));

	h1 = tau4;
	h2 = _mm512_mul_pd(h1, vs_1_4);
	h3 = _mm512_mul_pd(h1, vs_2_4);
	h4 = _mm512_mul_pd(h1, vs_3_4);

	w1 = _mm512_FMSUB_pd(w1, h1, _mm512_FMA_pd(z1, h4, _mm512_FMA_pd(y1, h3, _mm512_mul_pd(x1,h2))));
	w2 = _mm512_FMSUB_pd(w2, h1, _mm512_FMA_pd(z2, h4, _mm512_FMA_pd(y2, h3, _mm512_mul_pd(x2,h2))));
	w3 = _mm512_FMSUB_pd(w3, h1, _mm512_FMA_pd(z3, h4, _mm512_FMA_pd(y3, h3, _mm512_mul_pd(x3,h2))));

	q1 = _mm512_load_pd(&q[0]);
	q1 = _mm512_sub_pd(q1, w1);
	_mm512_store_pd(&q[0],q1);

	q2 = _mm512_load_pd(&q[0+8]);
	q2 = _mm512_sub_pd(q2, w2);
	_mm512_store_pd(&q[0+8],q2);

	q3 = _mm512_load_pd(&q[0+16]);
	q3 = _mm512_sub_pd(q3, w3);
	_mm512_store_pd(&q[0+16],q3);


	h4 = _mm512_set1_pd(hh[(ldh*3)+1]);
	q1 = _mm512_load_pd(&q[ldq]);
	q2 = _mm512_load_pd(&q[ldq+8]);
	q3 = _mm512_load_pd(&q[ldq+16]);

	q1 = _mm512_sub_pd(q1, _mm512_FMA_pd(w1, h4, z1));
	_mm512_store_pd(&q[ldq],q1);
	q2 = _mm512_sub_pd(q2, _mm512_FMA_pd(w2, h4, z2));
	_mm512_store_pd(&q[ldq+8],q2);
	q3 = _mm512_sub_pd(q3, _mm512_FMA_pd(w3, h4, z3));
	_mm512_store_pd(&q[ldq+16],q3);


	h3 = _mm512_set1_pd(hh[(ldh*2)+1]);
	h4 = _mm512_set1_pd(hh[(ldh*3)+2]);
	q1 = _mm512_load_pd(&q[ldq*2]);
	q2 = _mm512_load_pd(&q[(ldq*2)+8]);
	q3 = _mm512_load_pd(&q[(ldq*2)+16]);

        q1 = _mm512_sub_pd(q1, y1);
        q1 = _mm512_NFMA_pd(z1, h3, q1);
        q1 = _mm512_NFMA_pd(w1, h4, q1);

	_mm512_store_pd(&q[ldq*2],q1);

        q2 = _mm512_sub_pd(q2, y2);
        q2 = _mm512_NFMA_pd(z2, h3, q2);
        q2 = _mm512_NFMA_pd(w2, h4, q2);

	_mm512_store_pd(&q[(ldq*2)+8],q2);

        q3 = _mm512_sub_pd(q3, y3);
        q3 = _mm512_NFMA_pd(z3, h3, q3);
        q3 = _mm512_NFMA_pd(w3, h4, q3);

	_mm512_store_pd(&q[(ldq*2)+16],q3);

	h2 = _mm512_set1_pd(hh[ldh+1]);
	h3 = _mm512_set1_pd(hh[(ldh*2)+2]);
	h4 = _mm512_set1_pd(hh[(ldh*3)+3]);

	q1 = _mm512_load_pd(&q[ldq*3]);

	q1 = _mm512_sub_pd(q1, x1);
        q1 = _mm512_NFMA_pd(y1, h2, q1);
        q1 = _mm512_NFMA_pd(z1, h3, q1);
        q1 = _mm512_NFMA_pd(w1, h4, q1);

	_mm512_store_pd(&q[ldq*3], q1);

	q2 = _mm512_load_pd(&q[(ldq*3)+8]);

	q2 = _mm512_sub_pd(q2, x2);
        q2 = _mm512_NFMA_pd(y2, h2, q2);
        q2 = _mm512_NFMA_pd(z2, h3, q2);
        q2 = _mm512_NFMA_pd(w2, h4, q2);

	_mm512_store_pd(&q[(ldq*3)+8], q2);

	q3 = _mm512_load_pd(&q[(ldq*3)+16]);

	q3 = _mm512_sub_pd(q3, x3);
        q3 = _mm512_NFMA_pd(y3, h2, q3);
        q3 = _mm512_NFMA_pd(z3, h3, q3);
        q3 = _mm512_NFMA_pd(w3, h4, q3);

	_mm512_store_pd(&q[(ldq*3)+16], q3);

	for (i = 4; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-3]);
		h2 = _mm512_set1_pd(hh[ldh+i-2]);
		h3 = _mm512_set1_pd(hh[(ldh*2)+i-1]);
		h4 = _mm512_set1_pd(hh[(ldh*3)+i]);

		q1 = _mm512_load_pd(&q[i*ldq]);
                q1 = _mm512_NFMA_pd(x1, h1, q1);
                q1 = _mm512_NFMA_pd(y1, h2, q1);
                q1 = _mm512_NFMA_pd(z1, h3, q1);
                q1 = _mm512_NFMA_pd(w1, h4, q1);
		_mm512_store_pd(&q[i*ldq],q1);

		q2 = _mm512_load_pd(&q[(i*ldq)+8]);
                q2 = _mm512_NFMA_pd(x2, h1, q2);
                q2 = _mm512_NFMA_pd(y2, h2, q2);
                q2 = _mm512_NFMA_pd(z2, h3, q2);
                q2 = _mm512_NFMA_pd(w2, h4, q2);
		_mm512_store_pd(&q[(i*ldq)+8],q2);

		q3 = _mm512_load_pd(&q[(i*ldq)+16]);
                q3 = _mm512_NFMA_pd(x3, h1, q3);
                q3 = _mm512_NFMA_pd(y3, h2, q3);
                q3 = _mm512_NFMA_pd(z3, h3, q3);
                q3 = _mm512_NFMA_pd(w3, h4, q3);
		_mm512_store_pd(&q[(i*ldq)+16],q3);

	}

	h1 = _mm512_set1_pd(hh[nb-3]);
	h2 = _mm512_set1_pd(hh[ldh+nb-2]);
	h3 = _mm512_set1_pd(hh[(ldh*2)+nb-1]);
	q1 = _mm512_load_pd(&q[nb*ldq]);

	q1 = _mm512_NFMA_pd(x1, h1, q1);
        q1 = _mm512_NFMA_pd(y1, h2, q1);
        q1 = _mm512_NFMA_pd(z1, h3, q1);

	_mm512_store_pd(&q[nb*ldq],q1);

	q2 = _mm512_load_pd(&q[(nb*ldq)+8]);

	q2 = _mm512_NFMA_pd(x2, h1, q2);
        q2 = _mm512_NFMA_pd(y2, h2, q2);
        q2 = _mm512_NFMA_pd(z2, h3, q2);

	_mm512_store_pd(&q[(nb*ldq)+8],q2);

	q3 = _mm512_load_pd(&q[(nb*ldq)+16]);

	q3 = _mm512_NFMA_pd(x3, h1, q3);
        q3 = _mm512_NFMA_pd(y3, h2, q3);
        q3 = _mm512_NFMA_pd(z3, h3, q3);

	_mm512_store_pd(&q[(nb*ldq)+16],q3);


	h1 = _mm512_set1_pd(hh[nb-2]);
	h2 = _mm512_set1_pd(hh[ldh+nb-1]);
	q1 = _mm512_load_pd(&q[(nb+1)*ldq]);

	q1 = _mm512_NFMA_pd(x1, h1, q1);
	q1 = _mm512_NFMA_pd(y1, h2, q1);

	_mm512_store_pd(&q[(nb+1)*ldq],q1);

	q2 = _mm512_load_pd(&q[((nb+1)*ldq)+8]);

	q2 = _mm512_NFMA_pd(x2, h1, q2);
	q2 = _mm512_NFMA_pd(y2, h2, q2);

	_mm512_store_pd(&q[((nb+1)*ldq)+8],q2);

	q3 = _mm512_load_pd(&q[((nb+1)*ldq)+16]);

	q3 = _mm512_NFMA_pd(x3, h1, q3);
	q3 = _mm512_NFMA_pd(y3, h2, q3);

	_mm512_store_pd(&q[((nb+1)*ldq)+16],q3);


	h1 = _mm512_set1_pd(hh[nb-1]);
	q1 = _mm512_load_pd(&q[(nb+2)*ldq]);

	q1 = _mm512_NFMA_pd(x1, h1, q1);

	_mm512_store_pd(&q[(nb+2)*ldq],q1);

	q2 = _mm512_load_pd(&q[((nb+2)*ldq)+8]);

	q2 = _mm512_NFMA_pd(x2, h1, q2);

	_mm512_store_pd(&q[((nb+2)*ldq)+8],q2);

	q3 = _mm512_load_pd(&q[((nb+2)*ldq)+16]);

	q3 = _mm512_NFMA_pd(x3, h1, q3);

	_mm512_store_pd(&q[((nb+2)*ldq)+16],q3);

}

/**
 * Unrolled kernel that computes
 * 16 rows of Q simultaneously, a
 * matrix Vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_16_AVX512_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m512d a1_1 = _mm512_load_pd(&q[ldq*3]);
	__m512d a2_1 = _mm512_load_pd(&q[ldq*2]);
	__m512d a3_1 = _mm512_load_pd(&q[ldq]);
	__m512d a4_1 = _mm512_load_pd(&q[0]);

	__m512d a1_2 = _mm512_load_pd(&q[(ldq*3)+8]);
	__m512d a2_2 = _mm512_load_pd(&q[(ldq*2)+8]);
	__m512d a3_2 = _mm512_load_pd(&q[ldq+8]);
	__m512d a4_2 = _mm512_load_pd(&q[0+8]);

	__m512d h_2_1 = _mm512_set1_pd(hh[ldh+1]);
	__m512d h_3_2 = _mm512_set1_pd(hh[(ldh*2)+1]);
	__m512d h_3_1 = _mm512_set1_pd(hh[(ldh*2)+2]);
	__m512d h_4_3 = _mm512_set1_pd(hh[(ldh*3)+1]);
	__m512d h_4_2 = _mm512_set1_pd(hh[(ldh*3)+2]);
	__m512d h_4_1 = _mm512_set1_pd(hh[(ldh*3)+3]);

	__m512d w1 = _mm512_FMA_pd(a3_1, h_4_3, a4_1);
	w1 = _mm512_FMA_pd(a2_1, h_4_2, w1);
	w1 = _mm512_FMA_pd(a1_1, h_4_1, w1);
	__m512d z1 = _mm512_FMA_pd(a2_1, h_3_2, a3_1);
	z1 = _mm512_FMA_pd(a1_1, h_3_1, z1);
	__m512d y1 = _mm512_FMA_pd(a1_1, h_2_1, a2_1);
	__m512d x1 = a1_1;

	__m512d q1;
	__m512d q2;

	__m512d w2 = _mm512_FMA_pd(a3_2, h_4_3, a4_2);
	w2 = _mm512_FMA_pd(a2_2, h_4_2, w2);
	w2 = _mm512_FMA_pd(a1_2, h_4_1, w2);
	__m512d z2 = _mm512_FMA_pd(a2_2, h_3_2, a3_2);
	z2 = _mm512_FMA_pd(a1_2, h_3_1, z2);
	__m512d y2 = _mm512_FMA_pd(a1_2, h_2_1, a2_2);
	__m512d x2 = a1_2;

	__m512d h1;
	__m512d h2;
	__m512d h3;
	__m512d h4;

	for(i = 4; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-3]);
		h2 = _mm512_set1_pd(hh[ldh+i-2]);
		h3 = _mm512_set1_pd(hh[(ldh*2)+i-1]);
		h4 = _mm512_set1_pd(hh[(ldh*3)+i]);

		q1 = _mm512_load_pd(&q[i*ldq]);
		q2 = _mm512_load_pd(&q[(i*ldq)+8]);

		x1 = _mm512_FMA_pd(q1, h1, x1);
		y1 = _mm512_FMA_pd(q1, h2, y1);
		z1 = _mm512_FMA_pd(q1, h3, z1);
		w1 = _mm512_FMA_pd(q1, h4, w1);

		x2 = _mm512_FMA_pd(q2, h1, x2);
		y2 = _mm512_FMA_pd(q2, h2, y2);
		z2 = _mm512_FMA_pd(q2, h3, z2);
		w2 = _mm512_FMA_pd(q2, h4, w2);
	}

	h1 = _mm512_set1_pd(hh[nb-3]);
	h2 = _mm512_set1_pd(hh[ldh+nb-2]);
	h3 = _mm512_set1_pd(hh[(ldh*2)+nb-1]);

	q1 = _mm512_load_pd(&q[nb*ldq]);
	q2 = _mm512_load_pd(&q[(nb*ldq)+8]);

	x1 = _mm512_FMA_pd(q1, h1, x1);
	y1 = _mm512_FMA_pd(q1, h2, y1);
	z1 = _mm512_FMA_pd(q1, h3, z1);

	x2 = _mm512_FMA_pd(q2, h1, x2);
	y2 = _mm512_FMA_pd(q2, h2, y2);
	z2 = _mm512_FMA_pd(q2, h3, z2);

	h1 = _mm512_set1_pd(hh[nb-2]);
	h2 = _mm512_set1_pd(hh[(ldh*1)+nb-1]);

	q1 = _mm512_load_pd(&q[(nb+1)*ldq]);
	q2 = _mm512_load_pd(&q[((nb+1)*ldq)+8]);

	x1 = _mm512_FMA_pd(q1, h1, x1);
	y1 = _mm512_FMA_pd(q1, h2, y1);
	x2 = _mm512_FMA_pd(q2, h1, x2);
	y2 = _mm512_FMA_pd(q2, h2, y2);

	h1 = _mm512_set1_pd(hh[nb-1]);

	q1 = _mm512_load_pd(&q[(nb+2)*ldq]);
	q2 = _mm512_load_pd(&q[((nb+2)*ldq)+8]);

	x1 = _mm512_FMA_pd(q1, h1, x1);
	x2 = _mm512_FMA_pd(q2, h1, x2);

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [8 x nb+3]
	/////////////////////////////////////////////////////

	__m512d tau1 = _mm512_set1_pd(hh[0]);
	__m512d tau2 = _mm512_set1_pd(hh[ldh]);
	__m512d tau3 = _mm512_set1_pd(hh[ldh*2]);
	__m512d tau4 = _mm512_set1_pd(hh[ldh*3]);

	__m512d vs_1_2 = _mm512_set1_pd(s_1_2);
	__m512d vs_1_3 = _mm512_set1_pd(s_1_3);
	__m512d vs_2_3 = _mm512_set1_pd(s_2_3);
	__m512d vs_1_4 = _mm512_set1_pd(s_1_4);
	__m512d vs_2_4 = _mm512_set1_pd(s_2_4);
	__m512d vs_3_4 = _mm512_set1_pd(s_3_4);

	h1 = tau1;
	x1 = _mm512_mul_pd(x1, h1);
	x2 = _mm512_mul_pd(x2, h1);

	h1 = tau2;
	h2 = _mm512_mul_pd(h1, vs_1_2);

	y1 = _mm512_FMSUB_pd(y1, h1, _mm512_mul_pd(x1,h2));
	y2 = _mm512_FMSUB_pd(y2, h1, _mm512_mul_pd(x2,h2));

	h1 = tau3;
	h2 = _mm512_mul_pd(h1, vs_1_3);
	h3 = _mm512_mul_pd(h1, vs_2_3);

	z1 = _mm512_FMSUB_pd(z1, h1, _mm512_FMA_pd(y1, h3, _mm512_mul_pd(x1,h2)));
	z2 = _mm512_FMSUB_pd(z2, h1, _mm512_FMA_pd(y2, h3, _mm512_mul_pd(x2,h2)));

	h1 = tau4;
	h2 = _mm512_mul_pd(h1, vs_1_4);
	h3 = _mm512_mul_pd(h1, vs_2_4);
	h4 = _mm512_mul_pd(h1, vs_3_4);

	w1 = _mm512_FMSUB_pd(w1, h1, _mm512_FMA_pd(z1, h4, _mm512_FMA_pd(y1, h3, _mm512_mul_pd(x1,h2))));
	w2 = _mm512_FMSUB_pd(w2, h1, _mm512_FMA_pd(z2, h4, _mm512_FMA_pd(y2, h3, _mm512_mul_pd(x2,h2))));

	q1 = _mm512_load_pd(&q[0]);
	q1 = _mm512_sub_pd(q1, w1);
	_mm512_store_pd(&q[0],q1);

	q2 = _mm512_load_pd(&q[0+8]);
	q2 = _mm512_sub_pd(q2, w2);
	_mm512_store_pd(&q[0+8],q2);


	h4 = _mm512_set1_pd(hh[(ldh*3)+1]);
	q1 = _mm512_load_pd(&q[ldq]);
	q2 = _mm512_load_pd(&q[ldq+8]);

	q1 = _mm512_sub_pd(q1, _mm512_FMA_pd(w1, h4, z1));
	_mm512_store_pd(&q[ldq],q1);
	q2 = _mm512_sub_pd(q2, _mm512_FMA_pd(w2, h4, z2));
	_mm512_store_pd(&q[ldq+8],q2);

	h3 = _mm512_set1_pd(hh[(ldh*2)+1]);
	h4 = _mm512_set1_pd(hh[(ldh*3)+2]);
	q1 = _mm512_load_pd(&q[ldq*2]);
	q2 = _mm512_load_pd(&q[(ldq*2)+8]);

        q1 = _mm512_sub_pd(q1, y1);
        q1 = _mm512_NFMA_pd(z1, h3, q1);
        q1 = _mm512_NFMA_pd(w1, h4, q1);

	_mm512_store_pd(&q[ldq*2],q1);

        q2 = _mm512_sub_pd(q2, y2);
        q2 = _mm512_NFMA_pd(z2, h3, q2);
        q2 = _mm512_NFMA_pd(w2, h4, q2);

	_mm512_store_pd(&q[(ldq*2)+8],q2);

	h2 = _mm512_set1_pd(hh[ldh+1]);
	h3 = _mm512_set1_pd(hh[(ldh*2)+2]);
	h4 = _mm512_set1_pd(hh[(ldh*3)+3]);

	q1 = _mm512_load_pd(&q[ldq*3]);

	q1 = _mm512_sub_pd(q1, x1);
        q1 = _mm512_NFMA_pd(y1, h2, q1);
        q1 = _mm512_NFMA_pd(z1, h3, q1);
        q1 = _mm512_NFMA_pd(w1, h4, q1);

	_mm512_store_pd(&q[ldq*3], q1);

	q2 = _mm512_load_pd(&q[(ldq*3)+8]);

	q2 = _mm512_sub_pd(q2, x2);
        q2 = _mm512_NFMA_pd(y2, h2, q2);
        q2 = _mm512_NFMA_pd(z2, h3, q2);
        q2 = _mm512_NFMA_pd(w2, h4, q2);

	_mm512_store_pd(&q[(ldq*3)+8], q2);

	for (i = 4; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-3]);
		h2 = _mm512_set1_pd(hh[ldh+i-2]);
		h3 = _mm512_set1_pd(hh[(ldh*2)+i-1]);
		h4 = _mm512_set1_pd(hh[(ldh*3)+i]);

		q1 = _mm512_load_pd(&q[i*ldq]);
                q1 = _mm512_NFMA_pd(x1, h1, q1);
                q1 = _mm512_NFMA_pd(y1, h2, q1);
                q1 = _mm512_NFMA_pd(z1, h3, q1);
                q1 = _mm512_NFMA_pd(w1, h4, q1);
		_mm512_store_pd(&q[i*ldq],q1);

		q2 = _mm512_load_pd(&q[(i*ldq)+8]);
                q2 = _mm512_NFMA_pd(x2, h1, q2);
                q2 = _mm512_NFMA_pd(y2, h2, q2);
                q2 = _mm512_NFMA_pd(z2, h3, q2);
                q2 = _mm512_NFMA_pd(w2, h4, q2);
		_mm512_store_pd(&q[(i*ldq)+8],q2);

	}

	h1 = _mm512_set1_pd(hh[nb-3]);
	h2 = _mm512_set1_pd(hh[ldh+nb-2]);
	h3 = _mm512_set1_pd(hh[(ldh*2)+nb-1]);
	q1 = _mm512_load_pd(&q[nb*ldq]);

	q1 = _mm512_NFMA_pd(x1, h1, q1);
        q1 = _mm512_NFMA_pd(y1, h2, q1);
        q1 = _mm512_NFMA_pd(z1, h3, q1);

	_mm512_store_pd(&q[nb*ldq],q1);

	q2 = _mm512_load_pd(&q[(nb*ldq)+8]);

	q2 = _mm512_NFMA_pd(x2, h1, q2);
        q2 = _mm512_NFMA_pd(y2, h2, q2);
        q2 = _mm512_NFMA_pd(z2, h3, q2);

	_mm512_store_pd(&q[(nb*ldq)+8],q2);

	h1 = _mm512_set1_pd(hh[nb-2]);
	h2 = _mm512_set1_pd(hh[ldh+nb-1]);
	q1 = _mm512_load_pd(&q[(nb+1)*ldq]);

	q1 = _mm512_NFMA_pd(x1, h1, q1);
	q1 = _mm512_NFMA_pd(y1, h2, q1);

	_mm512_store_pd(&q[(nb+1)*ldq],q1);

	q2 = _mm512_load_pd(&q[((nb+1)*ldq)+8]);

	q2 = _mm512_NFMA_pd(x2, h1, q2);
	q2 = _mm512_NFMA_pd(y2, h2, q2);

	_mm512_store_pd(&q[((nb+1)*ldq)+8],q2);

	h1 = _mm512_set1_pd(hh[nb-1]);
	q1 = _mm512_load_pd(&q[(nb+2)*ldq]);

	q1 = _mm512_NFMA_pd(x1, h1, q1);

	_mm512_store_pd(&q[(nb+2)*ldq],q1);

	q2 = _mm512_load_pd(&q[((nb+2)*ldq)+8]);

	q2 = _mm512_NFMA_pd(x2, h1, q2);

	_mm512_store_pd(&q[((nb+2)*ldq)+8],q2);
}

/**
 * Unrolled kernel that computes
 * 8 rows of Q simultaneously, a
 * matrix Vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_8_AVX512_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m512d a1_1 = _mm512_load_pd(&q[ldq*3]);
	__m512d a2_1 = _mm512_load_pd(&q[ldq*2]);
	__m512d a3_1 = _mm512_load_pd(&q[ldq]);
	__m512d a4_1 = _mm512_load_pd(&q[0]);

	__m512d h_2_1 = _mm512_set1_pd(hh[ldh+1]);
	__m512d h_3_2 = _mm512_set1_pd(hh[(ldh*2)+1]);
	__m512d h_3_1 = _mm512_set1_pd(hh[(ldh*2)+2]);
	__m512d h_4_3 = _mm512_set1_pd(hh[(ldh*3)+1]);
	__m512d h_4_2 = _mm512_set1_pd(hh[(ldh*3)+2]);
	__m512d h_4_1 = _mm512_set1_pd(hh[(ldh*3)+3]);

	__m512d w1 = _mm512_FMA_pd(a3_1, h_4_3, a4_1);
	w1 = _mm512_FMA_pd(a2_1, h_4_2, w1);
	w1 = _mm512_FMA_pd(a1_1, h_4_1, w1);
	__m512d z1 = _mm512_FMA_pd(a2_1, h_3_2, a3_1);
	z1 = _mm512_FMA_pd(a1_1, h_3_1, z1);
	__m512d y1 = _mm512_FMA_pd(a1_1, h_2_1, a2_1);
	__m512d x1 = a1_1;

	__m512d q1;

	__m512d h1;
	__m512d h2;
	__m512d h3;
	__m512d h4;

	for(i = 4; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-3]);
		h2 = _mm512_set1_pd(hh[ldh+i-2]);
		h3 = _mm512_set1_pd(hh[(ldh*2)+i-1]);
		h4 = _mm512_set1_pd(hh[(ldh*3)+i]);

		q1 = _mm512_load_pd(&q[i*ldq]);

		x1 = _mm512_FMA_pd(q1, h1, x1);
		y1 = _mm512_FMA_pd(q1, h2, y1);
		z1 = _mm512_FMA_pd(q1, h3, z1);
		w1 = _mm512_FMA_pd(q1, h4, w1);

	}

	h1 = _mm512_set1_pd(hh[nb-3]);
	h2 = _mm512_set1_pd(hh[ldh+nb-2]);
	h3 = _mm512_set1_pd(hh[(ldh*2)+nb-1]);

	q1 = _mm512_load_pd(&q[nb*ldq]);

	x1 = _mm512_FMA_pd(q1, h1, x1);
	y1 = _mm512_FMA_pd(q1, h2, y1);
	z1 = _mm512_FMA_pd(q1, h3, z1);

	h1 = _mm512_set1_pd(hh[nb-2]);
	h2 = _mm512_set1_pd(hh[(ldh*1)+nb-1]);

	q1 = _mm512_load_pd(&q[(nb+1)*ldq]);

	x1 = _mm512_FMA_pd(q1, h1, x1);
	y1 = _mm512_FMA_pd(q1, h2, y1);

	h1 = _mm512_set1_pd(hh[nb-1]);

	q1 = _mm512_load_pd(&q[(nb+2)*ldq]);

	x1 = _mm512_FMA_pd(q1, h1, x1);

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [8 x nb+3]
	/////////////////////////////////////////////////////

	__m512d tau1 = _mm512_set1_pd(hh[0]);
	__m512d tau2 = _mm512_set1_pd(hh[ldh]);
	__m512d tau3 = _mm512_set1_pd(hh[ldh*2]);
	__m512d tau4 = _mm512_set1_pd(hh[ldh*3]);

	__m512d vs_1_2 = _mm512_set1_pd(s_1_2);
	__m512d vs_1_3 = _mm512_set1_pd(s_1_3);
	__m512d vs_2_3 = _mm512_set1_pd(s_2_3);
	__m512d vs_1_4 = _mm512_set1_pd(s_1_4);
	__m512d vs_2_4 = _mm512_set1_pd(s_2_4);
	__m512d vs_3_4 = _mm512_set1_pd(s_3_4);

	h1 = tau1;
	x1 = _mm512_mul_pd(x1, h1);

	h1 = tau2;
	h2 = _mm512_mul_pd(h1, vs_1_2);

	y1 = _mm512_FMSUB_pd(y1, h1, _mm512_mul_pd(x1,h2));

	h1 = tau3;
	h2 = _mm512_mul_pd(h1, vs_1_3);
	h3 = _mm512_mul_pd(h1, vs_2_3);

	z1 = _mm512_FMSUB_pd(z1, h1, _mm512_FMA_pd(y1, h3, _mm512_mul_pd(x1,h2)));

	h1 = tau4;
	h2 = _mm512_mul_pd(h1, vs_1_4);
	h3 = _mm512_mul_pd(h1, vs_2_4);
	h4 = _mm512_mul_pd(h1, vs_3_4);

	w1 = _mm512_FMSUB_pd(w1, h1, _mm512_FMA_pd(z1, h4, _mm512_FMA_pd(y1, h3, _mm512_mul_pd(x1,h2))));

	q1 = _mm512_load_pd(&q[0]);
	q1 = _mm512_sub_pd(q1, w1);
	_mm512_store_pd(&q[0],q1);

	h4 = _mm512_set1_pd(hh[(ldh*3)+1]);
	q1 = _mm512_load_pd(&q[ldq]);

	q1 = _mm512_sub_pd(q1, _mm512_FMA_pd(w1, h4, z1));

	_mm512_store_pd(&q[ldq],q1);

	h3 = _mm512_set1_pd(hh[(ldh*2)+1]);
	h4 = _mm512_set1_pd(hh[(ldh*3)+2]);
	q1 = _mm512_load_pd(&q[ldq*2]);

        q1 = _mm512_sub_pd(q1, y1);
        q1 = _mm512_NFMA_pd(z1, h3, q1);
        q1 = _mm512_NFMA_pd(w1, h4, q1);

	_mm512_store_pd(&q[ldq*2],q1);

	h2 = _mm512_set1_pd(hh[ldh+1]);
	h3 = _mm512_set1_pd(hh[(ldh*2)+2]);
	h4 = _mm512_set1_pd(hh[(ldh*3)+3]);

	q1 = _mm512_load_pd(&q[ldq*3]);

	q1 = _mm512_sub_pd(q1, x1);
        q1 = _mm512_NFMA_pd(y1, h2, q1);
        q1 = _mm512_NFMA_pd(z1, h3, q1);
        q1 = _mm512_NFMA_pd(w1, h4, q1);

	_mm512_store_pd(&q[ldq*3], q1);

	for (i = 4; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-3]);
		h2 = _mm512_set1_pd(hh[ldh+i-2]);
		h3 = _mm512_set1_pd(hh[(ldh*2)+i-1]);
		h4 = _mm512_set1_pd(hh[(ldh*3)+i]);

		q1 = _mm512_load_pd(&q[i*ldq]);
                q1 = _mm512_NFMA_pd(x1, h1, q1);
                q1 = _mm512_NFMA_pd(y1, h2, q1);
                q1 = _mm512_NFMA_pd(z1, h3, q1);
                q1 = _mm512_NFMA_pd(w1, h4, q1);
		_mm512_store_pd(&q[i*ldq],q1);
	}

	h1 = _mm512_set1_pd(hh[nb-3]);
	h2 = _mm512_set1_pd(hh[ldh+nb-2]);
	h3 = _mm512_set1_pd(hh[(ldh*2)+nb-1]);
	q1 = _mm512_load_pd(&q[nb*ldq]);

	q1 = _mm512_NFMA_pd(x1, h1, q1);
        q1 = _mm512_NFMA_pd(y1, h2, q1);
        q1 = _mm512_NFMA_pd(z1, h3, q1);

	_mm512_store_pd(&q[nb*ldq],q1);

	h1 = _mm512_set1_pd(hh[nb-2]);
	h2 = _mm512_set1_pd(hh[ldh+nb-1]);
	q1 = _mm512_load_pd(&q[(nb+1)*ldq]);

	q1 = _mm512_NFMA_pd(x1, h1, q1);
	q1 = _mm512_NFMA_pd(y1, h2, q1);

	_mm512_store_pd(&q[(nb+1)*ldq],q1);

	h1 = _mm512_set1_pd(hh[nb-1]);
	q1 = _mm512_load_pd(&q[(nb+2)*ldq]);

	q1 = _mm512_NFMA_pd(x1, h1, q1);

	_mm512_store_pd(&q[(nb+2)*ldq],q1);


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

