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

#include <x86intrin.h>

#define __forceinline __attribute__((always_inline)) static
#ifdef DOUBLE_PRECISION_REAL
#define __SSE_DATATYPE __m128d
#define _SSE_LOAD _mm_load_pd
#define _SSE_ADD _mm_add_pd
#define _SSE_MUL _mm_mul_pd
#define _SSE_XOR _mm_xor_pd
#define _SSE_STORE _mm_store_pd
#define offset 2
#endif

#ifdef SINGLE_PRECISION_REAL
#define __SSE_DATATYPE __m128
#define _SSE_LOAD _mm_load_ps
#define _SSE_ADD _mm_add_ps
#define _SSE_MUL _mm_mul_ps
#define _SSE_XOR _mm_xor_ps
#define _SSE_STORE _mm_store_ps
#define offset 4
#endif


#ifdef HAVE_SSE_INTRINSICS
#undef __AVX__
#endif

//Forward declaration
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_4_SSE_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_8_SSE_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_12_SSE_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s);
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_4_SSE_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_8_SSE_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_12_SSE_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
#endif


#ifdef DOUBLE_PRECISION_REAL
void double_hh_trafo_real_sse_2hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif
#ifdef SINGLE_PRECISION_REAL
void double_hh_trafo_real_sse_2hv_single_(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif

/*
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine double_hh_trafo_real_sse_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="double_hh_trafo_real_sse_2hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int) :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value  :: q
!f>     real(kind=c_double) :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine double_hh_trafo_real_sse_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="double_hh_trafo_real_sse_2hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)      :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

#ifdef DOUBLE_PRECISION_REAL
void double_hh_trafo_real_sse_2hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
#ifdef SINGLE_PRECISION_REAL
void double_hh_trafo_real_sse_2hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;

	// calculating scalar product to compute
	// 2 householder vectors simultaneously
#ifdef DOUBLE_PRECISION_REAL
	double s = hh[(ldh)+1]*1.0;
#endif
#ifdef SINGLE_PRECISION_REAL
	float s = hh[(ldh)+1]*1.0;
#endif
	#pragma ivdep
	for (i = 2; i < nb; i++)
	{
		s += hh[i-1] * hh[(i+ldh)];
	}

	// Production level kernel calls with padding
	for (i = 0; i < nq-8; i+=12)
	{
#ifdef DOUBLE_PRECISION_REAL
		hh_trafo_kernel_12_SSE_2hv_double(&q[i], hh, nb, ldq, ldh, s);
#endif
#ifdef SINGLE_PRECISION_REAL
	        hh_trafo_kernel_12_SSE_2hv_single(&q[i], hh, nb, ldq, ldh, s);
#endif
	}
	if (nq == i)
	{
		return;
	}
#ifdef SINGLE_PRECISION_REAL
        if (nq-i == 8)
	{
		hh_trafo_kernel_8_SSE_2hv_single(&q[i], hh, nb, ldq, ldh, s);
	}
#endif
	else
	{
#ifdef DOUBLE_PRECISION_REAL
		if (nq-i > 4)
		{
			hh_trafo_kernel_8_SSE_2hv_double(&q[i], hh, nb, ldq, ldh, s);
		}
		else if (nq-i > 0)
		{
			hh_trafo_kernel_4_SSE_2hv_double(&q[i], hh, nb, ldq, ldh, s);
		}
#endif
#ifdef SINGLE_PRECISION_REAL
		hh_trafo_kernel_4_SSE_2hv_single(&q[i], hh, nb, ldq, ldh, s);
#endif
	}
}

/**
 * Unrolled kernel that computes
 * 12 rows of Q simultaneously, a
 * matrix Vector product with two householder
 * vectors + a rank 2 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_12_SSE_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
#endif
#ifdef SINGLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_12_SSE_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [12 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
__SSE_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif

	__SSE_DATATYPE x1 = _SSE_LOAD(&q[ldq]);
	__SSE_DATATYPE x2 = _SSE_LOAD(&q[ldq+offset]);
	__SSE_DATATYPE x3 = _SSE_LOAD(&q[ldq+2*offset]);
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE x4 = _SSE_LOAD(&q[ldq+3*offset]);
	__SSE_DATATYPE x5 = _SSE_LOAD(&q[ldq+4*offset]);
	__SSE_DATATYPE x6 = _SSE_LOAD(&q[ldq+5*offset]);
#endif

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h1 = _mm_loaddup_pd(&hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SSE_DATATYPE h1 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+1])));
#endif
	__SSE_DATATYPE h2;

	__SSE_DATATYPE q1 = _SSE_LOAD(q);
	__SSE_DATATYPE y1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
	__SSE_DATATYPE q2 = _SSE_LOAD(&q[offset]);
	__SSE_DATATYPE y2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
	__SSE_DATATYPE q3 = _SSE_LOAD(&q[2*offset]);
	__SSE_DATATYPE y3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE q4 = _SSE_LOAD(&q[3*offset]);
	__SSE_DATATYPE y4 = _SSE_ADD(q4, _SSE_MUL(x4, h1));
	__SSE_DATATYPE q5 = _SSE_LOAD(&q[4*offset]);
	__SSE_DATATYPE y5 = _SSE_ADD(q5, _SSE_MUL(x5, h1));
	__SSE_DATATYPE q6 = _SSE_LOAD(&q[5*offset]);
	__SSE_DATATYPE y6 = _SSE_ADD(q6, _SSE_MUL(x6, h1));
#endif
	for(i = 2; i < nb; i++)
	{
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_loaddup_pd(&hh[i-1]);
		h2 = _mm_loaddup_pd(&hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[i-1])));
		h2 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+i])));
#endif


		q1 = _SSE_LOAD(&q[i*ldq]);
		x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
		y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
		q2 = _SSE_LOAD(&q[(i*ldq)+offset]);
		x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
		y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
		q3 = _SSE_LOAD(&q[(i*ldq)+2*offset]);
		x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
		y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
#ifdef DOUBLE_PRECISION_REAL
		q4 = _SSE_LOAD(&q[(i*ldq)+3*offset]);
		x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
		y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
		q5 = _SSE_LOAD(&q[(i*ldq)+4*offset]);
		x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));
		y5 = _SSE_ADD(y5, _SSE_MUL(q5,h2));
		q6 = _SSE_LOAD(&q[(i*ldq)+5*offset]);
		x6 = _SSE_ADD(x6, _SSE_MUL(q6,h1));
		y6 = _SSE_ADD(y6, _SSE_MUL(q6,h2));
#endif
	}

#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_loaddup_pd(&hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
        h1 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[nb-1])));
#endif

	q1 = _SSE_LOAD(&q[nb*ldq]);
	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
	q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);
	x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
	q3 = _SSE_LOAD(&q[(nb*ldq)+2*offset]);
	x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
#ifdef DOUBLE_PRECISION_REAL
	q4 = _SSE_LOAD(&q[(nb*ldq)+3*offset]);
	x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
	q5 = _SSE_LOAD(&q[(nb*ldq)+4*offset]);
	x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));
	q6 = _SSE_LOAD(&q[(nb*ldq)+5*offset]);
	x6 = _SSE_ADD(x6, _SSE_MUL(q6,h1));
#endif
	/////////////////////////////////////////////////////
	// Rank-2 update of Q [12 x nb+1]
	/////////////////////////////////////////////////////
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = _mm_loaddup_pd(hh);
	__SSE_DATATYPE tau2 = _mm_loaddup_pd(&hh[ldh]);
	__SSE_DATATYPE vs = _mm_loaddup_pd(&s);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) hh)));
        __SSE_DATATYPE tau2 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh])));
	__SSE_DATATYPE vs = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd((double*) &s)));

#endif

	h1 = _SSE_XOR(tau1, sign);
	x1 = _SSE_MUL(x1, h1);
	x2 = _SSE_MUL(x2, h1);
	x3 = _SSE_MUL(x3, h1);
#ifdef DOUBLE_PRECISION_REAL
	x4 = _SSE_MUL(x4, h1);
	x5 = _SSE_MUL(x5, h1);
	x6 = _SSE_MUL(x6, h1);
#endif
	h1 = _SSE_XOR(tau2, sign);
	h2 = _SSE_MUL(h1, vs);

	y1 = _SSE_ADD(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
	y2 = _SSE_ADD(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
	y3 = _SSE_ADD(_SSE_MUL(y3,h1), _SSE_MUL(x3,h2));
#ifdef DOUBLE_PRECISION_REAL
	y4 = _SSE_ADD(_SSE_MUL(y4,h1), _SSE_MUL(x4,h2));
	y5 = _SSE_ADD(_SSE_MUL(y5,h1), _SSE_MUL(x5,h2));
	y6 = _SSE_ADD(_SSE_MUL(y6,h1), _SSE_MUL(x6,h2));
#endif
	q1 = _SSE_LOAD(q);
	q1 = _SSE_ADD(q1, y1);
	_SSE_STORE(q,q1);
	q2 = _SSE_LOAD(&q[offset]);
	q2 = _SSE_ADD(q2, y2);
	_SSE_STORE(&q[offset],q2);
	q3 = _SSE_LOAD(&q[2*offset]);
	q3 = _SSE_ADD(q3, y3);
	_SSE_STORE(&q[2*offset],q3);
#ifdef DOUBLE_PRECISION_REAL
	q4 = _SSE_LOAD(&q[3*offset]);
	q4 = _SSE_ADD(q4, y4);
	_SSE_STORE(&q[3*offset],q4);
	q5 = _SSE_LOAD(&q[4*offset]);
	q5 = _SSE_ADD(q5, y5);
	_SSE_STORE(&q[4*offset],q5);
	q6 = _SSE_LOAD(&q[5*offset]);
	q6 = _SSE_ADD(q6, y6);
	_SSE_STORE(&q[5*offset],q6);
#endif

#ifdef DOUBLE_PRECISION_REAL
	h2 = _mm_loaddup_pd(&hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
        h2 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+1])));
//	h2 = _mm_castpd_ps(_mm_loaddup_pd(&hh[ldh+1]));
#endif

	q1 = _SSE_LOAD(&q[ldq]);
	q1 = _SSE_ADD(q1, _SSE_ADD(x1, _SSE_MUL(y1, h2)));
	_SSE_STORE(&q[ldq],q1);
	q2 = _SSE_LOAD(&q[ldq+offset]);
	q2 = _SSE_ADD(q2, _SSE_ADD(x2, _SSE_MUL(y2, h2)));
	_SSE_STORE(&q[ldq+offset],q2);
	q3 = _SSE_LOAD(&q[ldq+2*offset]);
	q3 = _SSE_ADD(q3, _SSE_ADD(x3, _SSE_MUL(y3, h2)));
	_SSE_STORE(&q[ldq+2*offset],q3);
#ifdef DOUBLE_PRECISION_REAL
	q4 = _SSE_LOAD(&q[ldq+3*offset]);
	q4 = _SSE_ADD(q4, _SSE_ADD(x4, _SSE_MUL(y4, h2)));
	_SSE_STORE(&q[ldq+3*offset],q4);
	q5 = _SSE_LOAD(&q[ldq+4*offset]);
	q5 = _SSE_ADD(q5, _SSE_ADD(x5, _SSE_MUL(y5, h2)));
	_SSE_STORE(&q[ldq+4*offset],q5);
	q6 = _SSE_LOAD(&q[ldq+5*offset]);
	q6 = _SSE_ADD(q6, _SSE_ADD(x6, _SSE_MUL(y6, h2)));
	_SSE_STORE(&q[ldq+5*offset],q6);
#endif
	for (i = 2; i < nb; i++)
	{
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_loaddup_pd(&hh[i-1]);
		h2 = _mm_loaddup_pd(&hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[i-1])));
//		h1 = _mm_castpd_ps(_mm_loaddup_pd(&hh[i-1]));

		h2 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+i])));
//		h2 = _mm_castpd_ps(_mm_loaddup_pd(&hh[ldh+i]));
#endif

		q1 = _SSE_LOAD(&q[i*ldq]);
		q1 = _SSE_ADD(q1, _SSE_ADD(_SSE_MUL(x1,h1), _SSE_MUL(y1, h2)));
		_SSE_STORE(&q[i*ldq],q1);
		q2 = _SSE_LOAD(&q[(i*ldq)+offset]);
		q2 = _SSE_ADD(q2, _SSE_ADD(_SSE_MUL(x2,h1), _SSE_MUL(y2, h2)));
		_SSE_STORE(&q[(i*ldq)+offset],q2);
		q3 = _SSE_LOAD(&q[(i*ldq)+2*offset]);
		q3 = _SSE_ADD(q3, _SSE_ADD(_SSE_MUL(x3,h1), _SSE_MUL(y3, h2)));
		_SSE_STORE(&q[(i*ldq)+2*offset],q3);
#ifdef DOUBLE_PRECISION_REAL
		q4 = _SSE_LOAD(&q[(i*ldq)+3*offset]);
		q4 = _SSE_ADD(q4, _SSE_ADD(_SSE_MUL(x4,h1), _SSE_MUL(y4, h2)));
		_SSE_STORE(&q[(i*ldq)+3*offset],q4);
		q5 = _SSE_LOAD(&q[(i*ldq)+4*offset]);
		q5 = _SSE_ADD(q5, _SSE_ADD(_SSE_MUL(x5,h1), _SSE_MUL(y5, h2)));
		_SSE_STORE(&q[(i*ldq)+4*offset],q5);
		q6 = _SSE_LOAD(&q[(i*ldq)+5*offset]);
		q6 = _SSE_ADD(q6, _SSE_ADD(_SSE_MUL(x6,h1), _SSE_MUL(y6, h2)));
		_SSE_STORE(&q[(i*ldq)+5*offset],q6);
#endif
	}
#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_loaddup_pd(&hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[nb-1])));
//	h1 = _mm_castpd_ps(_mm_loaddup_pd(&hh[nb-1]));
#endif
	q1 = _SSE_LOAD(&q[nb*ldq]);
	q1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
	_SSE_STORE(&q[nb*ldq],q1);
	q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);
	q2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
	_SSE_STORE(&q[(nb*ldq)+offset],q2);
	q3 = _SSE_LOAD(&q[(nb*ldq)+2*offset]);
	q3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
	_SSE_STORE(&q[(nb*ldq)+2*offset],q3);
#ifdef DOUBLE_PRECISION_REAL
	q4 = _SSE_LOAD(&q[(nb*ldq)+3*offset]);
	q4 = _SSE_ADD(q4, _SSE_MUL(x4, h1));
	_SSE_STORE(&q[(nb*ldq)+3*offset],q4);
	q5 = _SSE_LOAD(&q[(nb*ldq)+4*offset]);
	q5 = _SSE_ADD(q5, _SSE_MUL(x5, h1));
	_SSE_STORE(&q[(nb*ldq)+4*offset],q5);
	q6 = _SSE_LOAD(&q[(nb*ldq)+5*offset]);
	q6 = _SSE_ADD(q6, _SSE_MUL(x6, h1));
	_SSE_STORE(&q[(nb*ldq)+5*offset],q6);
#endif
}

/**
 * Unrolled kernel that computes
 * 8 rows of Q simultaneously, a
 * matrix Vector product with two householder
 * vectors + a rank 2 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_8_SSE_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_8_SSE_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [8 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif

	__SSE_DATATYPE x1 = _SSE_LOAD(&q[ldq]);
	__SSE_DATATYPE x2 = _SSE_LOAD(&q[ldq+offset]);
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE x3 = _SSE_LOAD(&q[ldq+2*offset]);
	__SSE_DATATYPE x4 = _SSE_LOAD(&q[ldq+3*offset]);
#endif

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h1 = _mm_loaddup_pd(&hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SSE_DATATYPE x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+1]));
        __SSE_DATATYPE x4 ;
        __SSE_DATATYPE h1 = _mm_moveldup_ps(x3);
#endif
	__SSE_DATATYPE h2;

	__SSE_DATATYPE q1 = _SSE_LOAD(q);
	__SSE_DATATYPE y1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
	__SSE_DATATYPE q2 = _SSE_LOAD(&q[offset]);
	__SSE_DATATYPE y2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE q3 = _SSE_LOAD(&q[2*offset]);
	__SSE_DATATYPE y3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
	__SSE_DATATYPE q4 = _SSE_LOAD(&q[3*offset]);
	__SSE_DATATYPE y4 = _SSE_ADD(q4, _SSE_MUL(x4, h1));
#endif

	for(i = 2; i < nb; i++)
	{
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_loaddup_pd(&hh[i-1]);
		h2 = _mm_loaddup_pd(&hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[i-1]));
		h1 = _mm_moveldup_ps(x3);

		x4 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+i]));
                h2 = _mm_moveldup_ps(x4);
#endif


		q1 = _SSE_LOAD(&q[i*ldq]);
		x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
		y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
		q2 = _SSE_LOAD(&q[(i*ldq)+offset]);
		x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
		y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
#ifdef DOUBLE_PRECISION_REAL
		q3 = _SSE_LOAD(&q[(i*ldq)+2*offset]);
		x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
		y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
		q4 = _SSE_LOAD(&q[(i*ldq)+3*offset]);
		x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
		y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
#endif
	}
#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_loaddup_pd(&hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[nb-1]));
	h1 = _mm_moveldup_ps(x3);
#endif

	q1 = _SSE_LOAD(&q[nb*ldq]);
	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
	q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);
	x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
#ifdef DOUBLE_PRECISION_REAL
	q3 = _SSE_LOAD(&q[(nb*ldq)+2*offset]);
	x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
	q4 = _SSE_LOAD(&q[(nb*ldq)+3*offset]);
	x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
#endif

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [8 x nb+1]
	/////////////////////////////////////////////////////

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = _mm_loaddup_pd(hh);
	__SSE_DATATYPE tau2 = _mm_loaddup_pd(&hh[ldh]);
	__SSE_DATATYPE vs = _mm_loaddup_pd(&s);
#endif
#ifdef SINGLE_PRECISION_REAL
	x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) hh));
        __SSE_DATATYPE tau1 = _mm_moveldup_ps(x3);

	x4 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh]));
  	__SSE_DATATYPE tau2 = _mm_moveldup_ps(x4);
        __SSE_DATATYPE x5 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &s));
        __SSE_DATATYPE vs = _mm_moveldup_ps(x5);
#endif
	h1 = _SSE_XOR(tau1, sign);
	x1 = _SSE_MUL(x1, h1);
	x2 = _SSE_MUL(x2, h1);
#ifdef DOUBLE_PRECISION_REAL
	x3 = _SSE_MUL(x3, h1);
	x4 = _SSE_MUL(x4, h1);
#endif

	h1 = _SSE_XOR(tau2, sign);
	h2 = _SSE_MUL(h1, vs);

	y1 = _SSE_ADD(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
	y2 = _SSE_ADD(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
#ifdef DOUBLE_PRECISION_REAL
	y3 = _SSE_ADD(_SSE_MUL(y3,h1), _SSE_MUL(x3,h2));
	y4 = _SSE_ADD(_SSE_MUL(y4,h1), _SSE_MUL(x4,h2));
#endif
	q1 = _SSE_LOAD(q);
	q1 = _SSE_ADD(q1, y1);
	_SSE_STORE(q,q1);
	q2 = _SSE_LOAD(&q[offset]);
	q2 = _SSE_ADD(q2, y2);
	_SSE_STORE(&q[offset],q2);
#ifdef DOUBLE_PRECISION_REAL
	q3 = _SSE_LOAD(&q[2*offset]);
	q3 = _SSE_ADD(q3, y3);
	_SSE_STORE(&q[2*offset],q3);
	q4 = _SSE_LOAD(&q[3*offset]);
	q4 = _SSE_ADD(q4, y4);
	_SSE_STORE(&q[3*offset],q4);
#endif

#ifdef DOUBLE_PRECISION_REAL
	h2 = _mm_loaddup_pd(&hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
        x4 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+1]));
        h2 = _mm_moveldup_ps(x4);
#endif

	q1 = _SSE_LOAD(&q[ldq]);
	q1 = _SSE_ADD(q1, _SSE_ADD(x1, _SSE_MUL(y1, h2)));
	_SSE_STORE(&q[ldq],q1);
	q2 = _SSE_LOAD(&q[ldq+offset]);
	q2 = _SSE_ADD(q2, _SSE_ADD(x2, _SSE_MUL(y2, h2)));
	_SSE_STORE(&q[ldq+offset],q2);
#ifdef DOUBLE_PRECISION_REAL
	q3 = _SSE_LOAD(&q[ldq+2*offset]);
	q3 = _SSE_ADD(q3, _SSE_ADD(x3, _SSE_MUL(y3, h2)));
	_SSE_STORE(&q[ldq+2*offset],q3);
	q4 = _SSE_LOAD(&q[ldq+3*offset]);
	q4 = _SSE_ADD(q4, _SSE_ADD(x4, _SSE_MUL(y4, h2)));
	_SSE_STORE(&q[ldq+3*offset],q4);
#endif

	for (i = 2; i < nb; i++)
	{
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_loaddup_pd(&hh[i-1]);
		h2 = _mm_loaddup_pd(&hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
                x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[i-1]));
                h1 = _mm_moveldup_ps(x3);

		x4 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+i]));
                h2 = _mm_moveldup_ps(x4);
#endif

		q1 = _SSE_LOAD(&q[i*ldq]);
		q1 = _SSE_ADD(q1, _SSE_ADD(_SSE_MUL(x1,h1), _SSE_MUL(y1, h2)));
		_SSE_STORE(&q[i*ldq],q1);
		q2 = _SSE_LOAD(&q[(i*ldq)+offset]);
		q2 = _SSE_ADD(q2, _SSE_ADD(_SSE_MUL(x2,h1), _SSE_MUL(y2, h2)));
		_SSE_STORE(&q[(i*ldq)+offset],q2);
#ifdef DOUBLE_PRECISION_REAL
		q3 = _SSE_LOAD(&q[(i*ldq)+2*offset]);
		q3 = _SSE_ADD(q3, _SSE_ADD(_SSE_MUL(x3,h1), _SSE_MUL(y3, h2)));
		_SSE_STORE(&q[(i*ldq)+2*offset],q3);
		q4 = _SSE_LOAD(&q[(i*ldq)+3*offset]);
		q4 = _SSE_ADD(q4, _SSE_ADD(_SSE_MUL(x4,h1), _SSE_MUL(y4, h2)));
		_SSE_STORE(&q[(i*ldq)+3*offset],q4);
#endif
	}
#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_loaddup_pd(&hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[nb-1]));
        h1 = _mm_moveldup_ps(x3);
#endif

	q1 = _SSE_LOAD(&q[nb*ldq]);
	q1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
	_SSE_STORE(&q[nb*ldq],q1);
	q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);
	q2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
	_SSE_STORE(&q[(nb*ldq)+offset],q2);
#ifdef DOUBLE_PRECISION_REAL
	q3 = _SSE_LOAD(&q[(nb*ldq)+2*offset]);
	q3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
	_SSE_STORE(&q[(nb*ldq)+2*offset],q3);
	q4 = _SSE_LOAD(&q[(nb*ldq)+3*offset]);
	q4 = _SSE_ADD(q4, _SSE_MUL(x4, h1));
	_SSE_STORE(&q[(nb*ldq)+3*offset],q4);
#endif
}

/**
 * Unrolled kernel that computes
 * 4 rows of Q simultaneously, a
 * matrix Vector product with two householder
 * vectors + a rank 2 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_4_SSE_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_4_SSE_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif

	__SSE_DATATYPE x1 = _SSE_LOAD(&q[ldq]);
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE x2 = _SSE_LOAD(&q[ldq+offset]);
#endif

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h1 = _mm_loaddup_pd(&hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SSE_DATATYPE x2 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+1]));
	__SSE_DATATYPE x3 ;
        __SSE_DATATYPE h1 = _mm_moveldup_ps(x2);
#endif
	__SSE_DATATYPE h2;

	__SSE_DATATYPE q1 = _SSE_LOAD(q);
	__SSE_DATATYPE y1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE q2 = _SSE_LOAD(&q[offset]);
	__SSE_DATATYPE y2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
#endif

#ifdef SINGLE_PRECISION_REAL
       __SSE_DATATYPE q2;
#endif
	for(i = 2; i < nb; i++)
	{
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_loaddup_pd(&hh[i-1]);
		h2 = _mm_loaddup_pd(&hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		x2 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[i-1]));
                h1 = _mm_moveldup_ps(x2);

		x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+i]));
                h2 = _mm_moveldup_ps(x3);
#endif
		q1 = _SSE_LOAD(&q[i*ldq]);
		x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
		y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
#ifdef DOUBLE_PRECISION_REAL
		q2 = _SSE_LOAD(&q[(i*ldq)+offset]);
		x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
		y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
#endif
	}

#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_loaddup_pd(&hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	x2 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[nb-1]));
        h1 = _mm_moveldup_ps(x2);
#endif
	q1 = _SSE_LOAD(&q[nb*ldq]);
	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
#ifdef DOUBLE_PRECISION_REAL
	q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);
	x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
#endif
	/////////////////////////////////////////////////////
	// Rank-2 update of Q [12 x nb+1]
	/////////////////////////////////////////////////////

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = _mm_loaddup_pd(hh);
	__SSE_DATATYPE tau2 = _mm_loaddup_pd(&hh[ldh]);
	__SSE_DATATYPE vs = _mm_loaddup_pd(&s);
#endif
#ifdef SINGLE_PRECISION_REAL
	x2 = _mm_castpd_ps(_mm_loaddup_pd( (double *) hh));
        __SSE_DATATYPE tau1 = _mm_moveldup_ps(x2);

	x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh]));
        __SSE_DATATYPE tau2 = _mm_moveldup_ps(x3);

	__SSE_DATATYPE x4 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &s));
        __SSE_DATATYPE vs = _mm_moveldup_ps(x4);
#endif

	h1 = _SSE_XOR(tau1, sign);
	x1 = _SSE_MUL(x1, h1);
#ifdef DOUBLE_PRECISION_REAL
	x2 = _SSE_MUL(x2, h1);
#endif
	h1 = _SSE_XOR(tau2, sign);
	h2 = _SSE_MUL(h1, vs);

	y1 = _SSE_ADD(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
#ifdef DOUBLE_PRECISION_REAL
	y2 = _SSE_ADD(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
#endif
	q1 = _SSE_LOAD(q);
	q1 = _SSE_ADD(q1, y1);
	_SSE_STORE(q,q1);
#ifdef DOUBLE_PRECISION_REAL
	q2 = _SSE_LOAD(&q[offset]);
	q2 = _SSE_ADD(q2, y2);
	_SSE_STORE(&q[offset],q2);
#endif

#ifdef DOUBLE_PRECISION_REAL
	h2 = _mm_loaddup_pd(&hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	x2 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+1]));
        h2 = _mm_moveldup_ps(x2);
#endif
	q1 = _SSE_LOAD(&q[ldq]);
	q1 = _SSE_ADD(q1, _SSE_ADD(x1, _SSE_MUL(y1, h2)));
	_SSE_STORE(&q[ldq],q1);
#ifdef DOUBLE_PRECISION_REAL
	q2 = _SSE_LOAD(&q[ldq+offset]);
	q2 = _SSE_ADD(q2, _SSE_ADD(x2, _SSE_MUL(y2, h2)));
	_SSE_STORE(&q[ldq+offset],q2);
#endif

	for (i = 2; i < nb; i++)
	{
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_loaddup_pd(&hh[i-1]);
		h2 = _mm_loaddup_pd(&hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
                x2 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[i-1]));
                h1 = _mm_moveldup_ps(x2);

		x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+i]));
		h2 = _mm_moveldup_ps(x3);
#endif
		q1 = _SSE_LOAD(&q[i*ldq]);
		q1 = _SSE_ADD(q1, _SSE_ADD(_SSE_MUL(x1,h1), _SSE_MUL(y1, h2)));
		_SSE_STORE(&q[i*ldq],q1);
#ifdef DOUBLE_PRECISION_REAL
		q2 = _SSE_LOAD(&q[(i*ldq)+offset]);
		q2 = _SSE_ADD(q2, _SSE_ADD(_SSE_MUL(x2,h1), _SSE_MUL(y2, h2)));
		_SSE_STORE(&q[(i*ldq)+offset],q2);
#endif
	}

#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_loaddup_pd(&hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
        x2 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[nb-1]));
        h1 = _mm_moveldup_ps(x2);
#endif

	q1 = _SSE_LOAD(&q[nb*ldq]);
	q1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
	_SSE_STORE(&q[nb*ldq],q1);
#ifdef DOUBLE_PRECISION_REAL
	q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);
	q2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
	_SSE_STORE(&q[(nb*ldq)+offset],q2);
#endif
}
