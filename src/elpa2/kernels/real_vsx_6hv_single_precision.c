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
// Author: Andreas Marek, MPCDF

#include "config-f90.h"

#define REALCASE 1
#define SINGLE_PRECISION 1
#include "../../general/precision_macros.h"

#include "config-f90.h"

#ifdef HAVE_VSX_SSE
#include <altivec.h>
#endif
#include <stdio.h>
#include <stdlib.h>

#define __forceinline __attribute__((always_inline)) static

#ifdef DOUBLE_PRECISION_REAL
#define offset 2
#define __SSE_DATATYPE  __vector double
#define _SSE_LOAD (__vector double) vec_ld
#define _SSE_ADD vec_add
#define _SSE_SUB vec_sub
#define _SSE_MUL vec_mul
#define _SSE_STORE vec_st
#endif
#ifdef SINGLE_PRECISION_REAL
#define offset 4
#define __SSE_DATATYPE __vector float
#define _SSE_LOAD (__vector float) vec_ld
#define _SSE_ADD vec_add
#define _SSE_SUB vec_sub
#define _SSE_MUL vec_mul
#define _SSE_STORE vec_st
#endif

#ifdef HAVE_SSE_INTRINSICS
#undef __AVX__
#endif

#ifdef DOUBLE_PRECISION_REAL
//Forward declaration
static void hh_trafo_kernel_2_vsx_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);
static void hh_trafo_kernel_4_vsx_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);
void hexa_hh_trafo_real_vsx_6hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif

#ifdef SINGLE_PRECISION_REAL
static void hh_trafo_kernel_4_vsx_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);
static void hh_trafo_kernel_8_vsx_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);

void hexa_hh_trafo_real_vsx_6hv_single_(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif

#ifdef DOUBLE_PRECISION_REAL
void hexa_hh_trafo_real_vsx_6hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
#ifdef SINGLE_PRECISION_REAL
void hexa_hh_trafo_real_vsx_6hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;
	int worked_on ;

	worked_on = 0;

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
	for (i = 0; i < nq-2; i+=4)
	{
		hh_trafo_kernel_4_VSX_6hv_double(&q[i], hh, nb, ldq, ldh, scalarprods);
		worked_on += 4;
	}
#endif
#ifdef SINGLE_PRECISION_REAL
	for (i = 0; i < nq-4; i+=8)
	{
		hh_trafo_kernel_8_VSX_6hv_single(&q[i], hh, nb, ldq, ldh, scalarprods);
		worked_on += 8;
	}
#endif
	if (nq == i)
	{
		return;
	}
#ifdef DOUBLE_PRECISION_REAL
	if (nq -i == 2)
	{
		hh_trafo_kernel_2_VSX_6hv_double(&q[i], hh, nb, ldq, ldh, scalarprods);
		worked_on += 2;
	}
#endif
#ifdef SINGLE_PRECISION_REAL
	if (nq -i == 4)
	{
		hh_trafo_kernel_4_VSX_6hv_single(&q[i], hh, nb, ldq, ldh, scalarprods);
		worked_on += 4;
	}
#endif
#ifdef WITH_DEBUG
	if (worked_on != nq)
	{
		printf("Error in real SSE BLOCK6 kernel \n");
		abort();
	}
#endif
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
__forceinline void hh_trafo_kernel_4_VSX_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_8_VSX_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__SSE_DATATYPE a1_1 = _SSE_LOAD(0, &q[ldq*5]);
	__SSE_DATATYPE a2_1 = _SSE_LOAD(0, &q[ldq*4]);
	__SSE_DATATYPE a3_1 = _SSE_LOAD(0, &q[ldq*3]);
	__SSE_DATATYPE a4_1 = _SSE_LOAD(0, &q[ldq*2]);
	__SSE_DATATYPE a5_1 = _SSE_LOAD(0, &q[ldq]);
	__SSE_DATATYPE a6_1 = _SSE_LOAD(0, &q[0]);

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h_6_5 = vec_splats(hh[(ldh*5)+1]);
	__SSE_DATATYPE h_6_4 = vec_splats(hh[(ldh*5)+2]);
	__SSE_DATATYPE h_6_3 = vec_splats(hh[(ldh*5)+3]);
	__SSE_DATATYPE h_6_2 = vec_splats(hh[(ldh*5)+4]);
	__SSE_DATATYPE h_6_1 = vec_splats(hh[(ldh*5)+5]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h_6_5 =	 vec_splats(hh[(ldh*5)+1]) ;
	__SSE_DATATYPE h_6_4 =	 vec_splats(hh[(ldh*5)+2]) ;
	__SSE_DATATYPE h_6_3 =	 vec_splats(hh[(ldh*5)+3]) ;
	__SSE_DATATYPE h_6_2 =	 vec_splats(hh[(ldh*5)+4]) ;
	__SSE_DATATYPE h_6_1 =	 vec_splats(hh[(ldh*5)+5]) ;
#endif


	register __SSE_DATATYPE t1 = _SSE_ADD(a6_1, _SSE_MUL(a5_1, h_6_5));
	t1 = _SSE_ADD(t1, _SSE_MUL(a4_1, h_6_4));
	t1 = _SSE_ADD(t1, _SSE_MUL(a3_1, h_6_3));
	t1 = _SSE_ADD(t1, _SSE_MUL(a2_1, h_6_2));
	t1 = _SSE_ADD(t1, _SSE_MUL(a1_1, h_6_1));

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h_5_4 = vec_splats(hh[(ldh*4)+1]);
	__SSE_DATATYPE h_5_3 = vec_splats(hh[(ldh*4)+2]);
	__SSE_DATATYPE h_5_2 = vec_splats(hh[(ldh*4)+3]);
	__SSE_DATATYPE h_5_1 = vec_splats(hh[(ldh*4)+4]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h_5_4 =	 vec_splats(hh[(ldh*4)+1]) ;
	__SSE_DATATYPE h_5_3 =	 vec_splats(hh[(ldh*4)+2]) ;
	__SSE_DATATYPE h_5_2 =	 vec_splats(hh[(ldh*4)+3]) ;
	__SSE_DATATYPE h_5_1 =	 vec_splats(hh[(ldh*4)+4]) ;
#endif

	register __SSE_DATATYPE v1 = _SSE_ADD(a5_1, _SSE_MUL(a4_1, h_5_4));
	v1 = _SSE_ADD(v1, _SSE_MUL(a3_1, h_5_3));
	v1 = _SSE_ADD(v1, _SSE_MUL(a2_1, h_5_2));
	v1 = _SSE_ADD(v1, _SSE_MUL(a1_1, h_5_1));

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h_4_3 = vec_splats(hh[(ldh*3)+1]);
	__SSE_DATATYPE h_4_2 = vec_splats(hh[(ldh*3)+2]);
	__SSE_DATATYPE h_4_1 = vec_splats(hh[(ldh*3)+3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h_4_3 =	 vec_splats(hh[(ldh*3)+1]) ;
	__SSE_DATATYPE h_4_2 =	 vec_splats(hh[(ldh*3)+2]) ;
	__SSE_DATATYPE h_4_1 =	 vec_splats(hh[(ldh*3)+3]) ;
#endif

	register __SSE_DATATYPE w1 = _SSE_ADD(a4_1, _SSE_MUL(a3_1, h_4_3));
	w1 = _SSE_ADD(w1, _SSE_MUL(a2_1, h_4_2));
	w1 = _SSE_ADD(w1, _SSE_MUL(a1_1, h_4_1));

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h_2_1 = vec_splats(hh[ldh+1]);
	__SSE_DATATYPE h_3_2 = vec_splats(hh[(ldh*2)+1]);
	__SSE_DATATYPE h_3_1 = vec_splats(hh[(ldh*2)+2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h_2_1 =	 vec_splats(hh[ldh+1]) ;
	__SSE_DATATYPE h_3_2 =	 vec_splats(hh[(ldh*2)+1]) ;
	__SSE_DATATYPE h_3_1 =	 vec_splats(hh[(ldh*2)+2]) ;
#endif

	register __SSE_DATATYPE z1 = _SSE_ADD(a3_1, _SSE_MUL(a2_1, h_3_2));
	z1 = _SSE_ADD(z1, _SSE_MUL(a1_1, h_3_1));
	register __SSE_DATATYPE y1 = _SSE_ADD(a2_1, _SSE_MUL(a1_1, h_2_1));

	register __SSE_DATATYPE x1 = a1_1;

	__SSE_DATATYPE a1_2 = _SSE_LOAD(0, &q[(ldq*5)+offset]);
	__SSE_DATATYPE a2_2 = _SSE_LOAD(0, &q[(ldq*4)+offset]);
	__SSE_DATATYPE a3_2 = _SSE_LOAD(0, &q[(ldq*3)+offset]);
	__SSE_DATATYPE a4_2 = _SSE_LOAD(0, &q[(ldq*2)+offset]);
	__SSE_DATATYPE a5_2 = _SSE_LOAD(0, &q[(ldq)+offset]);
	__SSE_DATATYPE a6_2 = _SSE_LOAD(0, &q[offset]);

	register __SSE_DATATYPE t2 = _SSE_ADD(a6_2, _SSE_MUL(a5_2, h_6_5));
	t2 = _SSE_ADD(t2, _SSE_MUL(a4_2, h_6_4));
	t2 = _SSE_ADD(t2, _SSE_MUL(a3_2, h_6_3));
	t2 = _SSE_ADD(t2, _SSE_MUL(a2_2, h_6_2));
	t2 = _SSE_ADD(t2, _SSE_MUL(a1_2, h_6_1));
	register __SSE_DATATYPE v2 = _SSE_ADD(a5_2, _SSE_MUL(a4_2, h_5_4));
	v2 = _SSE_ADD(v2, _SSE_MUL(a3_2, h_5_3));
	v2 = _SSE_ADD(v2, _SSE_MUL(a2_2, h_5_2));
	v2 = _SSE_ADD(v2, _SSE_MUL(a1_2, h_5_1));
	register __SSE_DATATYPE w2 = _SSE_ADD(a4_2, _SSE_MUL(a3_2, h_4_3));
	w2 = _SSE_ADD(w2, _SSE_MUL(a2_2, h_4_2));
	w2 = _SSE_ADD(w2, _SSE_MUL(a1_2, h_4_1));
	register __SSE_DATATYPE z2 = _SSE_ADD(a3_2, _SSE_MUL(a2_2, h_3_2));
	z2 = _SSE_ADD(z2, _SSE_MUL(a1_2, h_3_1));
	register __SSE_DATATYPE y2 = _SSE_ADD(a2_2, _SSE_MUL(a1_2, h_2_1));

	register __SSE_DATATYPE x2 = a1_2;

	__SSE_DATATYPE q1;
	__SSE_DATATYPE q2;

	__SSE_DATATYPE h1;
	__SSE_DATATYPE h2;
	__SSE_DATATYPE h3;
	__SSE_DATATYPE h4;
	__SSE_DATATYPE h5;
	__SSE_DATATYPE h6;

	for(i = 6; i < nb; i++)
	{
#ifdef DOUBLE_PRECISION_REAL
		h1 = vec_splats(hh[i-5]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = vec_splats(hh[i-5] );
#endif
		q1 = _SSE_LOAD(0, &q[i*ldq]);
		q2 = _SSE_LOAD(0, &q[(i*ldq)+offset]);

		x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
		x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));

#ifdef DOUBLE_PRECISION_REAL
		h2 = vec_splats(hh[ldh+i-4]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h2 = vec_splats(hh[ldh+i-4] );
#endif
		y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
		y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));

#ifdef DOUBLE_PRECISION_REAL
		h3 = vec_splats(hh[(ldh*2)+i-3]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h3 = vec_splats(hh[(ldh*2)+i-3] );
#endif
		z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
		z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));

#ifdef DOUBLE_PRECISION_REAL
		h4 = vec_splats(hh[(ldh*3)+i-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h4 = vec_splats(hh[(ldh*3)+i-2] );
#endif
		w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
		w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));

#ifdef DOUBLE_PRECISION_REAL
		h5 = vec_splats(hh[(ldh*4)+i-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h5 = vec_splats(hh[(ldh*4)+i-1] );
#endif
		v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));
		v2 = _SSE_ADD(v2, _SSE_MUL(q2,h5));

#ifdef DOUBLE_PRECISION_REAL
		h6 = vec_splats(hh[(ldh*5)+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h6 = vec_splats(hh[(ldh*5)+i] );
#endif
		t1 = _SSE_ADD(t1, _SSE_MUL(q1,h6));
		t2 = _SSE_ADD(t2, _SSE_MUL(q2,h6));
	}

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-5]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-5] );
#endif
	q1 = _SSE_LOAD(0, &q[nb*ldq]);
	q2 = _SSE_LOAD(0, &q[(nb*ldq)+offset]);

	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
	x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-4]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-4] );
#endif
	y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
	y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));

#ifdef DOUBLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-3] );
#endif
	z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
	z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));

#ifdef DOUBLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+nb-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+nb-2] );
#endif
	w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
	w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));

#ifdef DOUBLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+nb-1] );
#endif

	v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));
	v2 = _SSE_ADD(v2, _SSE_MUL(q2,h5));

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-4]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-4] );
#endif
	q1 = _SSE_LOAD(0, &q[(nb+1)*ldq]);
	q2 = _SSE_LOAD(0, &q[((nb+1)*ldq)+offset]);

	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
	x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-3] );
#endif

	y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
	y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));

#ifdef DOUBLE_PRECISION
	h3 = vec_splats(hh[(ldh*2)+nb-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-2] );
#endif

	z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
	z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));

#ifdef DOUBLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+nb-1] );
#endif

	w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
	w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-3] );
#endif

	q1 = _SSE_LOAD(0, &q[(nb+2)*ldq]);
	q2 = _SSE_LOAD(0, &q[((nb+2)*ldq)+offset]);

	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
	x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-2] );
#endif

	y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
	y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));

#ifdef DOUBLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-1] );
#endif

	z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
	z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-2] );
#endif

	q1 = _SSE_LOAD(0, &q[(nb+3)*ldq]);
	q2 = _SSE_LOAD(0, &q[((nb+3)*ldq)+offset]);

	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
	x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-1] );
#endif


	y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
	y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1] );
#endif

	q1 = _SSE_LOAD(0, &q[(nb+4)*ldq]);
	q2 = _SSE_LOAD(0, &q[((nb+4)*ldq)+offset]);

	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
	x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));

	/////////////////////////////////////////////////////
	// Apply tau, correct wrong calculation using pre-calculated scalar products
	/////////////////////////////////////////////////////

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = vec_splats(hh[0]);
	x1 = _SSE_MUL(x1, tau1);
	x2 = _SSE_MUL(x2, tau1);

	__SSE_DATATYPE tau2 = vec_splats(hh[ldh]);
	__SSE_DATATYPE vs_1_2 = vec_splats(scalarprods[0]);
	h2 = _SSE_MUL(tau2, vs_1_2);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = vec_splats(hh[0] );
	x1 = _SSE_MUL(x1, tau1);
	x2 = _SSE_MUL(x2, tau1);

	__SSE_DATATYPE tau2 = vec_splats(hh[ldh] );
	__SSE_DATATYPE vs_1_2 = vec_splats(scalarprods[0] );
	h2 = _SSE_MUL(tau2, vs_1_2);
#endif

	y1 = _SSE_SUB(_SSE_MUL(y1,tau2), _SSE_MUL(x1,h2));
	y2 = _SSE_SUB(_SSE_MUL(y2,tau2), _SSE_MUL(x2,h2));

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau3 = vec_splats(hh[ldh*2]);
	__SSE_DATATYPE vs_1_3 = vec_splats(scalarprods[1]);
	__SSE_DATATYPE vs_2_3 = vec_splats(scalarprods[2]);
	h2 = _SSE_MUL(tau3, vs_1_3);
	h3 = _SSE_MUL(tau3, vs_2_3);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau3 = vec_splats(hh[ldh*2] );
	__SSE_DATATYPE vs_1_3 = vec_splats(scalarprods[1] );
	__SSE_DATATYPE vs_2_3 = vec_splats(scalarprods[2] );
	h2 = _SSE_MUL(tau3, vs_1_3);
	h3 = _SSE_MUL(tau3, vs_2_3);
#endif

	z1 = _SSE_SUB(_SSE_MUL(z1,tau3), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)));
	z2 = _SSE_SUB(_SSE_MUL(z2,tau3), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2)));

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau4 = vec_splats(hh[ldh*3]);
	__SSE_DATATYPE vs_1_4 = vec_splats(scalarprods[3]);
	__SSE_DATATYPE vs_2_4 = vec_splats(scalarprods[4]);
	h2 = _SSE_MUL(tau4, vs_1_4);
	h3 = _SSE_MUL(tau4, vs_2_4);
	__SSE_DATATYPE vs_3_4 = vec_splats(scalarprods[5]);
	h4 = _SSE_MUL(tau4, vs_3_4);
#endif
#ifdef SINGLE_PRECISION_REAL
       __SSE_DATATYPE tau4 = vec_splats(hh[ldh*3] );
	__SSE_DATATYPE vs_1_4 = vec_splats(scalarprods[3] );
	__SSE_DATATYPE vs_2_4 = vec_splats(scalarprods[4] );
	h2 = _SSE_MUL(tau4, vs_1_4);
	h3 = _SSE_MUL(tau4, vs_2_4);

	__SSE_DATATYPE vs_3_4 = vec_splats(scalarprods[5] );
	h4 = _SSE_MUL(tau4, vs_3_4);
#endif

	w1 = _SSE_SUB(_SSE_MUL(w1,tau4), _SSE_ADD(_SSE_MUL(z1,h4), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));
	w2 = _SSE_SUB(_SSE_MUL(w2,tau4), _SSE_ADD(_SSE_MUL(z2,h4), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2))));

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau5 = vec_splats(hh[ldh*4]);
	__SSE_DATATYPE vs_1_5 = vec_splats(scalarprods[6]);
	__SSE_DATATYPE vs_2_5 = vec_splats(scalarprods[7]);
	h2 = _SSE_MUL(tau5, vs_1_5);
	h3 = _SSE_MUL(tau5, vs_2_5);
	__SSE_DATATYPE vs_3_5 = vec_splats(scalarprods[8]);
	__SSE_DATATYPE vs_4_5 = vec_splats(scalarprods[9]);
	h4 = _SSE_MUL(tau5, vs_3_5);
	h5 = _SSE_MUL(tau5, vs_4_5);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau5 = vec_splats(hh[ldh*4] );
	__SSE_DATATYPE vs_1_5 = vec_splats(scalarprods[6] );
	__SSE_DATATYPE vs_2_5 = vec_splats(scalarprods[7] );
	h2 = _SSE_MUL(tau5, vs_1_5);
	h3 = _SSE_MUL(tau5, vs_2_5);

	__SSE_DATATYPE vs_3_5 = vec_splats(scalarprods[8] );
	__SSE_DATATYPE vs_4_5 = vec_splats(scalarprods[9] );

	h4 = _SSE_MUL(tau5, vs_3_5);
	h5 = _SSE_MUL(tau5, vs_4_5);
#endif

	v1 = _SSE_SUB(_SSE_MUL(v1,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));
	v2 = _SSE_SUB(_SSE_MUL(v2,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w2,h5), _SSE_MUL(z2,h4)), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2))));

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau6 = vec_splats(hh[ldh*5]);
	__SSE_DATATYPE vs_1_6 = vec_splats(scalarprods[10]);
	__SSE_DATATYPE vs_2_6 = vec_splats(scalarprods[11]);
	h2 = _SSE_MUL(tau6, vs_1_6);
	h3 = _SSE_MUL(tau6, vs_2_6);
	__SSE_DATATYPE vs_3_6 = vec_splats(scalarprods[12]);
	__SSE_DATATYPE vs_4_6 = vec_splats(scalarprods[13]);
	__SSE_DATATYPE vs_5_6 = vec_splats(scalarprods[14]);
	h4 = _SSE_MUL(tau6, vs_3_6);
	h5 = _SSE_MUL(tau6, vs_4_6);
	h6 = _SSE_MUL(tau6, vs_5_6);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau6 = vec_splats(hh[ldh*5] );
	__SSE_DATATYPE vs_1_6 = vec_splats(scalarprods[10] );
	__SSE_DATATYPE vs_2_6 = vec_splats(scalarprods[11] );

	h2 = _SSE_MUL(tau6, vs_1_6);
	h3 = _SSE_MUL(tau6, vs_2_6);

	__SSE_DATATYPE vs_3_6 = vec_splats(scalarprods[12] );
	__SSE_DATATYPE vs_4_6 = vec_splats(scalarprods[13] );
	__SSE_DATATYPE vs_5_6 = vec_splats(scalarprods[14] );

	h4 = _SSE_MUL(tau6, vs_3_6);
	h5 = _SSE_MUL(tau6, vs_4_6);
	h6 = _SSE_MUL(tau6, vs_5_6);
#endif

	t1 = _SSE_SUB(_SSE_MUL(t1,tau6), _SSE_ADD( _SSE_MUL(v1,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)))));
	t2 = _SSE_SUB(_SSE_MUL(t2,tau6), _SSE_ADD( _SSE_MUL(v2,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w2,h5), _SSE_MUL(z2,h4)), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2)))));

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [4 x nb+3]
	/////////////////////////////////////////////////////

	q1 = _SSE_LOAD(0, &q[0]);
	q2 = _SSE_LOAD(0, &q[offset]);
	q1 = _SSE_SUB(q1, t1);
	q2 = _SSE_SUB(q2, t2);
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[0]);
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[offset]);


#ifdef DOUBLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+1] );
#endif


	q1 = _SSE_LOAD(0, &q[ldq]);
	q2 = _SSE_LOAD(0, &q[(ldq+offset)]);
	q1 = _SSE_SUB(q1, v1);
	q2 = _SSE_SUB(q2, v2);

	q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
	q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));

	_SSE_STORE((__vector unsigned int), 0, (unsigned int *) &q[ldq]);
	_SSE_STORE((__vector unsigned int), 0, (unsigned int *) &q[(ldq+offset)]);

#ifdef DOUBLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+1] );
#endif
	q1 = _SSE_LOAD(0, &q[ldq*2]);
	q2 = _SSE_LOAD(0, &q[(ldq*2)+offset]);
	q1 = _SSE_SUB(q1, w1);
	q2 = _SSE_SUB(q2, w2);
	q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
	q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));

#ifdef DOUBLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+2] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
	q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));

	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[ldq*2]);
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[(ldq*2)+offset]);

#ifdef DOUBLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+1] );
#endif

	q1 = _SSE_LOAD(0, &q[ldq*3]);
	q2 = _SSE_LOAD(0, &q[(ldq*3)+offset]);
	q1 = _SSE_SUB(q1, z1);
	q2 = _SSE_SUB(q2, z2);

	q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
	q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));

#ifdef DOUBLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+2] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
	q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));

#ifdef DOUBLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+3] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
	q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));

	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[ldq*3]);
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[(ldq*3)+offset]);

#ifdef DOUBLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+1] );
#endif

	q1 = _SSE_LOAD(0, &q[ldq*4]);
	q2 = _SSE_LOAD(0, &q[(ldq*4)+offset]);
	q1 = _SSE_SUB(q1, y1);
	q2 = _SSE_SUB(q2, y2);

	q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
	q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));

#ifdef DOUBLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+2] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
	q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));

#ifdef DOUBLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+3] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
	q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));

#ifdef DOUBLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+4]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+4] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
	q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));

	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[ldq*4]);
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[(ldq*4)+offset]);

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[(ldh)+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[(ldh)+1] );
#endif

	q1 = _SSE_LOAD(0, &q[ldq*5]);
	q2 = _SSE_LOAD(0, &q[(ldq*5)+offset]);
	q1 = _SSE_SUB(q1, x1);
	q2 = _SSE_SUB(q2, x2);

	q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
	q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));

#ifdef DOUBLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+2] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
	q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));

#ifdef DOUBLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+3] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
	q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));

#ifdef DOUBLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+4]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+4] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
	q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));

#ifdef DOUBLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+5]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+5] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
	q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));

	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[ldq*5]);
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[(ldq*5)+offset]);

	for (i = 6; i < nb; i++)
	{
		q1 = _SSE_LOAD(0, &q[i*ldq]);
		q2 = _SSE_LOAD(0, &q[(i*ldq)+offset]);
#ifdef DOUBLE_PRECISION_REAL
		h1 = vec_splats(hh[i-5]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = vec_splats(hh[i-5] );
#endif

		q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
		q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));

#ifdef DOUBLE_PRECISION_REAL
		h2 = vec_splats(hh[ldh+i-4]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h2 = vec_splats(hh[ldh+i-4] );
#endif

		q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
		q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));

#ifdef DOUBLE_PRECISION_REAL
		h3 = vec_splats(hh[(ldh*2)+i-3]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h3 = vec_splats(hh[(ldh*2)+i-3] );
#endif

		q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
		q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));

#ifdef DOUBLE_PRECISION_REAL
		h4 = vec_splats(hh[(ldh*3)+i-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h4 = vec_splats(hh[(ldh*3)+i-2] );
#endif

		q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
		q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));

#ifdef DOUBLE_PRECISION_REAL
		h5 = vec_splats(hh[(ldh*4)+i-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h5 = vec_splats(hh[(ldh*4)+i-1] );
#endif

		q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
		q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));

#ifdef DOUBLE_PRECISION_REAL
		h6 = vec_splats(hh[(ldh*5)+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h6 = vec_splats(hh[(ldh*5)+i] );
#endif

		q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
		q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));

		_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[i*ldq]);
		_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[(i*ldq)+offset]);
	}

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-5]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-5] );
#endif

	q1 = _SSE_LOAD(0, &q[nb*ldq]);
	q2 = _SSE_LOAD(0, &q[(nb*ldq)+offset]);

	q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
	q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-4]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-4] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
	q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));

#ifdef DOUBLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-3] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
	q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));

#ifdef DOUBLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+nb-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+nb-2] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
	q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));

#ifdef DOUBLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+nb-1] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
	q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));

	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[nb*ldq]);
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[(nb*ldq)+offset]);

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-4]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-4] );
#endif

	q1 = _SSE_LOAD(0, &q[(nb+1)*ldq]);
	q2 = _SSE_LOAD(0, &q[((nb+1)*ldq)+offset]);

	q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
	q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-3] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
	q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));

#ifdef DOUBLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-2] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
	q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));

#ifdef DOUBLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+nb-1] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
	q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));

	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[(nb+1)*ldq]);
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[((nb+1)*ldq)+offset]);

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-3] );
#endif

	q1 = _SSE_LOAD(0, &q[(nb+2)*ldq]);
	q2 = _SSE_LOAD(0, &q[((nb+2)*ldq)+offset]);

	q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
	q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-2] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
	q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));

#ifdef DOUBLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-1] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
	q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));

	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[(nb+2)*ldq]);
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[((nb+2)*ldq)+offset]);

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-2] );
#endif

	q1 = _SSE_LOAD(0, &q[(nb+3)*ldq]);
	q2 = _SSE_LOAD(0, &q[((nb+3)*ldq)+offset]);

	q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
	q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-1] );
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
	q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));

	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[(nb+3)*ldq]);
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[((nb+3)*ldq)+offset]);

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1] );
#endif

	q1 = _SSE_LOAD(0, &q[(nb+4)*ldq]);
	q2 = _SSE_LOAD(0, &q[((nb+4)*ldq)+offset]);

	q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
	q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));

	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[(nb+4)*ldq]);
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[((nb+4)*ldq)+offset]);
}
/**
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 2 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 4 rows of Q simultaneously, a
#endif
 * matrix Vector product with two householder
 * vectors + a rank 1 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_2_SSE_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_4_SSE_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [2 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__SSE_DATATYPE a1_1 = _SSE_LOAD(0,  &q[ldq*5]);
	__SSE_DATATYPE a2_1 = _SSE_LOAD(0,  &q[ldq*4]);
	__SSE_DATATYPE a3_1 = _SSE_LOAD(0,  &q[ldq*3]);
	__SSE_DATATYPE a4_1 = _SSE_LOAD(0,  &q[ldq*2]);
	__SSE_DATATYPE a5_1 = _SSE_LOAD(0,  &q[ldq]);
	__SSE_DATATYPE a6_1 = _SSE_LOAD(0,  &q[0]);

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h_6_5 = vec_splats(hh[(ldh*5)+1]);
	__SSE_DATATYPE h_6_4 = vec_splats(hh[(ldh*5)+2]);
	__SSE_DATATYPE h_6_3 = vec_splats(hh[(ldh*5)+3]);
	__SSE_DATATYPE h_6_2 = vec_splats(hh[(ldh*5)+4]);
	__SSE_DATATYPE h_6_1 = vec_splats(hh[(ldh*5)+5]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h_6_5 = vec_splats(hh[(ldh*5)+1]) ;
	__SSE_DATATYPE h_6_4 = vec_splats(hh[(ldh*5)+2]) ;
	__SSE_DATATYPE h_6_3 = vec_splats(hh[(ldh*5)+3]) ;
	__SSE_DATATYPE h_6_2 = vec_splats(hh[(ldh*5)+4]) ;
	__SSE_DATATYPE h_6_1 = vec_splats(hh[(ldh*5)+5]) ;
#endif

	register __SSE_DATATYPE t1 = _SSE_ADD(a6_1, _SSE_MUL(a5_1, h_6_5));
	t1 = _SSE_ADD(t1, _SSE_MUL(a4_1, h_6_4));
	t1 = _SSE_ADD(t1, _SSE_MUL(a3_1, h_6_3));
	t1 = _SSE_ADD(t1, _SSE_MUL(a2_1, h_6_2));
	t1 = _SSE_ADD(t1, _SSE_MUL(a1_1, h_6_1));

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h_5_4 = vec_splats(hh[(ldh*4)+1]);
	__SSE_DATATYPE h_5_3 = vec_splats(hh[(ldh*4)+2]);
	__SSE_DATATYPE h_5_2 = vec_splats(hh[(ldh*4)+3]);
	__SSE_DATATYPE h_5_1 = vec_splats(hh[(ldh*4)+4]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h_5_4 = vec_splats(hh[(ldh*4)+1]) ;
	__SSE_DATATYPE h_5_3 = vec_splats(hh[(ldh*4)+2]) ;
	__SSE_DATATYPE h_5_2 = vec_splats(hh[(ldh*4)+3]) ;
	__SSE_DATATYPE h_5_1 = vec_splats(hh[(ldh*4)+4]) ;
#endif

	register __SSE_DATATYPE v1 = _SSE_ADD(a5_1, _SSE_MUL(a4_1, h_5_4));
	v1 = _SSE_ADD(v1, _SSE_MUL(a3_1, h_5_3));
	v1 = _SSE_ADD(v1, _SSE_MUL(a2_1, h_5_2));
	v1 = _SSE_ADD(v1, _SSE_MUL(a1_1, h_5_1));

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h_4_3 = vec_splats(hh[(ldh*3)+1]);
	__SSE_DATATYPE h_4_2 = vec_splats(hh[(ldh*3)+2]);
	__SSE_DATATYPE h_4_1 = vec_splats(hh[(ldh*3)+3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h_4_3 = vec_splats(hh[(ldh*3)+1]) ;
	__SSE_DATATYPE h_4_2 = vec_splats(hh[(ldh*3)+2]) ;
	__SSE_DATATYPE h_4_1 = vec_splats(hh[(ldh*3)+3]) ;
#endif
	register __SSE_DATATYPE w1 = _SSE_ADD(a4_1, _SSE_MUL(a3_1, h_4_3));
	w1 = _SSE_ADD(w1, _SSE_MUL(a2_1, h_4_2));
	w1 = _SSE_ADD(w1, _SSE_MUL(a1_1, h_4_1));

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h_2_1 = vec_splats(hh[ldh+1]);
	__SSE_DATATYPE h_3_2 = vec_splats(hh[(ldh*2)+1]);
	__SSE_DATATYPE h_3_1 = vec_splats(hh[(ldh*2)+2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h_2_1 = vec_splats(hh[ldh+1]) ;
	__SSE_DATATYPE h_3_2 = vec_splats(hh[(ldh*2)+1]) ;
	__SSE_DATATYPE h_3_1 = vec_splats(hh[(ldh*2)+2]) ;
#endif

	register __SSE_DATATYPE z1 = _SSE_ADD(a3_1, _SSE_MUL(a2_1, h_3_2));
	z1 = _SSE_ADD(z1, _SSE_MUL(a1_1, h_3_1));
	register __SSE_DATATYPE y1 = _SSE_ADD(a2_1, _SSE_MUL(a1_1, h_2_1));

	register __SSE_DATATYPE x1 = a1_1;

	__SSE_DATATYPE q1;

	__SSE_DATATYPE h1;
	__SSE_DATATYPE h2;
	__SSE_DATATYPE h3;
	__SSE_DATATYPE h4;
	__SSE_DATATYPE h5;
	__SSE_DATATYPE h6;

	for(i = 6; i < nb; i++)
	{
#ifdef DOUBLE_PRECISION_REAL
		h1 = vec_splats(hh[i-5]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = vec_splats(hh[i-5]) ;
#endif
		q1 = _SSE_LOAD(0,  &q[i*ldq]);

		x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));

#ifdef DOUBLE_PRECISION_REAL
		h2 = vec_splats(hh[ldh+i-4]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h2 = vec_splats(hh[ldh+i-4]) ;
#endif

		y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));

#ifdef DOUBLE_PRECISION_REAL
		h3 = vec_splats(hh[(ldh*2)+i-3]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h3 = vec_splats(hh[(ldh*2)+i-3]) ;
#endif

		z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));

#ifdef DOUBLE_PRECISION_REAL
		h4 = vec_splats(hh[(ldh*3)+i-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h4 = vec_splats(hh[(ldh*3)+i-2]) ;
#endif

		w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));

#ifdef DOUBLE_PRECISION_REAL
		h5 = vec_splats(hh[(ldh*4)+i-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h5 = vec_splats(hh[(ldh*4)+i-1]) ;
#endif

		v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));

#ifdef DOUBLE_PRECISION_REAL
		h6 = vec_splats(hh[(ldh*5)+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h6 = vec_splats(hh[(ldh*5)+i]) ;
#endif

		t1 = _SSE_ADD(t1, _SSE_MUL(q1,h6));

	}

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-5]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-5]) ;
#endif

	q1 = _SSE_LOAD(0,  &q[nb*ldq]);

	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-4]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-4]) ;
#endif

	y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));

#ifdef DOUBLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-3]) ;
#endif

	z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));

#ifdef DOUBLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+nb-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+nb-2]) ;
#endif


	w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));

#ifdef DOUBLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+nb-1]) ;
#endif

	v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-4]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-4]) ;
#endif

	q1 = _SSE_LOAD(0,  &q[(nb+1)*ldq]);

	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-3]) ;
#endif

	y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));

#ifdef DOUBLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-2]) ;
#endif

	z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));

#ifdef DOUBLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+nb-1]) ;
#endif

	w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-3]) ;
#endif

	q1 = _SSE_LOAD(0,  &q[(nb+2)*ldq]);

	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-2]) ;
#endif

	y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));

#ifdef DOUBLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-1]) ;
#endif

	z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-2]) ;
#endif

	q1 = _SSE_LOAD(0,  &q[(nb+3)*ldq]);

	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-1]) ;
#endif

	y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]) ;
#endif

	q1 = _SSE_LOAD(0,  &q[(nb+4)*ldq]);

	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));

	/////////////////////////////////////////////////////
	// Apply tau, correct wrong calculation using pre-calculated scalar products
	/////////////////////////////////////////////////////

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = vec_splats(hh[0]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = vec_splats(hh[0]) ;
#endif
	x1 = _SSE_MUL(x1, tau1);

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau2 = vec_splats(hh[ldh]);
	__SSE_DATATYPE vs_1_2 = vec_splats(scalarprods[0]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau2 = vec_splats(hh[ldh]) ;
	__SSE_DATATYPE vs_1_2 = vec_splats(scalarprods[0]) ;
#endif

	h2 = _SSE_MUL(tau2, vs_1_2);

	y1 = _SSE_SUB(_SSE_MUL(y1,tau2), _SSE_MUL(x1,h2));

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau3 = vec_splats(hh[ldh*2]);
	__SSE_DATATYPE vs_1_3 = vec_splats(scalarprods[1]);
	__SSE_DATATYPE vs_2_3 = vec_splats(scalarprods[2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau3 = vec_splats(hh[ldh*2]) ;
	__SSE_DATATYPE vs_1_3 = vec_splats(scalarprods[1]) ;
	__SSE_DATATYPE vs_2_3 = vec_splats(scalarprods[2]) ;
#endif

	h2 = _SSE_MUL(tau3, vs_1_3);
	h3 = _SSE_MUL(tau3, vs_2_3);

	z1 = _SSE_SUB(_SSE_MUL(z1,tau3), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)));

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau4 = vec_splats(hh[ldh*3]);
	__SSE_DATATYPE vs_1_4 = vec_splats(scalarprods[3]);
	__SSE_DATATYPE vs_2_4 = vec_splats(scalarprods[4]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau4 = vec_splats(hh[ldh*3]) ;
	__SSE_DATATYPE vs_1_4 = vec_splats(scalarprods[3]) ;
	__SSE_DATATYPE vs_2_4 = vec_splats(scalarprods[4]) ;
#endif
	h2 = _SSE_MUL(tau4, vs_1_4);
	h3 = _SSE_MUL(tau4, vs_2_4);

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE vs_3_4 = vec_splats(scalarprods[5]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE vs_3_4 = vec_splats(scalarprods[5]) ;
#endif

	h4 = _SSE_MUL(tau4, vs_3_4);

	w1 = _SSE_SUB(_SSE_MUL(w1,tau4), _SSE_ADD(_SSE_MUL(z1,h4), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau5 = vec_splats(hh[ldh*4]);
	__SSE_DATATYPE vs_1_5 = vec_splats(scalarprods[6]);
	__SSE_DATATYPE vs_2_5 = vec_splats(scalarprods[7]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau5 = vec_splats(hh[ldh*4]) ;
	__SSE_DATATYPE vs_1_5 = vec_splats(scalarprods[6]) ;
	__SSE_DATATYPE vs_2_5 = vec_splats(scalarprods[7]) ;
#endif
	h2 = _SSE_MUL(tau5, vs_1_5);
	h3 = _SSE_MUL(tau5, vs_2_5);

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE vs_3_5 = vec_splats(scalarprods[8]);
	__SSE_DATATYPE vs_4_5 = vec_splats(scalarprods[9]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE vs_3_5 = vec_splats(scalarprods[8]) ;
	__SSE_DATATYPE vs_4_5 = vec_splats(scalarprods[9]) ;
#endif
	h4 = _SSE_MUL(tau5, vs_3_5);
	h5 = _SSE_MUL(tau5, vs_4_5);

	v1 = _SSE_SUB(_SSE_MUL(v1,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau6 = vec_splats(hh[ldh*5]);
	__SSE_DATATYPE vs_1_6 = vec_splats(scalarprods[10]);
	__SSE_DATATYPE vs_2_6 = vec_splats(scalarprods[11]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau6 = vec_splats(hh[ldh*5]) ;
	__SSE_DATATYPE vs_1_6 = vec_splats(scalarprods[10]) ;
	__SSE_DATATYPE vs_2_6 = vec_splats(scalarprods[11]) ;
#endif
	h2 = _SSE_MUL(tau6, vs_1_6);
	h3 = _SSE_MUL(tau6, vs_2_6);

#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE vs_3_6 = vec_splats(scalarprods[12]);
	__SSE_DATATYPE vs_4_6 = vec_splats(scalarprods[13]);
	__SSE_DATATYPE vs_5_6 = vec_splats(scalarprods[14]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE vs_3_6 = vec_splats(scalarprods[12]) ;
	__SSE_DATATYPE vs_4_6 = vec_splats(scalarprods[13]) ;
	__SSE_DATATYPE vs_5_6 = vec_splats(scalarprods[14]) ;
#endif

	h4 = _SSE_MUL(tau6, vs_3_6);
	h5 = _SSE_MUL(tau6, vs_4_6);
	h6 = _SSE_MUL(tau6, vs_5_6);

	t1 = _SSE_SUB(_SSE_MUL(t1,tau6), _SSE_ADD( _SSE_MUL(v1,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)))));

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [2 x nb+3]
	/////////////////////////////////////////////////////

	q1 = _SSE_LOAD(0,  &q[0]);
	q1 = _SSE_SUB(q1, t1);
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[0]);

#ifdef DOUBLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+1]) ;
#endif

	q1 = _SSE_LOAD(0,  &q[ldq]);
	q1 = _SSE_SUB(q1, v1);

	q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));

	_SSE_STORE((__vector unsigned int), q1, 0, (unsigned int *) &q[ldq]);

#ifdef DOUBLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+1]) ;
#endif
	q1 = _SSE_LOAD(0,  &q[ldq*2]);
	q1 = _SSE_SUB(q1, w1);

	q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));

#ifdef DOUBLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+2]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));

	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[ldq*2]);

#ifdef DOUBLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+1]) ;
#endif

	q1 = _SSE_LOAD(0,  &q[ldq*3]);
	q1 = _SSE_SUB(q1, z1);

	q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));

#ifdef DOUBLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+2]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));

#ifdef DOUBLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+3]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));

	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[ldq*3]);

#ifdef DOUBLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+1]) ;
#endif

	q1 = _SSE_LOAD(0,  &q[ldq*4]);
	q1 = _SSE_SUB(q1, y1);

	q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));

#ifdef DOUBLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+2]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));

#ifdef DOUBLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+3]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));

#ifdef DOUBLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+4]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+4]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));

	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[ldq*4]);

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[(ldh)+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[(ldh)+1]) ;
#endif

	q1 = _SSE_LOAD(0,  &q[ldq*5]);
	q1 = _SSE_SUB(q1, x1);

	q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));

#ifdef DOUBLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+2]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));

#ifdef DOUBLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+3]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));

#ifdef DOUBLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+4]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+4]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));

#ifdef DOUBLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+5]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h6 = vec_splats(hh[(ldh*5)+5]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));

	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[ldq*5]);

	for (i = 6; i < nb; i++)
	{
		q1 = _SSE_LOAD(0,  &q[i*ldq]);
#ifdef DOUBLE_PRECISION_REAL
		h1 = vec_splats(hh[i-5]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = vec_splats(hh[i-5]) ;
#endif

		q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));

#ifdef DOUBLE_PRECISION_REAL
		h2 = vec_splats(hh[ldh+i-4]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h2 = vec_splats(hh[ldh+i-4]) ;
#endif

		q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));

#ifdef DOUBLE_PRECISION_REAL
		h3 = vec_splats(hh[(ldh*2)+i-3]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h3 = vec_splats(hh[(ldh*2)+i-3]) ;
#endif

		q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));

#ifdef DOUBLE_PRECISION_REAL
		h4 = vec_splats(hh[(ldh*3)+i-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h4 = vec_splats(hh[(ldh*3)+i-2]) ;
#endif

		q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));

#ifdef DOUBLE_PRECISION_REAL
		h5 = vec_splats(hh[(ldh*4)+i-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h5 = vec_splats(hh[(ldh*4)+i-1]) ;
#endif

		q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));

#ifdef DOUBLE_PRECISION_REAL
		h6 = vec_splats(hh[(ldh*5)+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h6 = vec_splats(hh[(ldh*5)+i]) ;
#endif

		q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));

		_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[i*ldq]);
	}

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-5]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-5]) ;
#endif

	q1 = _SSE_LOAD(0,  &q[nb*ldq]);

	q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-4]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-4]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));

#ifdef DOUBLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-3]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));

#ifdef DOUBLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+nb-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+nb-2]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));

#ifdef DOUBLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h5 = vec_splats(hh[(ldh*4)+nb-1]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));

	_SSE_STORE((__vector unsigned int) q1, 0,  (unsigned int *) &q[nb*ldq]);

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-4]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-4]) ;
#endif

	q1 = _SSE_LOAD(0,  &q[(nb+1)*ldq]);

	q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-3]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));

#ifdef DOUBLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-2]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));

#ifdef DOUBLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h4 = vec_splats(hh[(ldh*3)+nb-1]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));

	_SSE_STORE((__vector unsigned int), q1, 0, (unsigned int *)&q[(nb+1)*ldq]);

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-3]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-3]) ;
#endif

	q1 = _SSE_LOAD(0,  &q[(nb+2)*ldq]);

	q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-2]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));

#ifdef DOUBLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h3 = vec_splats(hh[(ldh*2)+nb-1]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));

	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[(nb+2)*ldq]);

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-2]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-2]) ;
#endif

	q1 = _SSE_LOAD(0,  &q[(nb+3)*ldq]);

	q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));

#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+nb-1]) ;
#endif

	q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));

	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[(nb+3)*ldq]);

#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]) ;
#endif

	q1 = _SSE_LOAD(0,  &q[(nb+4)*ldq]);

	q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));

	_SSE_STORE((__vector unsigned int) q1, 0,  (unsigned int *) &q[(nb+4)*ldq]);
}

#undef REALCASE
#undef SINGLE_PRECISION

