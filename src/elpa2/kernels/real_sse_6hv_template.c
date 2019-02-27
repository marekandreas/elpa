//    This file is part of ELPA.
//
//    The ELPA library was originally created by the ELPA consortium,
//    consisting of the following organizations:
//
//    - Max Planck Computing and Data Facility (MPCDF), formerly known as
//        Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
//    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
//        Informatik,
//    - Technische Universität München, Lehrstuhl für Informatik mit
//        Schwerpunkt Wissenschaftliches Rechnen ,
//    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
//    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
//        Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
//        and
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
//    along with ELPA.        If not, see <http://www.gnu.org/licenses/>
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
// Author: Andreas Marek, MPCDF (andreas.marek@mpcdf.mpg.de), based on Alexander Heinecke (alexander.heinecke@mytum.de)
// --------------------------------------------------------------------------------------------------

#include "config-f90.h"

#ifdef HAVE_SSE_INTRINSICS
#include <x86intrin.h>
#endif
#ifdef HAVE_SPARC64_SSE
#include <fjmfunc.h>
#include <emmintrin.h>
#endif
#include <stdio.h>
#include <stdlib.h>


#define __forceinline __attribute__((always_inline)) static

#ifdef DOUBLE_PRECISION_REAL
#define offset 2
#define __SSE_DATATYPE __m128d
#define _SSE_LOAD _mm_load_pd
#define _SSE_ADD _mm_add_pd
#define _SSE_SUB _mm_sub_pd
#define _SSE_MUL _mm_mul_pd
#define _SSE_STORE _mm_store_pd
#define _SSE_SET _mm_set_pd
#define _SSE_SET1 _mm_set1_pd
#endif
#ifdef SINGLE_PRECISION_REAL
#define offset 4
#define __SSE_DATATYPE __m128
#define _SSE_LOAD _mm_load_ps
#define _SSE_ADD _mm_add_ps
#define _SSE_SUB _mm_sub_ps
#define _SSE_MUL _mm_mul_ps
#define _SSE_STORE _mm_store_ps
#define _SSE_SET _mm_set_ps
#define _SSE_SET1 _mm_set1_ps
#endif

#ifdef HAVE_SSE_INTRINSICS
#undef __AVX__
#endif

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
//Forward declaration
static void hh_trafo_kernel_2_SSE_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);
static void hh_trafo_kernel_4_SSE_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);
void hexa_hh_trafo_real_sse_6hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif

#ifdef SINGLE_PRECISION_REAL
static void hh_trafo_kernel_4_SSE_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);
static void hh_trafo_kernel_8_SSE_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);
void hexa_hh_trafo_real_sse_6hv_single_(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif
#endif

#ifdef HAVE_SPARC64_SSE
#ifdef DOUBLE_PRECISION_REAL
//Forward declaration
static void hh_trafo_kernel_2_SPARC64_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);
static void hh_trafo_kernel_4_SPARC64_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);
void hexa_hh_trafo_real_sparc64_6hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif

#ifdef SINGLE_PRECISION_REAL
static void hh_trafo_kernel_4_SPARC64_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);
static void hh_trafo_kernel_8_SPARC64_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);
void hexa_hh_trafo_real_sparc64_6hv_single_(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif
#endif



#ifdef DOUBLE_PRECISION_REAL
/*
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine hexa_hh_trafo_real_sse_6hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="hexa_hh_trafo_real_sse_6hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_double)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#ifdef HAVE_SPARC64_SSE
!f> interface
!f>   subroutine hexa_hh_trafo_real_sparc64_6hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="hexa_hh_trafo_real_sparc64_6hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_double)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
#endif

#ifdef SINGLE_PRECISION_REAL
/*
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine hexa_hh_trafo_real_sse_6hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="hexa_hh_trafo_real_sse_6hv_single")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_float)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#ifdef HAVE_SPARC64_SSE
!f> interface
!f>   subroutine hexa_hh_trafo_real_sparc64_6hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="hexa_hh_trafo_real_sparc64_6hv_single")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_float)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

#endif

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
void hexa_hh_trafo_real_sse_6hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
#ifdef SINGLE_PRECISION_REAL
void hexa_hh_trafo_real_sse_6hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
#endif
#ifdef HAVE_SPARC64_SSE
#ifdef DOUBLE_PRECISION_REAL
void hexa_hh_trafo_real_sparc64_6hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
#ifdef SINGLE_PRECISION_REAL
void hexa_hh_trafo_real_sparc64_6hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
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

        scalarprods[0] = hh[(ldh+1)];      // 1 = hh(2,2)
        scalarprods[1] = hh[(ldh*2)+2];    // 2 = hh(3,3)
        scalarprods[2] = hh[(ldh*2)+1];    // 3 = hh(2,3)
        scalarprods[3] = hh[(ldh*3)+3];    // 4 = hh(4,4)
        scalarprods[4] = hh[(ldh*3)+2];    // 5 = hh(3,4)
        scalarprods[5] = hh[(ldh*3)+1];    // 6 = hh(2,4)
        scalarprods[6] = hh[(ldh*4)+4];    // 7 = hh(5,5)
        scalarprods[7] = hh[(ldh*4)+3];    // 8 = hh(4,5)
        scalarprods[8] = hh[(ldh*4)+2];    // 9 = hh(3,5)
        scalarprods[9] = hh[(ldh*4)+1];    //10 = hh(2,5) 
        scalarprods[10] = hh[(ldh*5)+5];   //11 = hh(6,6) 
        scalarprods[11] = hh[(ldh*5)+4];   //12 = hh(5,6)
        scalarprods[12] = hh[(ldh*5)+3];   //13 = hh(4,6)
        scalarprods[13] = hh[(ldh*5)+2];   //14 = hh(3,6)
        scalarprods[14] = hh[(ldh*5)+1];   //15 = hh(2,6)

        // calculate scalar product of first and fourth householder Vector
        // loop counter = 2
        scalarprods[0] += hh[1] * hh[(2+ldh)];             // 1 = 1 + hh(2,1) * hh(3,2)
        scalarprods[2] += hh[(ldh)+1] * hh[2+(ldh*2)];     // 3 = 3 + hh(2,2) * hh(3,3)
        scalarprods[5] += hh[(ldh*2)+1] * hh[2+(ldh*3)];   // 6 = 6 + hh(2,3) * hh(3,4)
        scalarprods[9] += hh[(ldh*3)+1] * hh[2+(ldh*4)];   //10 =10 + hh(2,4) * hh(3,5) 
        scalarprods[14] += hh[(ldh*4)+1] * hh[2+(ldh*5)];  //15 =15 + hh(2,5) * hh(3,6)

        // loop counter = 3
        scalarprods[0] += hh[2] * hh[(3+ldh)];             // 1 = 1 + hh(3,1) * hh(4,2)
        scalarprods[2] += hh[(ldh)+2] * hh[3+(ldh*2)];     // 3 = 3 + hh(3,2) * hh(4,3)
        scalarprods[5] += hh[(ldh*2)+2] * hh[3+(ldh*3)];   // 6 = 6 + hh(3,3) * hh(4,4)
        scalarprods[9] += hh[(ldh*3)+2] * hh[3+(ldh*4)];   //10 =10 + hh(3,4) * hh(4,5)
        scalarprods[14] += hh[(ldh*4)+2] * hh[3+(ldh*5)];  //15 =15 + hh(3,5) * hh(4,6)

        scalarprods[1] += hh[1] * hh[3+(ldh*2)];           // 2 = 2 + hh(2,1) * hh(4,3)
        scalarprods[4] += hh[(ldh*1)+1] * hh[3+(ldh*3)];   // 5 = 5 + hh(2,2) * hh(4,4)
        scalarprods[8] += hh[(ldh*2)+1] * hh[3+(ldh*4)];   // 9 = 9 + hh(2,3) * hh(4,5)
        scalarprods[13] += hh[(ldh*3)+1] * hh[3+(ldh*5)];  //14 =14 + hh(2,4) * hh(4,6)

        // loop counter = 4
        scalarprods[0] += hh[3] * hh[(4+ldh)];            // 1 = 1 + hh(4,1) * hh(5,2)
        scalarprods[2] += hh[(ldh)+3] * hh[4+(ldh*2)];    // 3 = 3 + hh(4,2) * hh(5,3)
        scalarprods[5] += hh[(ldh*2)+3] * hh[4+(ldh*3)];  // 6 = 6 + hh(4,3) * hh(5,4)
        scalarprods[9] += hh[(ldh*3)+3] * hh[4+(ldh*4)];  //10 =10 + hh(4,4) * hh(5,5)
        scalarprods[14] += hh[(ldh*4)+3] * hh[4+(ldh*5)]; //15 =15 + hh(4,5) * hh(5,6)

        scalarprods[1] += hh[2] * hh[4+(ldh*2)];          // 2 = 2 + hh(3,1) * hh(5,3)
        scalarprods[4] += hh[(ldh*1)+2] * hh[4+(ldh*3)];  // 5 = 5 + hh(3,2) * hh(5,4)
        scalarprods[8] += hh[(ldh*2)+2] * hh[4+(ldh*4)];  // 9 = 9 + hh(3,3) * hh(5,5)
        scalarprods[13] += hh[(ldh*3)+2] * hh[4+(ldh*5)]; //14 =14 + hh(3,4) * hh(5,6)

        scalarprods[3] += hh[1] * hh[4+(ldh*3)];          // 4 = 4 + hh(2,1) * hh(5,4)
        scalarprods[7] += hh[(ldh)+1] * hh[4+(ldh*4)];    // 8 = 8 + hh(2,2) * hh(5,5)
        scalarprods[12] += hh[(ldh*2)+1] * hh[4+(ldh*5)]; //13 =13 + hh(2,3) * hh(5,6)

        // loop counter = 5
        scalarprods[0] += hh[4] * hh[(5+ldh)];            // 1 = 1 + hh(5,1) * hh(6,2)
        scalarprods[2] += hh[(ldh)+4] * hh[5+(ldh*2)];    // 3 = 3 + hh(5,2) * hh(6,3)
        scalarprods[5] += hh[(ldh*2)+4] * hh[5+(ldh*3)];  // 6 = 6 + hh(5,3) * hh(6,4)
        scalarprods[9] += hh[(ldh*3)+4] * hh[5+(ldh*4)];  //10 =10 + hh(5,4) * hh(6,5)
        scalarprods[14] += hh[(ldh*4)+4] * hh[5+(ldh*5)]; //15 =15 + hh(5,5) * hh(6,6)

        scalarprods[1] += hh[3] * hh[5+(ldh*2)];          // 2 = 2 + hh(4,1) * hh(6,3)
        scalarprods[4] += hh[(ldh*1)+3] * hh[5+(ldh*3)];  // 5 = 5 + hh(4,2) * hh(6,4)
        scalarprods[8] += hh[(ldh*2)+3] * hh[5+(ldh*4)];  // 9 = 9 + hh(4,3) * hh(6,5)
        scalarprods[13] += hh[(ldh*3)+3] * hh[5+(ldh*5)]; //14 =14 + hh(4,4) * hh(6,6)

        scalarprods[3] += hh[2] * hh[5+(ldh*3)];          // 4 = 4 + hh(3,1) * hh(6,4)
        scalarprods[7] += hh[(ldh)+2] * hh[5+(ldh*4)];    // 8 = 8 + hh(3,2) * hh(6,5)
        scalarprods[12] += hh[(ldh*2)+2] * hh[5+(ldh*5)]; //13 =13 + hh(3,3) * hh(6,6)

        scalarprods[6] += hh[1] * hh[5+(ldh*4)];          // 7 = 7 + hh(2,1) * hh(6,5)
        scalarprods[11] += hh[(ldh)+1] * hh[5+(ldh*5)];   //12 =12 + hh(2,2) * hh(6,6) 

        #pragma ivdep
        for (i = 6; i < nb; i++)                                     // do i = 7, nb
        {
                scalarprods[0] += hh[i-1] * hh[(i+ldh)];             // 1 = 1 + hh(i-1,1) * hh(i,2)
                scalarprods[2] += hh[(ldh)+i-1] * hh[i+(ldh*2)];     // 3 = 3 + hh(i-1,2) * hh(i,3)
                scalarprods[5] += hh[(ldh*2)+i-1] * hh[i+(ldh*3)];   // 6 = 6 + hh(i-1,3) * hh(i,4)
                scalarprods[9] += hh[(ldh*3)+i-1] * hh[i+(ldh*4)];   //10 =10 + hh(i-1,4) * hh(i,5)
                scalarprods[14] += hh[(ldh*4)+i-1] * hh[i+(ldh*5)];  //15 =15 + hh(i-1,5) * hh(i,6)

                scalarprods[1] += hh[i-2] * hh[i+(ldh*2)];          // 2 = 2 + hh(i-2,1) * hh(i,3)
                scalarprods[4] += hh[(ldh*1)+i-2] * hh[i+(ldh*3)];  // 5 = 5 + hh(i-2,2) * hh(i,4)
                scalarprods[8] += hh[(ldh*2)+i-2] * hh[i+(ldh*4)];  // 9 = 9 + hh(i-2,3) * hh(i,5)
                scalarprods[13] += hh[(ldh*3)+i-2] * hh[i+(ldh*5)]; //14 =14 + hh(i-2,4) * hh(i,6)

                scalarprods[3] += hh[i-3] * hh[i+(ldh*3)];          // 4 = 4 + hh(i-3,1) * hh(i,4)
                scalarprods[7] += hh[(ldh)+i-3] * hh[i+(ldh*4)];    // 8 = 8 + hh(i-3,2) * hh(i,5)
                scalarprods[12] += hh[(ldh*2)+i-3] * hh[i+(ldh*5)]; //13 =13 + hh(i-3,3) * hh(i,6)

                scalarprods[6] += hh[i-4] * hh[i+(ldh*4)];          // 7 = 7 + hh(i-4,1) * hh(i,5)
                scalarprods[11] += hh[(ldh)+i-4] * hh[i+(ldh*5)];   //12 =12 + hh(i-4,2) * hh(i,6)

                scalarprods[10] += hh[i-5] * hh[i+(ldh*5)];         //11 =11 + hh(i-5,1) * hh(i,6)
        }

        // Production level kernel calls with padding
#ifdef DOUBLE_PRECISION_REAL
        for (i = 0; i < nq-2; i+=4)
        {
#ifdef HAVE_SSE_INTRINSICS
                hh_trafo_kernel_4_SSE_6hv_double(&q[i], hh, nb, ldq, ldh, scalarprods);
#endif
#ifdef HAVE_SPARC64_SSE
                hh_trafo_kernel_4_SPARC64_6hv_double(&q[i], hh, nb, ldq, ldh, scalarprods);
#endif

                worked_on += 4;
        }
#endif
#ifdef SINGLE_PRECISION_REAL
        for (i = 0; i < nq-4; i+=8)
        {
#ifdef HAVE_SSE_INTRINSICS
                hh_trafo_kernel_8_SSE_6hv_single(&q[i], hh, nb, ldq, ldh, scalarprods);
#endif
#ifdef HAVE_SPARC64_SSE
                hh_trafo_kernel_8_SPARC64_6hv_single(&q[i], hh, nb, ldq, ldh, scalarprods);
#endif

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
#ifdef HAVE_SSE_INTRINSICS
                hh_trafo_kernel_2_SSE_6hv_double(&q[i], hh, nb, ldq, ldh, scalarprods);
#endif
#ifdef HAVE_SPARC64_SSE
                hh_trafo_kernel_2_SPARC64_6hv_double(&q[i], hh, nb, ldq, ldh, scalarprods);
#endif

                worked_on += 2;
        }
#endif
#ifdef SINGLE_PRECISION_REAL
        if (nq -i == 4)
        {
#ifdef HAVE_SSE_INTRINSICS
                hh_trafo_kernel_4_SSE_6hv_single(&q[i], hh, nb, ldq, ldh, scalarprods);
#endif
#ifdef HAVE_SPARC64_SSE
                hh_trafo_kernel_4_SPARC64_6hv_single(&q[i], hh, nb, ldq, ldh, scalarprods);
#endif
                worked_on += 4;
        }
#endif
#ifdef WITH_DEBUG
        if (worked_on != nq)
        {
#ifdef HAVE_SSE_INTRINSICS
                printf("Error in real SSE BLOCK6 kernel \n");
#endif
#ifdef HAVE_SPARC64_SSE
                printf("Error in real SPARC64 BLOCK6 kernel \n");
#endif

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
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_4_SSE_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_8_SSE_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods)
#endif
#endif
#ifdef HAVE_SPARC64_SSE
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_4_SPARC64_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_8_SPARC64_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods)
#endif
#endif

{
        /////////////////////////////////////////////////////
        // Matrix Vector Multiplication, Q [4 x nb+3] * hh
        // hh contains four householder vectors
        /////////////////////////////////////////////////////
        int i;

        __SSE_DATATYPE a1_1 = _SSE_LOAD(&q[ldq*5]);   // a_1_1 = q(1:nq,6)
        __SSE_DATATYPE a2_1 = _SSE_LOAD(&q[ldq*4]);   // a_2_1 = q(1:nq,5)
        __SSE_DATATYPE a3_1 = _SSE_LOAD(&q[ldq*3]);   // a_3_1 = q(1:nq,4)
        __SSE_DATATYPE a4_1 = _SSE_LOAD(&q[ldq*2]);   // a_4_1 = q(1:nq,3)
        __SSE_DATATYPE a5_1 = _SSE_LOAD(&q[ldq]);     // a_5_1 = q(1,nq,2)
        __SSE_DATATYPE a6_1 = _SSE_LOAD(&q[0]);       // a_6_1 = q(1:nq,1)

#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE h_6_5 = _SSE_SET1(hh[(ldh*5)+1]);  // h_6_5 = hh(2,6)
        __SSE_DATATYPE h_6_4 = _SSE_SET1(hh[(ldh*5)+2]);  // h_6_4 = hh(3,6)
        __SSE_DATATYPE h_6_3 = _SSE_SET1(hh[(ldh*5)+3]);  // h_6_3 = hh(4,6)
        __SSE_DATATYPE h_6_2 = _SSE_SET1(hh[(ldh*5)+4]);  // h_6_2 = hh(5,6)
        __SSE_DATATYPE h_6_1 = _SSE_SET1(hh[(ldh*5)+5]);  // h_6_1 = hh(6,6)
#endif

#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE h_6_5 = _SSE_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
        __SSE_DATATYPE h_6_4 = _SSE_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
        __SSE_DATATYPE h_6_3 = _SSE_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
        __SSE_DATATYPE h_6_2 = _SSE_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
        __SSE_DATATYPE h_6_1 = _SSE_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif

        register __SSE_DATATYPE t1 = _SSE_ADD(a6_1, _SSE_MUL(a5_1, h_6_5));  // t1 = a_6_1 + a_5_1 * h_6_5
        t1 = _SSE_ADD(t1, _SSE_MUL(a4_1, h_6_4)); // t1 = t1 + a_4_1 * h_6_4
        t1 = _SSE_ADD(t1, _SSE_MUL(a3_1, h_6_3)); // t1 = t1 + a_3_1 * h_6_3
        t1 = _SSE_ADD(t1, _SSE_MUL(a2_1, h_6_2)); // t1 = t1 + a_2_1 * h_6_2
        t1 = _SSE_ADD(t1, _SSE_MUL(a1_1, h_6_1)); // t1 = t1 + a_1_1 * h_6_1

#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE h_5_4 = _SSE_SET1(hh[(ldh*4)+1]);  // h_5_4 = hh(2,5)
        __SSE_DATATYPE h_5_3 = _SSE_SET1(hh[(ldh*4)+2]);  // h_5_3 = hh(3,5)
        __SSE_DATATYPE h_5_2 = _SSE_SET1(hh[(ldh*4)+3]);  // h_5_2 = hh(4,5)
        __SSE_DATATYPE h_5_1 = _SSE_SET1(hh[(ldh*4)+4]);  // h_5_1 = hh(5,5)
#endif

#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE h_5_4 = _SSE_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
        __SSE_DATATYPE h_5_3 = _SSE_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
        __SSE_DATATYPE h_5_2 = _SSE_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
        __SSE_DATATYPE h_5_1 = _SSE_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif

        register __SSE_DATATYPE v1 = _SSE_ADD(a5_1, _SSE_MUL(a4_1, h_5_4)); // v1 = a_5_1 + a_4_1 * h_5_4
        v1 = _SSE_ADD(v1, _SSE_MUL(a3_1, h_5_3)); // v1 = v1 + a_3_1 * h_5_3
        v1 = _SSE_ADD(v1, _SSE_MUL(a2_1, h_5_2)); // v1 = v1 + a_2_1 * h_5_2
        v1 = _SSE_ADD(v1, _SSE_MUL(a1_1, h_5_1)); // v1 = v1 + a_1_1 * h_5_1

#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE h_4_3 = _SSE_SET1(hh[(ldh*3)+1]);  // h_4_3 = hh(2,4)
        __SSE_DATATYPE h_4_2 = _SSE_SET1(hh[(ldh*3)+2]);  // h_4_2 = hh(3,4)
        __SSE_DATATYPE h_4_1 = _SSE_SET1(hh[(ldh*3)+3]);  // h_4_1 = hh(4,4)
#endif

#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE h_4_3 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
        __SSE_DATATYPE h_4_2 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
        __SSE_DATATYPE h_4_1 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

        register __SSE_DATATYPE w1 = _SSE_ADD(a4_1, _SSE_MUL(a3_1, h_4_3)); // w1 = a_4_1 + a_3_1 * h_4_3
        w1 = _SSE_ADD(w1, _SSE_MUL(a2_1, h_4_2));  // w1 = w1 + a_2_1 * h_4_2
        w1 = _SSE_ADD(w1, _SSE_MUL(a1_1, h_4_1));  // w1 = w1 + a_1_1 * h_4_1

#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE h_2_1 = _SSE_SET1(hh[ldh+1]);     // h_2_1 = hh(2,2)
        __SSE_DATATYPE h_3_2 = _SSE_SET1(hh[(ldh*2)+1]); // h_3_2 = hh(2,3)
        __SSE_DATATYPE h_3_1 = _SSE_SET1(hh[(ldh*2)+2]); // h_3_1 = hh(3,3)
#endif

#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE h_2_1 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
        __SSE_DATATYPE h_3_2 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
        __SSE_DATATYPE h_3_1 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

        register __SSE_DATATYPE z1 = _SSE_ADD(a3_1, _SSE_MUL(a2_1, h_3_2));  // z1 = a_3_1 + a_2_1 * h_3_2
        z1 = _SSE_ADD(z1, _SSE_MUL(a1_1, h_3_1));  // z1 = z1 + a_1_1 * h_3_1
        register __SSE_DATATYPE y1 = _SSE_ADD(a2_1, _SSE_MUL(a1_1, h_2_1));  // y1 = a_2_1 + a_1_1 * h_2_1

        register __SSE_DATATYPE x1 = a1_1;  // x1 = a_1_1

        __SSE_DATATYPE a1_2 = _SSE_LOAD(&q[(ldq*5)+offset]);
        __SSE_DATATYPE a2_2 = _SSE_LOAD(&q[(ldq*4)+offset]);
        __SSE_DATATYPE a3_2 = _SSE_LOAD(&q[(ldq*3)+offset]);
        __SSE_DATATYPE a4_2 = _SSE_LOAD(&q[(ldq*2)+offset]);
        __SSE_DATATYPE a5_2 = _SSE_LOAD(&q[(ldq)+offset]);
        __SSE_DATATYPE a6_2 = _SSE_LOAD(&q[offset]);

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

        for(i = 6; i < nb; i++)             // do i=7,nb
        {
#ifdef HAVE_SSE_INTRINSICS
                h1 = _SSE_SET1(hh[i-5]);  // h1 = hh(i-5,1)
#endif
#ifdef HAVE_SPARC64_SSE
                h1 = _SSE_SET(hh[i-5], hh[i-5]);
#endif
        
                q1 = _SSE_LOAD(&q[i*ldq]);
                q2 = _SSE_LOAD(&q[(i*ldq)+offset]);

                x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));  // x1 = x1 + q(1:nq,i) * h1
                x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));

#ifdef HAVE_SSE_INTRINSICS
                h2 = _SSE_SET1(hh[ldh+i-4]);   // h2 = hh(i-4,2)
#endif

#ifdef HAVE_SPARC64_SSE
                h2 = _SSE_SET(hh[ldh+i-4], hh[ldh+i-4]);
#endif

                y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));  // y1 = y1 + q1(1:nq,i) * h2
                y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));

#ifdef HAVE_SSE_INTRINSICS
                h3 = _SSE_SET1(hh[(ldh*2)+i-3]);  // h3 = hh(i-3,3)
#endif
#ifdef HAVE_SPARC64_SSE
                h3 = _SSE_SET(hh[(ldh*2)+i-3], hh[(ldh*2)+i-3]);
#endif

                z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));   // z1 = z1 + q(1:nq,i) * h3
                z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
#ifdef HAVE_SSE_INTRINSICS
                h4 = _SSE_SET1(hh[(ldh*3)+i-2]);    // h4 = hh(i-2,4)
#endif
#ifdef HAVE_SPARC64_SSE
                h4 = _SSE_SET(hh[(ldh*3)+i-2], hh[(ldh*3)+i-2]);
#endif

                w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));  // w1 = w1 + q1(1:nq,i) * h4
                w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));

#ifdef HAVE_SSE_INTRINSICS
                h5 = _SSE_SET1(hh[(ldh*4)+i-1]);   // h5 = hh(i-1,5)
#endif
#ifdef HAVE_SPARC64_SSE
                h5 = _SSE_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif
                v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5)); // v1 = v1 + q1(1:nq,i) * h5
                v2 = _SSE_ADD(v2, _SSE_MUL(q2,h5));

#ifdef HAVE_SSE_INTRINSICS
                h6 = _SSE_SET1(hh[(ldh*5)+i]);  // h6 = hh(i,6)
#endif

#ifdef HAVE_SPARC64_SSE
                h6 = _SSE_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif

                t1 = _SSE_ADD(t1, _SSE_MUL(q1,h6));  // t1 = t1 + q1(1:nq,i) * h6
                t2 = _SSE_ADD(t2, _SSE_MUL(q2,h6));
        }

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-5]);  // h1 = hh(nb-4,1)
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-5], hh[nb-5]);
#endif

        q1 = _SSE_LOAD(&q[nb*ldq]);
        q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));   // x1 = x1 + q1(1:nq,nb+1) * h1
        x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));

#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-4]);  // h2 = hh(nb-3,2)
#endif

#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-4], hh[ldh+nb-4]);
#endif


        y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));  // y1 = y1 + q1(1:nq,nb+1) * h2
        y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));

#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+nb-3]); // h3 = hh(nb-2,3)
#endif

#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+nb-3], hh[(ldh*2)+nb-3]);
#endif


        z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3)); // z1 = z1 + q1(1:nq,nb+1) * h3
        z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));

#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+nb-2]);  // h4 = hh(nb-1,4)
#endif

#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif

        w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));  // w1 = w1 + q1(1:nq,nb+1) * h4
        w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));

#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+nb-1]);  // h5 = hh(nb, 5)
#endif

#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif



        v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));  // v1 = v1 + q1(1:nq,nb+1) * h5
        v2 = _SSE_ADD(v2, _SSE_MUL(q2,h5));
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-4]);   // h1 = hh(nb-3,1)
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-4], hh[nb-4]);
#endif

        q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
        q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));  // x1 = x1 + q1(1:nq,nb+2) * h1
        x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-3]); // h2 = hh(nb-2,2)
#endif
#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-3], hh[ldh+nb-3]);
#endif

        y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));  // y1 = y1 + q1(1:nq,nb+2) * h2
        y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));

#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+nb-2]);  // h3 = hh(nb-1,3)
#endif
#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif

        z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));  // z1 = z1 + q1(1:nq,nb+2)  * h3
        z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));

#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+nb-1]); // h4 = hh(nb,4)
#endif
#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif


        w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));  // w1 = w1 + q1(1:nq,nb+2) * h4
        w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-3]); // h1 = hh(nb-2,1)
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-3], hh[nb-3]);
#endif


        q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
        q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));  // x1 = x1 + q1(1:nq,nb+3) * h1
        x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));

#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-2]); // h2 = hh(nb-1,2)
#endif

#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

        y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));  // y1 = y1 + q1(1:nq,nb+3) * h2
        y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+nb-1]); // h3 = hh(nb,3)
#endif

#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif

        z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3)); // z1 = z1 + q1(1:nq,nb+3) * h3
        z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-2]);  // h1 = hh(nb-1,1)
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

        q1 = _SSE_LOAD(&q[(nb+3)*ldq]);
        q2 = _SSE_LOAD(&q[((nb+3)*ldq)+offset]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));   // x1 = x1 + q1(1:nq,nb+4) * h1
        x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));

#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-1]);  // h2 = hh(nb,2)
#endif

#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif

        y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));  // y1 = y1 + q1(1:n1,nb+4) * h2
        y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-1]);  // h1 = hh(nb,1)
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif

        q1 = _SSE_LOAD(&q[(nb+4)*ldq]);
        q2 = _SSE_LOAD(&q[((nb+4)*ldq)+offset]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));  // x1 = x1 + q1(1:nq,nb+5) * h1
        x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));

        /////////////////////////////////////////////////////
        // Apply tau, correct wrong calculation using pre-calculated scalar products
        /////////////////////////////////////////////////////


#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau1 = _SSE_SET1(hh[0]);  // tau1 = hh(1,1)
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau1 = _SSE_SET(hh[0], hh[0]);
#endif
        x1 = _SSE_MUL(x1, tau1);  // x1 = x1 * tau1
        x2 = _SSE_MUL(x2, tau1);

#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau2 = _SSE_SET1(hh[ldh]);            // tau2 = hh(1,2)
        __SSE_DATATYPE vs_1_2 = _SSE_SET1(scalarprods[0]);   // vs_1_2 = product(1)
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau2 = _SSE_SET(hh[ldh], hh[ldh]);
        __SSE_DATATYPE vs_1_2 = _SSE_SET(scalarprods[0], scalarprods[0]);
#endif

        h2 = _SSE_MUL(tau2, vs_1_2);  // h2 = tau2 * vs_1_2

        y1 = _SSE_SUB(_SSE_MUL(y1,tau2), _SSE_MUL(x1,h2));   // y1 = y1 * tau2 - x1 * h2
        y2 = _SSE_SUB(_SSE_MUL(y2,tau2), _SSE_MUL(x2,h2));

#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau3 = _SSE_SET1(hh[ldh*2]);         // tau3 = hh(1,3)
        __SSE_DATATYPE vs_1_3 = _SSE_SET1(scalarprods[1]);  // vs_1_3 = prods(2)
        __SSE_DATATYPE vs_2_3 = _SSE_SET1(scalarprods[2]);  // vs_2_3 = prods(3)
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau3 = _SSE_SET(hh[ldh*2], hh[ldh*2]);
        __SSE_DATATYPE vs_1_3 = _SSE_SET(scalarprods[1], scalarprods[1]);
        __SSE_DATATYPE vs_2_3 = _SSE_SET(scalarprods[2], scalarprods[2]) ;
#endif

        h2 = _SSE_MUL(tau3, vs_1_3);   // h2 = tau3 * vs_1_3
        h3 = _SSE_MUL(tau3, vs_2_3);   // h3 = tau3 * vs_2_3

        z1 = _SSE_SUB(_SSE_MUL(z1,tau3), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))); // Z1 = z1 * tau3 - (y1*h3 + x1*h2)
        z2 = _SSE_SUB(_SSE_MUL(z2,tau3), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2)));

#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau4 = _SSE_SET1(hh[ldh*3]);         // tau4 = hh(1,4)
        __SSE_DATATYPE vs_1_4 = _SSE_SET1(scalarprods[3]);  // vs_1_4 = prods(4)
        __SSE_DATATYPE vs_2_4 = _SSE_SET1(scalarprods[4]);  // vs_2_4 = prods(5)
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau4 = _SSE_SET(hh[ldh*3], hh[ldh*3]);
        __SSE_DATATYPE vs_1_4 = _SSE_SET(scalarprods[3], scalarprods[3]);
        __SSE_DATATYPE vs_2_4 = _SSE_SET(scalarprods[4], scalarprods[4]);
#endif

        h2 = _SSE_MUL(tau4, vs_1_4);  // h2 = tau4 * vs_1_4
        h3 = _SSE_MUL(tau4, vs_2_4);  // h3 = tau4 * vs_2_4
#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE vs_3_4 = _SSE_SET1(scalarprods[5]);  // vs_3_4 = prods(6)
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE vs_3_4 = _SSE_SET(scalarprods[5], scalarprods[5]);
#endif

        h4 = _SSE_MUL(tau4, vs_3_4); // h4 = tau4 * vs_3_4

        w1 = _SSE_SUB(_SSE_MUL(w1,tau4), _SSE_ADD(_SSE_MUL(z1,h4), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));
        w2 = _SSE_SUB(_SSE_MUL(w2,tau4), _SSE_ADD(_SSE_MUL(z2,h4), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2))));
        // w1 = w1 * tau4 - (z1 *h4 + y1 * h3 + x1 *h2)
#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau5 = _SSE_SET1(hh[ldh*4]);         // tau5 = hh(1,5)
        __SSE_DATATYPE vs_1_5 = _SSE_SET1(scalarprods[6]);  // vs_1_5 = prods(7)
        __SSE_DATATYPE vs_2_5 = _SSE_SET1(scalarprods[7]);  // vs_2_5 = prods(8)
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau5 = _SSE_SET(hh[ldh*4], hh[ldh*4]);
        __SSE_DATATYPE vs_1_5 = _SSE_SET(scalarprods[6], scalarprods[6]);
        __SSE_DATATYPE vs_2_5 = _SSE_SET(scalarprods[7], scalarprods[7]);
#endif

        h2 = _SSE_MUL(tau5, vs_1_5);   // h2 = tau5 * vs_1_5
        h3 = _SSE_MUL(tau5, vs_2_5);   // h3 = tau5 * vs_2_5
#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE vs_3_5 = _SSE_SET1(scalarprods[8]);  // vs_3_5 = prods(9)
        __SSE_DATATYPE vs_4_5 = _SSE_SET1(scalarprods[9]);  // vs_4_5 = prods(10)
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE vs_3_5 = _SSE_SET(scalarprods[8], scalarprods[8]);
        __SSE_DATATYPE vs_4_5 = _SSE_SET(scalarprods[9], scalarprods[9]);
#endif

        h4 = _SSE_MUL(tau5, vs_3_5);  // h4 = tau5 * vs_3_5
        h5 = _SSE_MUL(tau5, vs_4_5);  // h5 = tau5 * vs_4_5

        v1 = _SSE_SUB(_SSE_MUL(v1,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));
        v2 = _SSE_SUB(_SSE_MUL(v2,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w2,h5), _SSE_MUL(z2,h4)), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2))));
       // v1 = v1 * tau5 - (w1 * h5 + z1 * h4 + y1 * h3 + x1 * h2)
#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau6 = _SSE_SET1(hh[ldh*5]);         // tau6 = hh(1,6)
 	__SSE_DATATYPE vs_1_6 = _SSE_SET1(scalarprods[10]); // vs_1_6 = prods(11)
        __SSE_DATATYPE vs_2_6 = _SSE_SET1(scalarprods[11]); // vs_2_6 = prods(12)
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau6 = _SSE_SET(hh[ldh*5], hh[ldh*5]);
        __SSE_DATATYPE vs_1_6 = _SSE_SET(scalarprods[10], scalarprods[10]);
        __SSE_DATATYPE vs_2_6 = _SSE_SET(scalarprods[11], scalarprods[11]);
#endif

        h2 = _SSE_MUL(tau6, vs_1_6); // h2 = tau6 * vs_1_6
        h3 = _SSE_MUL(tau6, vs_2_6); // h3 = tau6 * vs_2_6
#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE vs_3_6 = _SSE_SET1(scalarprods[12]); // vs_3_6 = prods(13)
        __SSE_DATATYPE vs_4_6 = _SSE_SET1(scalarprods[13]); // vs_4_6 = prods(14)
        __SSE_DATATYPE vs_5_6 = _SSE_SET1(scalarprods[14]); // vs_5_6 = prods(15)
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE vs_3_6 = _SSE_SET(scalarprods[12], scalarprods[12]);
        __SSE_DATATYPE vs_4_6 = _SSE_SET(scalarprods[13], scalarprods[13]);
        __SSE_DATATYPE vs_5_6 = _SSE_SET(scalarprods[14], scalarprods[14]);
#endif

        h4 = _SSE_MUL(tau6, vs_3_6); // h4 = tau6 * vs_3_6
        h5 = _SSE_MUL(tau6, vs_4_6); // h5 = tau6 * vs_4_6
        h6 = _SSE_MUL(tau6, vs_5_6); // h6 = tau6 * vs_5_6

        t1 = _SSE_SUB(_SSE_MUL(t1,tau6), _SSE_ADD( _SSE_MUL(v1,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)))));
        t2 = _SSE_SUB(_SSE_MUL(t2,tau6), _SSE_ADD( _SSE_MUL(v2,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w2,h5), _SSE_MUL(z2,h4)), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2)))));
        // t1 = t1 * tau6 - ( v1 * h6 + w1*h5 + z1*h4 +y1*h3 + x1*h2)

        /////////////////////////////////////////////////////
        // Rank-1 update of Q [4 x nb+3]
        /////////////////////////////////////////////////////

        q1 = _SSE_LOAD(&q[0]);
        q2 = _SSE_LOAD(&q[offset]);
        q1 = _SSE_SUB(q1, t1);       // q1(1:n1,1) = q1(1:nq,1) - t
        q2 = _SSE_SUB(q2, t2);
        _SSE_STORE(&q[0],q1);
        _SSE_STORE(&q[offset],q2);

#ifdef HAVE_SSE_INTRINSICS
        h6 = _SSE_SET1(hh[(ldh*5)+1]); // h6 = hh(2,6)
#endif
#ifdef HAVE_SPARC64_SSE
        h6 = _SSE_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
#endif

        q1 = _SSE_LOAD(&q[ldq]);
        q2 = _SSE_LOAD(&q[(ldq+offset)]);
        q1 = _SSE_SUB(q1, v1);  // q1(1:nq,2) = q1(1:nq,2) - v1
        q2 = _SSE_SUB(q2, v2);

        q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6)); // q1(1:nq,2) = q1(1:nq,2) - t1 *h6
        q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));

        _SSE_STORE(&q[ldq],q1);
        _SSE_STORE(&q[(ldq+offset)],q2);
#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+1]);    // h5 = hh(2,5)
#endif
#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
#endif
        q1 = _SSE_LOAD(&q[ldq*2]);
        q2 = _SSE_LOAD(&q[(ldq*2)+offset]);
        q1 = _SSE_SUB(q1, w1);      // q1(1:nq,3) =  q1(1:nq,3) - w
        q2 = _SSE_SUB(q2, w2);
        q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));  //  q1(1:nq,3) =  q1(1:nq,3) - v1 * h5
        q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));  
#ifdef HAVE_SSE_INTRINSICS
        h6 = _SSE_SET1(hh[(ldh*5)+2]);     // h6 = hh(3,6)
#endif
#ifdef HAVE_SPARC64_SSE
        h6 = _SSE_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
#endif

        q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));  // q1(1:nq,3) = q1(1:nq,3) - t1 * h6
        q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));

        _SSE_STORE(&q[ldq*2],q1);
        _SSE_STORE(&q[(ldq*2)+offset],q2);

#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+1]);  // h4 = hh(2,4)
#endif

#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

        q1 = _SSE_LOAD(&q[ldq*3]);
        q2 = _SSE_LOAD(&q[(ldq*3)+offset]);
        q1 = _SSE_SUB(q1, z1);        // q1(1:nq,4) = q1(1:nq,4) - z
        q2 = _SSE_SUB(q2, z2);

        q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));  // q1(1:nq,4) = q1(1:nq,4) - w * h4
        q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+2]);      // h5 = hh(3,5)
#endif
#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));   // q1(1:nq,4) = q1(1:nq,4) - v1 * h5
        q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
#ifdef HAVE_SSE_INTRINSICS
        h6 = _SSE_SET1(hh[(ldh*5)+3]);  // h6 = hh(4,6)
#endif
#ifdef HAVE_SPARC64_SSE
        h6 = _SSE_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));  // q1(1:nq,4) = q1(1:nq,4) - t1 * h6
        q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));

        _SSE_STORE(&q[ldq*3],q1);
        _SSE_STORE(&q[(ldq*3)+offset],q2);
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+1]);    // h3 = hh(2,3)
#endif
#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif
        q1 = _SSE_LOAD(&q[ldq*4]);
        q2 = _SSE_LOAD(&q[(ldq*4)+offset]);
        q1 = _SSE_SUB(q1, y1);  // q1(1:nq,5) = q1(1:nq,5) - y
        q2 = _SSE_SUB(q2, y2);

        q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3)); // q1(1:nq,5) = q1(1:nq,5) - z1 * h3
        q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+2]);     // h4 = hh(3,4)
#endif
#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));  // q(1:nq,5) = q(1:nq,5) - w1 * h4
        q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+3]);  // h5 = hh(4,5)
#endif
#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
#endif
        q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5)); // q(1:nq,5) = q(1:nq,5) - v * h5
        q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
#ifdef HAVE_SSE_INTRINSICS
        h6 = _SSE_SET1(hh[(ldh*5)+4]);     // h6 = hh(5,6)
#endif
#ifdef HAVE_SPARC64_SSE
        h6 = _SSE_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
#endif
        q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));  // q(1:nq,5) = q(1:nq,5) - t * h6
        q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));

        _SSE_STORE(&q[ldq*4],q1);
        _SSE_STORE(&q[(ldq*4)+offset],q2);
#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[(ldh)+1]);  // h2 = hh(2,2)
#endif
#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[(ldh)+1], hh[(ldh)+1]);
#endif
        q1 = _SSE_LOAD(&q[ldq*5]);
        q2 = _SSE_LOAD(&q[(ldq*5)+offset]);
        q1 = _SSE_SUB(q1, x1);  // q(1:nq,6) = q(1:nq,6) - x
        q2 = _SSE_SUB(q2, x2);

        q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));  // q(1:nq,6) = q(1:nq,6) - y1 * h2
        q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+2]);  // h3 = hh(3,3)
#endif
#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

        q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));  // q(1:nq,6) = q(1:nq,6) - z * h3
        q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+3]);  // h4 = hh(4,4)
#endif
#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif
        q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));  // q(1:nq,6) = q(1:nq,6) - w * h4
        q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+4]); // h5 = hh(5,5)
#endif
#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));  // q(1:nq,6) = q(1:nq,6) - v * h5
        q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
#ifdef HAVE_SSE_INTRINSICS
        h6 = _SSE_SET1(hh[(ldh*5)+5]);  // h6 = hh(6,6)
#endif
#ifdef HAVE_SPARC64_SSE
        h6 = _SSE_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif

        q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));  // q(1:nq,6) = q(1:nq,6) - t * h6
        q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));

        _SSE_STORE(&q[ldq*5],q1);
        _SSE_STORE(&q[(ldq*5)+offset],q2);

        for (i = 6; i < nb; i++)                            // for i=7,nb
        {
                q1 = _SSE_LOAD(&q[i*ldq]);
                q2 = _SSE_LOAD(&q[(i*ldq)+offset]);
#ifdef HAVE_SSE_INTRINSICS
                h1 = _SSE_SET1(hh[i-5]);                 // h1 = hh(i-5,1)  
#endif
#ifdef HAVE_SPARC64_SSE
                h1 = _SSE_SET(hh[i-5], hh[i-5]);
#endif


                q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));  // q(1:nq,i) = q(1:nq,i) - x1 * h1
                q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
#ifdef HAVE_SSE_INTRINSICS
                h2 = _SSE_SET1(hh[ldh+i-4]);  // h2 = hh(i-4,2)
#endif
#ifdef HAVE_SPARC64_SSE
                h2 = _SSE_SET(hh[ldh+i-4], hh[ldh+i-4]);
#endif


                q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));  // q(1:nq,i) - y * h2
                q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
#ifdef HAVE_SSE_INTRINSICS
                h3 = _SSE_SET1(hh[(ldh*2)+i-3]);   // h3 = hh(i-3,3)
#endif
#ifdef HAVE_SPARC64_SSE
                h3 = _SSE_SET(hh[(ldh*2)+i-3], hh[(ldh*2)+i-3]);
#endif
                q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));  // q(1:nq,i) = q(1:nq,i) - z * h3
                q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
#ifdef HAVE_SSE_INTRINSICS
                h4 = _SSE_SET1(hh[(ldh*3)+i-2]);  // h4 = hh(i-2,4)
#endif
#ifdef HAVE_SPARC64_SSE
                h4 = _SSE_SET(hh[(ldh*3)+i-2], hh[(ldh*3)+i-2]);
#endif
                q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));  // q(1:nq,i) = q(1:nq,i) - w * h4
                q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
#ifdef HAVE_SSE_INTRINSICS
                h5 = _SSE_SET1(hh[(ldh*4)+i-1]);  // h5 = hh(i-1,5)
#endif
#ifdef HAVE_SPARC64_SSE
                h5 = _SSE_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif

                q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));  // q(1:nq,i) = q(1:nq,i) - v * h5
                q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
#ifdef HAVE_SSE_INTRINSICS
                h6 = _SSE_SET1(hh[(ldh*5)+i]);     // h6 = hh(i,6)
#endif
#ifdef HAVE_SPARC64_SSE
                h6 = _SSE_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif


                q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));  // q(1:nq,i) = q(1:nq,i) - t * h6
                q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));

                _SSE_STORE(&q[i*ldq],q1);
                _SSE_STORE(&q[(i*ldq)+offset],q2);
        }
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-5]);   // h1 = hh(nb-4,1)
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-5], hh[nb-5]);
#endif


        q1 = _SSE_LOAD(&q[nb*ldq]);
        q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);

        q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));  // q(1:nq,nb+1) = q(1:nq,nb+1) - x * h1
        q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-4]);  // h2 = hh(nb-3,2)
#endif
#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-4], hh[ldh+nb-4]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));   // q(1:nq,nb+1) =  q(1:nq,nb+1) - y * h2
        q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+nb-3]);  // h3 = hh(nb-2,3)
#endif
#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+nb-3], hh[(ldh*2)+nb-3]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));  // q(1:nq,nb+1) = q(1:nq,nb+1) - z * h3
        q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+nb-2]);   // h4 = hh(nb-1,4)
#endif
#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif
        q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));  // q(1:nq,nb+1) = q(1:nq,nb+1) - w * h4
        q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+nb-1]);  // h5 = hh(nb,5)
#endif
#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif

        q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5)); // q(1:nq,nb+1) = q(1:nq,nb+1) - v * h5
        q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));

        _SSE_STORE(&q[nb*ldq],q1);
        _SSE_STORE(&q[(nb*ldq)+offset],q2);
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-4]);  // h1 = hh(nb-3,1)
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-4], hh[nb-4]);
#endif


        q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
        q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);

        q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));  // q1(1:nq,nb+2) = q1(1:nq,nb+2) + x1 * h1
        q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-3]); // h2 = hh(nb-2,2)
#endif
#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-3], hh[ldh+nb-3]);
#endif

        q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));  // q1(1:nq,nb+2) = q1(1:nq,nb+2) + y1 * h2
        q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+nb-2]);  // h3 = hh(nb-1,3)
#endif
#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif
        q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));  // q(1:nq,nb+2) = q(1:nq,nb+2) + z * h3
        q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+nb-1]); // h4 = hh(nb,4)
#endif
#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif

        q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));  // q(1:nq,nb+2) = q(1:nq,nb+2) + w * h4
        q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));

        _SSE_STORE(&q[(nb+1)*ldq],q1);
        _SSE_STORE(&q[((nb+1)*ldq)+offset],q2);
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-3]);  // h1 = hh(nb-2,1)
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-3], hh[nb-3]);
#endif


        q1 = _SSE_LOAD(&q[(nb+2)*ldq]);          
        q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);  

        q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));  // q(1:nq,nb+3) = q(1:nq,nb+3)  - x * h1
        q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-2]); // h2 = hh(nb-1,2)
#endif
#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

        q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2)); // q(1:nq,nb+3) = q(1:nq,nb+3) - y1 * h2
        q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+nb-1]); // h3 = hh(nb,3)
#endif
#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3)); // q(1:nq,nb+3) = q(1:nq,nb+3) + z * h3
        q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));

        _SSE_STORE(&q[(nb+2)*ldq],q1);
        _SSE_STORE(&q[((nb+2)*ldq)+offset],q2);
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-2]);  // h1 = hh(nb-1,1)
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif


        q1 = _SSE_LOAD(&q[(nb+3)*ldq]);
        q2 = _SSE_LOAD(&q[((nb+3)*ldq)+offset]);

        q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));  // q(1:nq,nb+4) = q(1:nq,nb+4) - x * h1
        q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-1]);  // h2 = hh(nb,2)
#endif
#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));   // q(1:nq,nb+4) = q(1:nq,nb+4) - y * h2
        q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));

        _SSE_STORE(&q[(nb+3)*ldq],q1);
        _SSE_STORE(&q[((nb+3)*ldq)+offset],q2);
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-1]);  // h1 = hh(nb,1)
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif

        q1 = _SSE_LOAD(&q[(nb+4)*ldq]);
        q2 = _SSE_LOAD(&q[((nb+4)*ldq)+offset]);

        q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));  // q(1:nq,nb+5) = q(1:nq,nb+5) - x * h1
        q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));

        _SSE_STORE(&q[(nb+4)*ldq],q1);
        _SSE_STORE(&q[((nb+4)*ldq)+offset],q2);
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
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_2_SSE_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_4_SSE_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods)
#endif
#endif
#ifdef HAVE_SPARC64_SSE
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_2_SPARC64_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_4_SPARC64_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods)
#endif
#endif

{
        /////////////////////////////////////////////////////
        // Matrix Vector Multiplication, Q [2 x nb+3] * hh
        // hh contains four householder vectors
        /////////////////////////////////////////////////////
        int i;

        __SSE_DATATYPE a1_1 = _SSE_LOAD(&q[ldq*5]);
        __SSE_DATATYPE a2_1 = _SSE_LOAD(&q[ldq*4]);
        __SSE_DATATYPE a3_1 = _SSE_LOAD(&q[ldq*3]);
        __SSE_DATATYPE a4_1 = _SSE_LOAD(&q[ldq*2]);
        __SSE_DATATYPE a5_1 = _SSE_LOAD(&q[ldq]);
        __SSE_DATATYPE a6_1 = _SSE_LOAD(&q[0]);

#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE h_6_5 = _SSE_SET1(hh[(ldh*5)+1]);
        __SSE_DATATYPE h_6_4 = _SSE_SET1(hh[(ldh*5)+2]);
        __SSE_DATATYPE h_6_3 = _SSE_SET1(hh[(ldh*5)+3]);
        __SSE_DATATYPE h_6_2 = _SSE_SET1(hh[(ldh*5)+4]);
        __SSE_DATATYPE h_6_1 = _SSE_SET1(hh[(ldh*5)+5]);
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE h_6_5 = _SSE_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
        __SSE_DATATYPE h_6_4 = _SSE_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
        __SSE_DATATYPE h_6_3 = _SSE_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
        __SSE_DATATYPE h_6_2 = _SSE_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
        __SSE_DATATYPE h_6_1 = _SSE_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif

        register __SSE_DATATYPE t1 = _SSE_ADD(a6_1, _SSE_MUL(a5_1, h_6_5));
        t1 = _SSE_ADD(t1, _SSE_MUL(a4_1, h_6_4));
        t1 = _SSE_ADD(t1, _SSE_MUL(a3_1, h_6_3));
        t1 = _SSE_ADD(t1, _SSE_MUL(a2_1, h_6_2));
        t1 = _SSE_ADD(t1, _SSE_MUL(a1_1, h_6_1));
#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE h_5_4 = _SSE_SET1(hh[(ldh*4)+1]);
        __SSE_DATATYPE h_5_3 = _SSE_SET1(hh[(ldh*4)+2]);
        __SSE_DATATYPE h_5_2 = _SSE_SET1(hh[(ldh*4)+3]);
        __SSE_DATATYPE h_5_1 = _SSE_SET1(hh[(ldh*4)+4]);
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE h_5_4 = _SSE_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
        __SSE_DATATYPE h_5_3 = _SSE_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
        __SSE_DATATYPE h_5_2 = _SSE_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
        __SSE_DATATYPE h_5_1 = _SSE_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif

        register __SSE_DATATYPE v1 = _SSE_ADD(a5_1, _SSE_MUL(a4_1, h_5_4));
        v1 = _SSE_ADD(v1, _SSE_MUL(a3_1, h_5_3));
        v1 = _SSE_ADD(v1, _SSE_MUL(a2_1, h_5_2));
        v1 = _SSE_ADD(v1, _SSE_MUL(a1_1, h_5_1));
#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE h_4_3 = _SSE_SET1(hh[(ldh*3)+1]);
        __SSE_DATATYPE h_4_2 = _SSE_SET1(hh[(ldh*3)+2]);
        __SSE_DATATYPE h_4_1 = _SSE_SET1(hh[(ldh*3)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE h_4_3 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
        __SSE_DATATYPE h_4_2 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
        __SSE_DATATYPE h_4_1 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

        register __SSE_DATATYPE w1 = _SSE_ADD(a4_1, _SSE_MUL(a3_1, h_4_3));
        w1 = _SSE_ADD(w1, _SSE_MUL(a2_1, h_4_2));
        w1 = _SSE_ADD(w1, _SSE_MUL(a1_1, h_4_1));
#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE h_2_1 = _SSE_SET1(hh[ldh+1]);
        __SSE_DATATYPE h_3_2 = _SSE_SET1(hh[(ldh*2)+1]);
        __SSE_DATATYPE h_3_1 = _SSE_SET1(hh[(ldh*2)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE h_2_1 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
        __SSE_DATATYPE h_3_2 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
        __SSE_DATATYPE h_3_1 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
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
#ifdef HAVE_SSE_INTRINSICS
                h1 = _SSE_SET1(hh[i-5]);
#endif
#ifdef HAVE_SPARC64_SSE
                h1 = _SSE_SET(hh[i-5], hh[i-5]);
#endif

                q1 = _SSE_LOAD(&q[i*ldq]);

                x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
#ifdef HAVE_SSE_INTRINSICS
                h2 = _SSE_SET1(hh[ldh+i-4]);
#endif
#ifdef HAVE_SPARC64_SSE
                h2 = _SSE_SET(hh[ldh+i-4], hh[ldh+i-4]);
#endif


                y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
#ifdef HAVE_SSE_INTRINSICS
                h3 = _SSE_SET1(hh[(ldh*2)+i-3]);
#endif
#ifdef HAVE_SPARC64_SSE
                h3 = _SSE_SET(hh[(ldh*2)+i-3], hh[(ldh*2)+i-3]);
#endif

                z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
#ifdef HAVE_SSE_INTRINSICS
                h4 = _SSE_SET1(hh[(ldh*3)+i-2]);
#endif
#ifdef HAVE_SPARC64_SSE
                h4 = _SSE_SET(hh[(ldh*3)+i-2], hh[(ldh*3)+i-2]);
#endif

                w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
#ifdef HAVE_SSE_INTRINSICS
                h5 = _SSE_SET1(hh[(ldh*4)+i-1]);
#endif
#ifdef HAVE_SPARC64_SSE
                h5 = _SSE_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif

                v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
                h6 = _SSE_SET1(hh[(ldh*5)+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
                h6 = _SSE_SET1(hh[(ldh*5)+i]) ;
#endif
#endif
#ifdef HAVE_SPARC64_SSE
                h6 = _SSE_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif

                t1 = _SSE_ADD(t1, _SSE_MUL(q1,h6));

        }
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-5]);
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-5], hh[nb-5]);
#endif


        q1 = _SSE_LOAD(&q[nb*ldq]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-4]);
#endif
#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-4], hh[ldh+nb-4]);
#endif


        y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+nb-3]);
#endif
#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+nb-3], hh[(ldh*2)+nb-3]);
#endif


        z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif




        w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif


        v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-4]);
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-4], hh[nb-4]);
#endif


        q1 = _SSE_LOAD(&q[(nb+1)*ldq]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-3]);
#endif
#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-3], hh[ldh+nb-3]);
#endif


        y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif


        z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif


        w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-3]);
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-3], hh[nb-3]);
#endif


        q1 = _SSE_LOAD(&q[(nb+2)*ldq]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif


        y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif


        z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif
        q1 = _SSE_LOAD(&q[(nb+3)*ldq]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif


        y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif


        q1 = _SSE_LOAD(&q[(nb+4)*ldq]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));

        /////////////////////////////////////////////////////
        // Apply tau, correct wrong calculation using pre-calculated scalar products
        /////////////////////////////////////////////////////
#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau1 = _SSE_SET1(hh[0]);
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau1 = _SSE_SET(hh[0], hh[0]);
#endif

        x1 = _SSE_MUL(x1, tau1);
#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau2 = _SSE_SET1(hh[ldh]);
        __SSE_DATATYPE vs_1_2 = _SSE_SET1(scalarprods[0]);
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau2 = _SSE_SET(hh[ldh], hh[ldh]);
        __SSE_DATATYPE vs_1_2 = _SSE_SET(scalarprods[0], scalarprods[0]);
#endif


        h2 = _SSE_MUL(tau2, vs_1_2);

        y1 = _SSE_SUB(_SSE_MUL(y1,tau2), _SSE_MUL(x1,h2));
#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau3 = _SSE_SET1(hh[ldh*2]);
        __SSE_DATATYPE vs_1_3 = _SSE_SET1(scalarprods[1]);
        __SSE_DATATYPE vs_2_3 = _SSE_SET1(scalarprods[2]);
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau3 = _SSE_SET(hh[ldh*2], hh[ldh*2]);
        __SSE_DATATYPE vs_1_3 = _SSE_SET(scalarprods[1], scalarprods[1]);
        __SSE_DATATYPE vs_2_3 = _SSE_SET(scalarprods[2], scalarprods[2]);
#endif


        h2 = _SSE_MUL(tau3, vs_1_3);
        h3 = _SSE_MUL(tau3, vs_2_3);

        z1 = _SSE_SUB(_SSE_MUL(z1,tau3), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)));
#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau4 = _SSE_SET1(hh[ldh*3]);
        __SSE_DATATYPE vs_1_4 = _SSE_SET1(scalarprods[3]);
        __SSE_DATATYPE vs_2_4 = _SSE_SET1(scalarprods[4]);
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau4 = _SSE_SET(hh[ldh*3], hh[ldh*3]);
        __SSE_DATATYPE vs_1_4 = _SSE_SET(scalarprods[3], scalarprods[3]);
        __SSE_DATATYPE vs_2_4 = _SSE_SET(scalarprods[4], scalarprods[4]);
#endif

        h2 = _SSE_MUL(tau4, vs_1_4);
        h3 = _SSE_MUL(tau4, vs_2_4);
#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE vs_3_4 = _SSE_SET1(scalarprods[5]);
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE vs_3_4 = _SSE_SET(scalarprods[5], scalarprods[5]);
#endif

        h4 = _SSE_MUL(tau4, vs_3_4);

        w1 = _SSE_SUB(_SSE_MUL(w1,tau4), _SSE_ADD(_SSE_MUL(z1,h4), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));
#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau5 = _SSE_SET1(hh[ldh*4]);
        __SSE_DATATYPE vs_1_5 = _SSE_SET1(scalarprods[6]);
        __SSE_DATATYPE vs_2_5 = _SSE_SET1(scalarprods[7]);
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau5 = _SSE_SET(hh[ldh*4], hh[ldh*4]);
        __SSE_DATATYPE vs_1_5 = _SSE_SET(scalarprods[6], scalarprods[6]);
        __SSE_DATATYPE vs_2_5 = _SSE_SET(scalarprods[7], scalarprods[7]) ;
#endif

        h2 = _SSE_MUL(tau5, vs_1_5);
        h3 = _SSE_MUL(tau5, vs_2_5);
#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE vs_3_5 = _SSE_SET1(scalarprods[8]);
        __SSE_DATATYPE vs_4_5 = _SSE_SET1(scalarprods[9]);
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE vs_3_5 = _SSE_SET(scalarprods[8], scalarprods[8]);
        __SSE_DATATYPE vs_4_5 = _SSE_SET(scalarprods[9], scalarprods[9]);
#endif

        h4 = _SSE_MUL(tau5, vs_3_5);
        h5 = _SSE_MUL(tau5, vs_4_5);

        v1 = _SSE_SUB(_SSE_MUL(v1,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));
#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau6 = _SSE_SET1(hh[ldh*5]);
        __SSE_DATATYPE vs_1_6 = _SSE_SET1(scalarprods[10]);
        __SSE_DATATYPE vs_2_6 = _SSE_SET1(scalarprods[11]);
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau6 = _SSE_SET(hh[ldh*5], hh[ldh*5]);
        __SSE_DATATYPE vs_1_6 = _SSE_SET(scalarprods[10], scalarprods[10]);
        __SSE_DATATYPE vs_2_6 = _SSE_SET(scalarprods[11], scalarprods[11]);
#endif

        h2 = _SSE_MUL(tau6, vs_1_6);
        h3 = _SSE_MUL(tau6, vs_2_6);
#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE vs_3_6 = _SSE_SET1(scalarprods[12]);
        __SSE_DATATYPE vs_4_6 = _SSE_SET1(scalarprods[13]);
        __SSE_DATATYPE vs_5_6 = _SSE_SET1(scalarprods[14]);
#endif
#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE vs_3_6 = _SSE_SET(scalarprods[12], scalarprods[12]);
        __SSE_DATATYPE vs_4_6 = _SSE_SET(scalarprods[13], scalarprods[13]);
        __SSE_DATATYPE vs_5_6 = _SSE_SET(scalarprods[14], scalarprods[14]);
#endif


        h4 = _SSE_MUL(tau6, vs_3_6);
        h5 = _SSE_MUL(tau6, vs_4_6);
        h6 = _SSE_MUL(tau6, vs_5_6);

        t1 = _SSE_SUB(_SSE_MUL(t1,tau6), _SSE_ADD( _SSE_MUL(v1,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)))));

        /////////////////////////////////////////////////////
        // Rank-1 update of Q [2 x nb+3]
        /////////////////////////////////////////////////////

        q1 = _SSE_LOAD(&q[0]);
        q1 = _SSE_SUB(q1, t1);
        _SSE_STORE(&q[0],q1);
#ifdef HAVE_SSE_INTRINSICS
        h6 = _SSE_SET1(hh[(ldh*5)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h6 = _SSE_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
#endif


        q1 = _SSE_LOAD(&q[ldq]);
        q1 = _SSE_SUB(q1, v1);

        q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));

        _SSE_STORE(&q[ldq],q1);
#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
#endif

        q1 = _SSE_LOAD(&q[ldq*2]);
        q1 = _SSE_SUB(q1, w1);

        q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
#ifdef HAVE_SSE_INTRINSICS
        h6 = _SSE_SET1(hh[(ldh*5)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
        h6 = _SSE_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));

        _SSE_STORE(&q[ldq*2],q1);
#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif


        q1 = _SSE_LOAD(&q[ldq*3]);
        q1 = _SSE_SUB(q1, z1);

        q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
#endif
        q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
#ifdef HAVE_SSE_INTRINSICS
        h6 = _SSE_SET1(hh[(ldh*5)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
        h6 = _SSE_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));

        _SSE_STORE(&q[ldq*3],q1);
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif


        q1 = _SSE_LOAD(&q[ldq*4]);
        q1 = _SSE_SUB(q1, y1);

        q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
#ifdef HAVE_SSE_INTRINSICS
        h6 = _SSE_SET1(hh[(ldh*5)+4]);
#endif
#ifdef HAVE_SPARC64_SSE
        h6 = _SSE_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));

        _SSE_STORE(&q[ldq*4],q1);
#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[(ldh)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[(ldh)+1], hh[(ldh)+1]);
#endif


        q1 = _SSE_LOAD(&q[ldq*5]);
        q1 = _SSE_SUB(q1, x1);

        q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+4]);
#endif
#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
#ifdef HAVE_SSE_INTRINSICS
        h6 = _SSE_SET1(hh[(ldh*5)+5]);
#endif
#ifdef HAVE_SPARC64_SSE
        h6 = _SSE_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif
        q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));

        _SSE_STORE(&q[ldq*5],q1);

        for (i = 6; i < nb; i++)
        {
                q1 = _SSE_LOAD(&q[i*ldq]);
#ifdef HAVE_SSE_INTRINSICS
                h1 = _SSE_SET1(hh[i-5]);
#endif
#ifdef HAVE_SPARC64_SSE
                h1 = _SSE_SET(hh[i-5], hh[i-5]);
#endif


                q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
#ifdef HAVE_SSE_INTRINSICS
                h2 = _SSE_SET1(hh[ldh+i-4]);
#endif
#ifdef HAVE_SPARC64_SSE
                h2 = _SSE_SET(hh[ldh+i-4], hh[ldh+i-4]);
#endif


                q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
#ifdef HAVE_SSE_INTRINSICS
                h3 = _SSE_SET1(hh[(ldh*2)+i-3]);
#endif
#ifdef HAVE_SPARC64_SSE
                h3 = _SSE_SET(hh[(ldh*2)+i-3], hh[(ldh*2)+i-3]);
#endif


                q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
#ifdef HAVE_SSE_INTRINSICS
                h4 = _SSE_SET1(hh[(ldh*3)+i-2]);
#endif
#ifdef HAVE_SPARC64_SSE
                h4 = _SSE_SET(hh[(ldh*3)+i-2], hh[(ldh*3)+i-2]);
#endif


                q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
#ifdef HAVE_SSE_INTRINSICS
                h5 = _SSE_SET1(hh[(ldh*4)+i-1]);
#endif
#ifdef HAVE_SPARC64_SSE
                h5 = _SSE_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif


                q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
#ifdef HAVE_SSE_INTRINSICS
                h6 = _SSE_SET1(hh[(ldh*5)+i]);
#endif
#ifdef HAVE_SPARC64_SSE
                h6 = _SSE_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif


                q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));

                _SSE_STORE(&q[i*ldq],q1);
        }
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-5]);
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-5], hh[nb-5]);
#endif


        q1 = _SSE_LOAD(&q[nb*ldq]);

        q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-4]);
#endif
#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-4], hh[ldh+nb-4]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+nb-3]);
#endif
#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+nb-3], hh[(ldh*2)+nb-3]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));

        _SSE_STORE(&q[nb*ldq],q1);
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-4]);
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-4], hh[nb-4]);
#endif

        q1 = _SSE_LOAD(&q[(nb+1)*ldq]);

        q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-3]);
#endif
#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-3], hh[ldh+nb-3]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));

        _SSE_STORE(&q[(nb+1)*ldq],q1);
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-3]);
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-3], hh[nb-3]);
#endif


        q1 = _SSE_LOAD(&q[(nb+2)*ldq]);

        q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));

        _SSE_STORE(&q[(nb+2)*ldq],q1);
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif


        q1 = _SSE_LOAD(&q[(nb+3)*ldq]);

        q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));

        _SSE_STORE(&q[(nb+3)*ldq],q1);
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif

        q1 = _SSE_LOAD(&q[(nb+4)*ldq]);

        q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));

        _SSE_STORE(&q[(nb+4)*ldq],q1);
}
