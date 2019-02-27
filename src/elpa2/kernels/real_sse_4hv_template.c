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
// Author: Alexander Heinecke (alexander.heinecke@mytum.de)
// Adapted for building a shared-library by Andreas Marek, MPCDF (andreas.marek@mpcdf.mpg.de)
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
#define _SSE_SET1 _mm_set1_ps
#define _SSE_SET _mm_set_ps
#endif

#define __forceinline __attribute__((always_inline)) static

#ifdef HAVE_SSE_INTRINSICS
#undef __AVX__
#endif

//Forward declaration
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_2_SSE_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4);
__forceinline void hh_trafo_kernel_4_SSE_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4);
__forceinline void hh_trafo_kernel_6_SSE_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4);
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_4_SSE_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);
__forceinline void hh_trafo_kernel_8_SSE_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);
__forceinline void hh_trafo_kernel_12_SSE_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);
#endif
#endif
#ifdef HAVE_SPARC64_SSE
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_2_SPARC64_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4);
__forceinline void hh_trafo_kernel_4_SPARC64_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4);
__forceinline void hh_trafo_kernel_6_SPARC64_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4);
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_4_SPARC64_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);
__forceinline void hh_trafo_kernel_8_SPARC64_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);
__forceinline void hh_trafo_kernel_12_SPARC64_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);
#endif
#endif

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
void quad_hh_trafo_real_sse_4hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif
#ifdef SINGLE_PRECISION_REAL
void quad_hh_trafo_real_sse_4hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif
#endif
#ifdef HAVE_SPARC64_SSE
#ifdef DOUBLE_PRECISION_REAL
void quad_hh_trafo_real_sparc64_4hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif
#ifdef SINGLE_PRECISION_REAL
void quad_hh_trafo_real_sparc64_4hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif
#endif

/*
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine quad_hh_trafo_real_sse_4hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="quad_hh_trafo_real_sse_4hv_double")
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
!f>   subroutine quad_hh_trafo_real_sparc64_4hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="quad_hh_trafo_real_sparc64_4hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_double)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine quad_hh_trafo_real_sse_4hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="quad_hh_trafo_real_sse_4hv_single")
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
!f>   subroutine quad_hh_trafo_real_sparc64_4hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="quad_hh_trafo_real_sparc64_4hv_single")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_float)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
void quad_hh_trafo_real_sse_4hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
#ifdef SINGLE_PRECISION_REAL
void quad_hh_trafo_real_sse_4hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
#endif
#ifdef HAVE_SPARC64_SSE
#ifdef DOUBLE_PRECISION_REAL
void quad_hh_trafo_real_sparc64_4hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
#ifdef SINGLE_PRECISION_REAL
void quad_hh_trafo_real_sparc64_4hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif

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
        double s_1_2 = hh[(ldh)+1];    // hh(2,2)
        double s_1_3 = hh[(ldh*2)+2];  // hh(3,3)
        double s_2_3 = hh[(ldh*2)+1];  // hh(2,3)
        double s_1_4 = hh[(ldh*3)+3];  // hh(4,4)
        double s_2_4 = hh[(ldh*3)+2];  // hh(3,4)
        double s_3_4 = hh[(ldh*3)+1];  // hh(2,4)
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
        s_1_2 += hh[2-1] * hh[(2+ldh)];               // s_1_2 = s_1_2 + hh(2,1) * hh(3,2)
        s_2_3 += hh[(ldh)+2-1] * hh[2+(ldh*2)];       // s_2_3 = s_2_3 + hh(2,2) * hh(3,3)
        s_3_4 += hh[(ldh*2)+2-1] * hh[2+(ldh*3)];     // s_3_4 = s_3_4 + hh(2,3) * hh(3,4)

        // loop counter = 3
        s_1_2 += hh[3-1] * hh[(3+ldh)];               // s_1_2 = s_1_2 + hh(3,1) * hh(4,2)
        s_2_3 += hh[(ldh)+3-1] * hh[3+(ldh*2)];       // s_2_3 = s_2_3 + hh(3,2) * hh(4,3)
        s_3_4 += hh[(ldh*2)+3-1] * hh[3+(ldh*3)];     // s_3_4 = s_3_4 + hh(3,3) * hh(4,4)

        s_1_3 += hh[3-2] * hh[3+(ldh*2)];             // s_1_3 = s_1_3 + hh(2,1) * hh(4,3)
        s_2_4 += hh[(ldh*1)+3-2] * hh[3+(ldh*3)];     // s_2_4 = s_2_4 + hh(2,2) * hh(4,4)

        #pragma ivdep
        for (i = 4; i < nb; i++)                            // do i = 5, nb
        {
                s_1_2 += hh[i-1] * hh[(i+ldh)];             // s_1_2 = s_1_2 + hh(i-1,1) * hh(i,2)
                s_2_3 += hh[(ldh)+i-1] * hh[i+(ldh*2)];     // s_2_3 = s_2_3 + hh(i-1,2) * hh(i,3)
                s_3_4 += hh[(ldh*2)+i-1] * hh[i+(ldh*3)];   // s_3_4 = s_3_4 + hh(i-1,3) * hh(i,4)

                s_1_3 += hh[i-2] * hh[i+(ldh*2)];           // s_1_3 = s_1_3 + hh(i-2,1) * hh(i,3)
                s_2_4 += hh[(ldh*1)+i-2] * hh[i+(ldh*3)];   // s_2_4 = s_2_4 + hh(i-2,2) * hh(i,4)

                s_1_4 += hh[i-3] * hh[i+(ldh*3)];           // s_1_4 = s_1_4 + hh(i-3,1) * hh(i,4)
        }

        // Production level kernel calls with padding
#ifdef DOUBLE_PRECISION_REAL
        for (i = 0; i < nq-4; i+=6)
        {
#ifdef HAVE_SSE_INTRINSICS
                hh_trafo_kernel_6_SSE_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
#endif
#ifdef HAVE_SPARC64_SSE
                hh_trafo_kernel_6_SPARC64_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
#endif
                worked_on += 6;
        }
#endif
#ifdef SINGLE_PRECISION_REAL
        for (i = 0; i < nq-8; i+=12)
        {
#ifdef HAVE_SSE_INTRINSICS
                hh_trafo_kernel_12_SSE_4hv_single(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
#endif
#ifdef HAVE_SPARC64_SSE
                hh_trafo_kernel_12_SPARC64_4hv_single(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
#endif

                worked_on += 12;

        }
#endif
        if (nq == i)
        {
                return;
        }

#ifdef DOUBLE_PRECISION_REAL
        if (nq-i ==4)
        {
#ifdef HAVE_SSE_INTRINSICS
                hh_trafo_kernel_4_SSE_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
#endif
#ifdef HAVE_SPARC64_SSE
                hh_trafo_kernel_4_SPARC64_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
#endif
                worked_on += 4;
        }
#endif

#ifdef SINGLE_PRECISION_REAL
        if (nq-i ==8)
        {
#ifdef HAVE_SSE_INTRINSICS
                hh_trafo_kernel_8_SSE_4hv_single(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
#endif
#ifdef HAVE_SPARC64_SSE
                hh_trafo_kernel_8_SPARC64_4hv_single(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
#endif
                worked_on += 8;
        }
#endif

#ifdef DOUBLE_PRECISION_REAL
        if (nq-i == 2)
        {
#ifdef HAVE_SSE_INTRINSICS
                hh_trafo_kernel_2_SSE_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
#endif
#ifdef HAVE_SPARC64_SSE
                hh_trafo_kernel_2_SPARC64_4hv_double(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
#endif

                worked_on += 2;
        }
#endif

#ifdef SINGLE_PRECISION_REAL
        if (nq-i ==4)
        {
#ifdef HAVE_SSE_INTRINSICS
                hh_trafo_kernel_4_SSE_4hv_single(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
#endif
#ifdef HAVE_SPARC64_SSE
                hh_trafo_kernel_4_SPARC64_4hv_single(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
#endif
                worked_on += 4;
        }
#endif
#ifdef WITH_DEBUG
        if (worked_on != nq)
        {
#ifdef HAVE_SSE_INTRINSICS
                printf("Error in real SSE BLOCK4 kernel \n");
#endif
#ifdef HAVE_SPARC64_SSE
                printf("Error in real SPARC64 BLOCK4 kernel \n");
#endif

                abort();
        }
#endif

}

/**
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 6 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 12 rows of Q simultaneously, a
#endif
 * matrix Vector product with two householder
 * vectors + a rank 1 update is performed
 */
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_6_SSE_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_12_SSE_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
#endif
#endif
#ifdef HAVE_SPARC64_SSE
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_6_SPARC64_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_12_SPARC64_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
#endif
#endif
{
        /////////////////////////////////////////////////////
        // Matrix Vector Multiplication, Q [6 x nb+3] * hh
        // hh contains four householder vectors
        /////////////////////////////////////////////////////
        int i;

        __SSE_DATATYPE a1_1 = _SSE_LOAD(&q[ldq*3]);   // a_1_1 = q(1,4)
        __SSE_DATATYPE a2_1 = _SSE_LOAD(&q[ldq*2]);   // a_2_1 = q(1,3)
        __SSE_DATATYPE a3_1 = _SSE_LOAD(&q[ldq]);     // a_3_1 = q(1,2)
        __SSE_DATATYPE a4_1 = _SSE_LOAD(&q[0]);       // a_4_1 = q(1,1)

#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE h_2_1 = _SSE_SET1(hh[ldh+1]);        // h_2_1 = hh(2,2)
        __SSE_DATATYPE h_3_2 = _SSE_SET1(hh[(ldh*2)+1]);    // h_3_2 = hh(2,3)
        __SSE_DATATYPE h_3_1 = _SSE_SET1(hh[(ldh*2)+2]);    // h_3_1 = hh(3,3)
        __SSE_DATATYPE h_4_3 = _SSE_SET1(hh[(ldh*3)+1]);    // h_4_3 = hh(2,4)
        __SSE_DATATYPE h_4_2 = _SSE_SET1(hh[(ldh*3)+2]);    // h_4_2 = hh(3,4)
        __SSE_DATATYPE h_4_1 = _SSE_SET1(hh[(ldh*3)+3]);    // h_4_1 = hh(4,4)
#endif

#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE h_2_1 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
        __SSE_DATATYPE h_3_2 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
        __SSE_DATATYPE h_3_1 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
        __SSE_DATATYPE h_4_3 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
        __SSE_DATATYPE h_4_2 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
        __SSE_DATATYPE h_4_1 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif



        register __SSE_DATATYPE w1 = _SSE_ADD(a4_1, _SSE_MUL(a3_1, h_4_3));   // w1 = a_4_1 + a_3_1 * h_4_3
        w1 = _SSE_ADD(w1, _SSE_MUL(a2_1, h_4_2));                             // w1 = w1 + a_2_1 * h_4_2
        w1 = _SSE_ADD(w1, _SSE_MUL(a1_1, h_4_1));                             // w1 = w1 + a_1_1 * h_4_1
        register __SSE_DATATYPE z1 = _SSE_ADD(a3_1, _SSE_MUL(a2_1, h_3_2));   // z1 = a_3_1 + a_2_1 * h_3_2
        z1 = _SSE_ADD(z1, _SSE_MUL(a1_1, h_3_1));                             // z1 = z1 + a_1_1 * h_3_1
        register __SSE_DATATYPE y1 = _SSE_ADD(a2_1, _SSE_MUL(a1_1, h_2_1));   // y1 = a_2_1 + a_1_1 * h_2_1 
        register __SSE_DATATYPE x1 = a1_1;                                    // x1 = a_1_1

        __SSE_DATATYPE a1_2 = _SSE_LOAD(&q[(ldq*3)+offset]);                  
        __SSE_DATATYPE a2_2 = _SSE_LOAD(&q[(ldq*2)+offset]);
        __SSE_DATATYPE a3_2 = _SSE_LOAD(&q[ldq+offset]);
        __SSE_DATATYPE a4_2 = _SSE_LOAD(&q[0+offset]);

        register __SSE_DATATYPE w2 = _SSE_ADD(a4_2, _SSE_MUL(a3_2, h_4_3));
        w2 = _SSE_ADD(w2, _SSE_MUL(a2_2, h_4_2));
        w2 = _SSE_ADD(w2, _SSE_MUL(a1_2, h_4_1));
        register __SSE_DATATYPE z2 = _SSE_ADD(a3_2, _SSE_MUL(a2_2, h_3_2));
        z2 = _SSE_ADD(z2, _SSE_MUL(a1_2, h_3_1));
        register __SSE_DATATYPE y2 = _SSE_ADD(a2_2, _SSE_MUL(a1_2, h_2_1));
        register __SSE_DATATYPE x2 = a1_2;

        __SSE_DATATYPE a1_3 = _SSE_LOAD(&q[(ldq*3)+2*offset]);
        __SSE_DATATYPE a2_3 = _SSE_LOAD(&q[(ldq*2)+2*offset]);
        __SSE_DATATYPE a3_3 = _SSE_LOAD(&q[ldq+2*offset]);
        __SSE_DATATYPE a4_3 = _SSE_LOAD(&q[0+2*offset]);

        register __SSE_DATATYPE w3 = _SSE_ADD(a4_3, _SSE_MUL(a3_3, h_4_3));
        w3 = _SSE_ADD(w3, _SSE_MUL(a2_3, h_4_2));
        w3 = _SSE_ADD(w3, _SSE_MUL(a1_3, h_4_1));
        register __SSE_DATATYPE z3 = _SSE_ADD(a3_3, _SSE_MUL(a2_3, h_3_2));
        z3 = _SSE_ADD(z3, _SSE_MUL(a1_3, h_3_1));
        register __SSE_DATATYPE y3 = _SSE_ADD(a2_3, _SSE_MUL(a1_3, h_2_1));
        register __SSE_DATATYPE x3 = a1_3;

        __SSE_DATATYPE q1;
        __SSE_DATATYPE q2;
        __SSE_DATATYPE q3;

        __SSE_DATATYPE h1;
        __SSE_DATATYPE h2;
        __SSE_DATATYPE h3;
        __SSE_DATATYPE h4;

        for(i = 4; i < nb; i++)                   // do i = 5, nb
        {
#ifdef HAVE_SSE_INTRINSICS
                h1 = _SSE_SET1(hh[i-3]);       // h1 = hh(i-3,1)
#endif

#ifdef HAVE_SPARC64_SSE
                h1 = _SSE_SET(hh[i-3], hh[i-3]);
#endif
                q1 = _SSE_LOAD(&q[i*ldq]);             // q(1:nq,i) carefull in fortran loop must start at higher index
                q2 = _SSE_LOAD(&q[(i*ldq)+offset]);
                q3 = _SSE_LOAD(&q[(i*ldq)+2*offset]);

                x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));    // x1 = x1 + q1(i,2) * h1
                x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
                x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));

#ifdef HAVE_SSE_INTRINSICS
                h2 = _SSE_SET1(hh[ldh+i-2]);          // h2 = hh(i-2,2)
#endif

#ifdef HAVE_SPARC64_SSE
                h2 = _SSE_SET(hh[ldh+i-2], hh[ldh+i-2]);
#endif
                y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));   // y1 = y1 + q1(i,2) * h2
                y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
                y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));

#ifdef HAVE_SSE_INTRINSICS
                h3 = _SSE_SET1(hh[(ldh*2)+i-1]);          // h3 = hh(i-1,3)
#endif

#ifdef HAVE_SPARC64_SSE
                h3 = _SSE_SET(hh[(ldh*2)+i-1], hh[(ldh*2)+i-1]);
#endif


                z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
                z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
                z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));
#ifdef HAVE_SSE_INTRINSICS
                h4 = _SSE_SET1(hh[(ldh*3)+i]);      // h4 = hh(i,4)
#endif

#ifdef HAVE_SPARC64_SSE
                h4 = _SSE_SET(hh[(ldh*3)+i], hh[(ldh*3)+i]);
#endif

                w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
                w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));
                w3 = _SSE_ADD(w3, _SSE_MUL(q3,h4));
        }

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-3]);  // h1 = hh(nb-2,1)
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-3], hh[nb-3]);
#endif
        q1 = _SSE_LOAD(&q[nb*ldq]);
        q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);
        q3 = _SSE_LOAD(&q[(nb*ldq)+2*offset]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));    // x1 = x1 + q1(1:nq,nb+1) * h1
        x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
        x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));

#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-2]);   // h2 = hh(nb-1,2)
#endif

#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

        y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));   // y1 = y1 + q1(1:nq,nb+1) * h2
        y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
        y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));

#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);   // h3 = hh(nb,3)
#endif

#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif

        z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));   // z1 = z1 + q1(1:nq,nb+1) * h3
        z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
        z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-2]);  // h1 = hh(nb-1,1)
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

        q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
        q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);
        q3 = _SSE_LOAD(&q[((nb+1)*ldq)+2*offset]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));    // x1 = x1 + q1(1:nq,nb+2) * h1
        x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
        x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));

#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[(ldh*1)+nb-1]);   // h2 = hh(nb, 2)
#endif

#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[(ldh*1)+nb-1], hh[(ldh*1)+nb-1]);
#endif


        y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));     // y1 = y1 + q1(1:nq,nb+2) * h2
        y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
        y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-1]);   // h1 = hh(nb,1)
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif


        q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
        q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);
        q3 = _SSE_LOAD(&q[((nb+2)*ldq)+2*offset]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));   // x1 = x1 + q1(1:nq,nb+3) * h1
        x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
        x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));

        /////////////////////////////////////////////////////
        // Rank-1 update of Q [6 x nb+3]
        /////////////////////////////////////////////////////

#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau1 = _SSE_SET1(hh[0]);   // tau1 = hh(1,1)
#endif

#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau1 = _SSE_SET(hh[0], hh[0]);
#endif


        h1 = tau1;
        x1 = _SSE_MUL(x1, h1);   // x1 = x1*h1
        x2 = _SSE_MUL(x2, h1);
        x3 = _SSE_MUL(x3, h1);

#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau2 = _SSE_SET1(hh[ldh]);
        __SSE_DATATYPE vs_1_2 = _SSE_SET1(s_1_2);
#endif

#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau2 = _SSE_SET(hh[ldh], hh[ldh]);
        __SSE_DATATYPE vs_1_2 = _SSE_SET(s_1_2, s_1_2);
#endif


        h1 = tau2;
        h2 = _SSE_MUL(h1, vs_1_2);   // h2 = h1 * vs_1_2

        y1 = _SSE_SUB(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));  // y1 = y1 * h1 - x1*h2
        y2 = _SSE_SUB(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
        y3 = _SSE_SUB(_SSE_MUL(y3,h1), _SSE_MUL(x3,h2));

#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau3 = _SSE_SET1(hh[ldh*2]);   // tau3 = hh(1,3)
        __SSE_DATATYPE vs_1_3 = _SSE_SET1(s_1_3);     // vs_1_3 = s_1_3
        __SSE_DATATYPE vs_2_3 = _SSE_SET1(s_2_3);     // vs_2_3 = s_2_3
#endif

#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau3 = _SSE_SET(hh[ldh*2], hh[ldh*2]);
        __SSE_DATATYPE vs_1_3 = _SSE_SET(s_1_3, s_1_3);
        __SSE_DATATYPE vs_2_3 = _SSE_SET(s_2_3, s_2_3);
#endif


        h1 = tau3;
        h2 = _SSE_MUL(h1, vs_1_3);
        h3 = _SSE_MUL(h1, vs_2_3);

        z1 = _SSE_SUB(_SSE_MUL(z1,h1), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)));   // z1 = z1 * h1 - (y1 * h3 + x1 * h2)
        z2 = _SSE_SUB(_SSE_MUL(z2,h1), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2)));
        z3 = _SSE_SUB(_SSE_MUL(z3,h1), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2)));

#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau4 = _SSE_SET1(hh[ldh*3]);    // tau4 = hh(1,4)
        __SSE_DATATYPE vs_1_4 = _SSE_SET1(s_1_4);      // vs_1_4 = s_1_4
        __SSE_DATATYPE vs_2_4 = _SSE_SET1(s_2_4);      // vs_2_4 = s_2_4
        __SSE_DATATYPE vs_3_4 = _SSE_SET1(s_3_4);      // vs_3_4 = s_3_4
#endif

#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau4 = _SSE_SET(hh[ldh*3], hh[ldh*3]);
        __SSE_DATATYPE vs_1_4 = _SSE_SET(s_1_4, s_1_4);
        __SSE_DATATYPE vs_2_4 = _SSE_SET(s_2_4, s_2_4);
        __SSE_DATATYPE vs_3_4 = _SSE_SET(s_3_4, s_3_4);
#endif

        h1 = tau4;
        h2 = _SSE_MUL(h1, vs_1_4);
        h3 = _SSE_MUL(h1, vs_2_4);
        h4 = _SSE_MUL(h1, vs_3_4);

        w1 = _SSE_SUB(_SSE_MUL(w1,h1), _SSE_ADD(_SSE_MUL(z1,h4), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));   // w1 = w1 * h1 - (z1 *h4 + ( y1 * h3 + x1 *h2))
        w2 = _SSE_SUB(_SSE_MUL(w2,h1), _SSE_ADD(_SSE_MUL(z2,h4), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2))));
        w3 = _SSE_SUB(_SSE_MUL(w3,h1), _SSE_ADD(_SSE_MUL(z3,h4), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2))));

        q1 = _SSE_LOAD(&q[0]);
        q2 = _SSE_LOAD(&q[offset]);
        q3 = _SSE_LOAD(&q[2*offset]);
        q1 = _SSE_SUB(q1, w1);    // q1 = q1(1:nq,1) - w1
        q2 = _SSE_SUB(q2, w2);
        q3 = _SSE_SUB(q3, w3);
        _SSE_STORE(&q[0],q1);
        _SSE_STORE(&q[offset],q2);
        _SSE_STORE(&q[2*offset],q3);

#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+1]);    // h4 = hh(2,4)
#endif

#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif


        q1 = _SSE_LOAD(&q[ldq]);
        q2 = _SSE_LOAD(&q[ldq+offset]);
        q3 = _SSE_LOAD(&q[ldq+2*offset]);

        q1 = _SSE_SUB(q1, _SSE_ADD(z1, _SSE_MUL(w1, h4)));    // q1(1:nq,2) = q1(1:nq,2) - (z1 + w1 *h4)
        q2 = _SSE_SUB(q2, _SSE_ADD(z2, _SSE_MUL(w2, h4)));
        q3 = _SSE_SUB(q3, _SSE_ADD(z3, _SSE_MUL(w3, h4)));

        _SSE_STORE(&q[ldq],q1);
        _SSE_STORE(&q[ldq+offset],q2);
        _SSE_STORE(&q[ldq+2*offset],q3);

#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+2]);   // h4 = hh(3,4)
#endif

#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

        q1 = _SSE_LOAD(&q[ldq*2]);
        q2 = _SSE_LOAD(&q[(ldq*2)+offset]);
        q3 = _SSE_LOAD(&q[(ldq*2)+2*offset]);
        q1 = _SSE_SUB(q1, y1);                   // q1(1:nq,3) = q1(1:nq,3) - y1
        q2 = _SSE_SUB(q2, y2);
        q3 = _SSE_SUB(q3, y3);

        q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));     // q1(1:nq,3) = q1(1:nq,3) - w1 *h4
        q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
        q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));

#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+1]);   // h3 = hh(2,3)
#endif

#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif

        q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));   // q1(1:nq,3) = q(1:nq,3) - (z1 * h3)
        q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
        q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));

        _SSE_STORE(&q[ldq*2],q1);
        _SSE_STORE(&q[(ldq*2)+offset],q2);
        _SSE_STORE(&q[(ldq*2)+2*offset],q3);

#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+3]);      // h4 = hh(4,4)
#endif

#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

        q1 = _SSE_LOAD(&q[ldq*3]);
        q2 = _SSE_LOAD(&q[(ldq*3)+offset]);
        q3 = _SSE_LOAD(&q[(ldq*3)+2*offset]);
        q1 = _SSE_SUB(q1, x1);                  // q1(1:nq,4) = q1(1:nq,4) - x1
        q2 = _SSE_SUB(q2, x2);
        q3 = _SSE_SUB(q3, x3);

        q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));    // q1(1:nq,4) = q1(1:nq,4) - w1 * h4
        q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
        q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));

#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+1]);    // h2 = hh(2,2)
#endif

#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));   // q1(1:nq,4) = q1(1:nq,4) - y1 * h2
        q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
        q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));

#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+2]);   // h3 = hh(3,3)
#endif

#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));   // q1(1:nq,4) = q1(1:nq,4) - z1 * h3
        q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
        q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
        _SSE_STORE(&q[ldq*3], q1);
        _SSE_STORE(&q[(ldq*3)+offset], q2);
        _SSE_STORE(&q[(ldq*3)+2*offset], q3);

        for (i = 4; i < nb; i++)                // do 5 = 1, nb
        {
#ifdef HAVE_SSE_INTRINSICS
                h1 = _SSE_SET1(hh[i-3]);     // h1 = hh(i-3,1)
#endif

#ifdef HAVE_SPARC64_SSE
                h1 = _SSE_SET(hh[i-3], hh[i-3]);
#endif

                q1 = _SSE_LOAD(&q[i*ldq]);
                q2 = _SSE_LOAD(&q[(i*ldq)+offset]);
                q3 = _SSE_LOAD(&q[(i*ldq)+2*offset]);

                q1 = _SSE_SUB(q1, _SSE_MUL(x1,h1));     // q1(1:nq,i) = q1(1:nq,i) - x1 * h1
                q2 = _SSE_SUB(q2, _SSE_MUL(x2,h1));
                q3 = _SSE_SUB(q3, _SSE_MUL(x3,h1));

#ifdef HAVE_SSE_INTRINSICS
                h2 = _SSE_SET1(hh[ldh+i-2]);    // h2 = hh(i-2,2)
#endif

#ifdef HAVE_SPARC64_SSE
                h2 = _SSE_SET(hh[ldh+i-2], hh[ldh+i-2]);
#endif

                q1 = _SSE_SUB(q1, _SSE_MUL(y1,h2));     // q1(1:nq,i) = q1(1:nq,i) - y1 * h2 
                q2 = _SSE_SUB(q2, _SSE_MUL(y2,h2));
                q3 = _SSE_SUB(q3, _SSE_MUL(y3,h2));

#ifdef HAVE_SSE_INTRINSICS
                h3 = _SSE_SET1(hh[(ldh*2)+i-1]);     // h3 = hh(i-1,3)
#endif

#ifdef HAVE_SPARC64_SSE
                h3 = _SSE_SET(hh[(ldh*2)+i-1], hh[(ldh*2)+i-1]);
#endif

                q1 = _SSE_SUB(q1, _SSE_MUL(z1,h3));   // q1(1:nq,i) = q1(1:nq,i) - z1 * h3
                q2 = _SSE_SUB(q2, _SSE_MUL(z2,h3));
                q3 = _SSE_SUB(q3, _SSE_MUL(z3,h3));

#ifdef HAVE_SSE_INTRINSICS
                h4 = _SSE_SET1(hh[(ldh*3)+i]);      // h4 = hh(i,4)
#endif

#ifdef HAVE_SPRC64_SSE
                h4 = _SSE_SET(hh[(ldh*3)+i], hh[(ldh*3)+i]);
#endif

                q1 = _SSE_SUB(q1, _SSE_MUL(w1,h4));   // q1(1:nq,i) = q1(1:nq,i) - w1 * h4
                q2 = _SSE_SUB(q2, _SSE_MUL(w2,h4));
                q3 = _SSE_SUB(q3, _SSE_MUL(w3,h4));

                _SSE_STORE(&q[i*ldq],q1);
                _SSE_STORE(&q[(i*ldq)+offset],q2);
                _SSE_STORE(&q[(i*ldq)+2*offset],q3);
        }

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-3]);    // h1 = hh(nb-2,1)
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-3], hh[nb-3]);
#endif

        q1 = _SSE_LOAD(&q[nb*ldq]);
        q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);
        q3 = _SSE_LOAD(&q[(nb*ldq)+2*offset]);

        q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));  // q1(1:nq,nb+1) = q1(1:nq,nb+1) - x1 * h1
        q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
        q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));

#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-2]);  // h2 = hh(nb-1,2)
#endif

#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));  // q1(1:nq,nb+1) = q1(1:nq,nb+1) - y1 * h2
        q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
        q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));

#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);   // h3 = hh(nb,3)
#endif

#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif


        q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));   //  q1(1:nq,nb+1) = q1(1:nq,nb+1) - z1 * h3
        q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
        q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));

        _SSE_STORE(&q[nb*ldq],q1);
        _SSE_STORE(&q[(nb*ldq)+offset],q2);
        _SSE_STORE(&q[(nb*ldq)+2*offset],q3);

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

        q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
        q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);
        q3 = _SSE_LOAD(&q[((nb+1)*ldq)+2*offset]);

        q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));  // q1(1:nq,nb+2) = q1(1:nq,nb+2) - x1 * h1
        q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
        q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));

#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+nb-1]);  // h2 = hh(nb,2)
#endif

#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif

        q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));    // q1(1:nq,nb+2) = q1(1:nq,nb+2) - y1 * h2
        q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
        q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));

        _SSE_STORE(&q[(nb+1)*ldq],q1);
        _SSE_STORE(&q[((nb+1)*ldq)+offset],q2);
        _SSE_STORE(&q[((nb+1)*ldq)+2*offset],q3);

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-1]);     // h1 = hh(nb,1)
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif

        q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
        q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);
        q3 = _SSE_LOAD(&q[((nb+2)*ldq)+2*offset]);

        q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));    // q1(1:nq,nb+3) = q1(1:nq,nb+3) - x1 * h1
        q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
        q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));

        _SSE_STORE(&q[(nb+2)*ldq],q1);
        _SSE_STORE(&q[((nb+2)*ldq)+offset],q2);
        _SSE_STORE(&q[((nb+2)*ldq)+2*offset],q3);
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
__forceinline void hh_trafo_kernel_4_SSE_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_8_SSE_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
#endif
#endif
#ifdef HAVE_SPARC64_SSE
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_4_SPARC64_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_8_SPARC64_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
#endif
#endif

{
        /////////////////////////////////////////////////////
        // Matrix Vector Multiplication, Q [4 x nb+3] * hh
        // hh contains four householder vectors
        /////////////////////////////////////////////////////
        int i;

        __SSE_DATATYPE a1_1 = _SSE_LOAD(&q[ldq*3]);
        __SSE_DATATYPE a2_1 = _SSE_LOAD(&q[ldq*2]);
        __SSE_DATATYPE a3_1 = _SSE_LOAD(&q[ldq]);
        __SSE_DATATYPE a4_1 = _SSE_LOAD(&q[0]);

#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE h_2_1 = _SSE_SET1(hh[ldh+1]);
        __SSE_DATATYPE h_3_2 = _SSE_SET1(hh[(ldh*2)+1]);
        __SSE_DATATYPE h_3_1 = _SSE_SET1(hh[(ldh*2)+2]);
        __SSE_DATATYPE h_4_3 = _SSE_SET1(hh[(ldh*3)+1]);
        __SSE_DATATYPE h_4_2 = _SSE_SET1(hh[(ldh*3)+2]);
        __SSE_DATATYPE h_4_1 = _SSE_SET1(hh[(ldh*3)+3]);
#endif

#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE h_2_1 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
        __SSE_DATATYPE h_3_2 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
        __SSE_DATATYPE h_3_1 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
        __SSE_DATATYPE h_4_3 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
        __SSE_DATATYPE h_4_2 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
        __SSE_DATATYPE h_4_1 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif


        __SSE_DATATYPE w1 = _SSE_ADD(a4_1, _SSE_MUL(a3_1, h_4_3));
        w1 = _SSE_ADD(w1, _SSE_MUL(a2_1, h_4_2));
        w1 = _SSE_ADD(w1, _SSE_MUL(a1_1, h_4_1));
        __SSE_DATATYPE z1 = _SSE_ADD(a3_1, _SSE_MUL(a2_1, h_3_2));
        z1 = _SSE_ADD(z1, _SSE_MUL(a1_1, h_3_1));
        __SSE_DATATYPE y1 = _SSE_ADD(a2_1, _SSE_MUL(a1_1, h_2_1));
        __SSE_DATATYPE x1 = a1_1;

        __SSE_DATATYPE a1_2 = _SSE_LOAD(&q[(ldq*3)+offset]);
        __SSE_DATATYPE a2_2 = _SSE_LOAD(&q[(ldq*2)+offset]);
        __SSE_DATATYPE a3_2 = _SSE_LOAD(&q[ldq+offset]);
        __SSE_DATATYPE a4_2 = _SSE_LOAD(&q[0+offset]);

        __SSE_DATATYPE w2 = _SSE_ADD(a4_2, _SSE_MUL(a3_2, h_4_3));
        w2 = _SSE_ADD(w2, _SSE_MUL(a2_2, h_4_2));
        w2 = _SSE_ADD(w2, _SSE_MUL(a1_2, h_4_1));
        __SSE_DATATYPE z2 = _SSE_ADD(a3_2, _SSE_MUL(a2_2, h_3_2));
        z2 = _SSE_ADD(z2, _SSE_MUL(a1_2, h_3_1));
        __SSE_DATATYPE y2 = _SSE_ADD(a2_2, _SSE_MUL(a1_2, h_2_1));
        __SSE_DATATYPE x2 = a1_2;

        __SSE_DATATYPE q1;
        __SSE_DATATYPE q2;

        __SSE_DATATYPE h1;
        __SSE_DATATYPE h2;
        __SSE_DATATYPE h3;
        __SSE_DATATYPE h4;

        for(i = 4; i < nb; i++)
        {

#ifdef HAVE_SSE_INTRINSICS
                h1 = _SSE_SET1(hh[i-3]);
                h2 = _SSE_SET1(hh[ldh+i-2]);
                h3 = _SSE_SET1(hh[(ldh*2)+i-1]);
                h4 = _SSE_SET1(hh[(ldh*3)+i]);
#endif

#ifdef HAVE_SPARC64_SSE
                h1 = _SSE_SET(hh[i-3], hh[i-3]);
                h2 = _SSE_SET(hh[ldh+i-2], hh[ldh+i-2]);
                h3 = _SSE_SET(hh[(ldh*2)+i-1], hh[(ldh*2)+i-1]);
                h4 = _SSE_SET(hh[(ldh*3)+i], hh[(ldh*3)+i]);
#endif

                q1 = _SSE_LOAD(&q[i*ldq]);

                x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
                y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
                z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
                w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));

                q2 = _SSE_LOAD(&q[(i*ldq)+offset]);

                x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
                y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
                z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
                w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));
        }

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-3]);
        h2 = _SSE_SET1(hh[ldh+nb-2]);
        h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-3], hh[nb-3]);
        h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
        h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif

        q1 = _SSE_LOAD(&q[nb*ldq]);
        q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
        x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
        y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
        y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
        z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
        z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-2]);
        h2 = _SSE_SET1(hh[(ldh*1)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
        h2 = _SSE_SET(hh[(ldh*1)+nb-1], hh[(ldh*1)+nb-1]);
#endif


        q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
        q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
        x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
        y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
        y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif


        q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
        q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
        x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));

        /////////////////////////////////////////////////////
        // Rank-1 update of Q [4 x nb+3]
        /////////////////////////////////////////////////////

#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau1 = _SSE_SET1(hh[0]);
        __SSE_DATATYPE tau2 = _SSE_SET1(hh[ldh]);
        __SSE_DATATYPE tau3 = _SSE_SET1(hh[ldh*2]);
        __SSE_DATATYPE tau4 = _SSE_SET1(hh[ldh*3]);

        __SSE_DATATYPE vs_1_2 = _SSE_SET1(s_1_2);
        __SSE_DATATYPE vs_1_3 = _SSE_SET1(s_1_3);
        __SSE_DATATYPE vs_2_3 = _SSE_SET1(s_2_3);
        __SSE_DATATYPE vs_1_4 = _SSE_SET1(s_1_4);
        __SSE_DATATYPE vs_2_4 = _SSE_SET1(s_2_4);
        __SSE_DATATYPE vs_3_4 = _SSE_SET1(s_3_4);
#endif

#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau1 = _SSE_SET(hh[0], hh[0]);
        __SSE_DATATYPE tau2 = _SSE_SET(hh[ldh], hh[ldh]);
        __SSE_DATATYPE tau3 = _SSE_SET(hh[ldh*2], hh[ldh*2]);
        __SSE_DATATYPE tau4 = _SSE_SET(hh[ldh*3], hh[ldh*3]);

        __SSE_DATATYPE vs_1_2 = _SSE_SET(s_1_2, s_1_2);
        __SSE_DATATYPE vs_1_3 = _SSE_SET(s_1_3, s_1_3);
        __SSE_DATATYPE vs_2_3 = _SSE_SET(s_2_3, s_2_3);
        __SSE_DATATYPE vs_1_4 = _SSE_SET(s_1_4, s_1_4);
        __SSE_DATATYPE vs_2_4 = _SSE_SET(s_2_4, s_2_4);
        __SSE_DATATYPE vs_3_4 = _SSE_SET(s_3_4, s_3_4);
#endif


        h1 = tau1;
        x1 = _SSE_MUL(x1, h1);
        x2 = _SSE_MUL(x2, h1);

        h1 = tau2;
        h2 = _SSE_MUL(h1, vs_1_2);

        y1 = _SSE_SUB(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
        y2 = _SSE_SUB(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));

        h1 = tau3;
        h2 = _SSE_MUL(h1, vs_1_3);
        h3 = _SSE_MUL(h1, vs_2_3);

        z1 = _SSE_SUB(_SSE_MUL(z1,h1), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)));
        z2 = _SSE_SUB(_SSE_MUL(z2,h1), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2)));

        h1 = tau4;
        h2 = _SSE_MUL(h1, vs_1_4);
        h3 = _SSE_MUL(h1, vs_2_4);
        h4 = _SSE_MUL(h1, vs_3_4);

        w1 = _SSE_SUB(_SSE_MUL(w1,h1), _SSE_ADD(_SSE_MUL(z1,h4), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));
        w2 = _SSE_SUB(_SSE_MUL(w2,h1), _SSE_ADD(_SSE_MUL(z2,h4), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2))));

        q1 = _SSE_LOAD(&q[0]);
        q2 = _SSE_LOAD(&q[offset]);
        q1 = _SSE_SUB(q1, w1);
        q2 = _SSE_SUB(q2, w2);
        _SSE_STORE(&q[0],q1);
        _SSE_STORE(&q[offset],q2);

#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+1]);
#endif

#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

        q1 = _SSE_LOAD(&q[ldq]);
        q2 = _SSE_LOAD(&q[ldq+offset]);

        q1 = _SSE_SUB(q1, _SSE_ADD(z1, _SSE_MUL(w1, h4)));
        q2 = _SSE_SUB(q2, _SSE_ADD(z2, _SSE_MUL(w2, h4)));

        _SSE_STORE(&q[ldq],q1);
        _SSE_STORE(&q[ldq+offset],q2);

#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+1]);
        h4 = _SSE_SET1(hh[(ldh*3)+2]);
#endif

#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
        h4 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif


        q1 = _SSE_LOAD(&q[ldq*2]);
        q2 = _SSE_LOAD(&q[(ldq*2)+offset]);

        q1 = _SSE_SUB(q1, _SSE_ADD(y1, _SSE_ADD(_SSE_MUL(z1, h3), _SSE_MUL(w1, h4))));
        q2 = _SSE_SUB(q2, _SSE_ADD(y2, _SSE_ADD(_SSE_MUL(z2, h3), _SSE_MUL(w2, h4))));
        _SSE_STORE(&q[ldq*2],q1);
        _SSE_STORE(&q[(ldq*2)+offset],q2);

#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+1]);
        h3 = _SSE_SET1(hh[(ldh*2)+2]);
        h4 = _SSE_SET1(hh[(ldh*3)+3]);
#endif

#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
        h3 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
        h4 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

        q1 = _SSE_LOAD(&q[ldq*3]);
        q2 = _SSE_LOAD(&q[(ldq*3)+offset]);

        q1 = _SSE_SUB(q1, _SSE_ADD(x1, _SSE_ADD(_SSE_MUL(y1, h2), _SSE_ADD(_SSE_MUL(z1, h3), _SSE_MUL(w1, h4)))));
        q2 = _SSE_SUB(q2, _SSE_ADD(x2, _SSE_ADD(_SSE_MUL(y2, h2), _SSE_ADD(_SSE_MUL(z2, h3), _SSE_MUL(w2, h4)))));

        _SSE_STORE(&q[ldq*3], q1);
        _SSE_STORE(&q[(ldq*3)+offset], q2);

        for (i = 4; i < nb; i++)
        {
#ifdef HAVE_SSE_INTRINSICS
                h1 = _SSE_SET1(hh[i-3]);
                h2 = _SSE_SET1(hh[ldh+i-2]);
                h3 = _SSE_SET1(hh[(ldh*2)+i-1]);
                h4 = _SSE_SET1(hh[(ldh*3)+i]);
#endif

#ifdef HAVE_SPARC64_SSE
                h1 = _SSE_SET(hh[i-3], hh[i-3]);
                h2 = _SSE_SET(hh[ldh+i-2], hh[ldh+i-2]);
                h3 = _SSE_SET(hh[(ldh*2)+i-1], hh[(ldh*2)+i-1]);
                h4 = _SSE_SET(hh[(ldh*3)+i], hh[(ldh*3)+i]);
#endif


                q1 = _SSE_LOAD(&q[i*ldq]);

                q1 = _SSE_SUB(q1, _SSE_ADD(_SSE_ADD(_SSE_MUL(w1, h4), _SSE_MUL(z1, h3)), _SSE_ADD(_SSE_MUL(x1,h1), _SSE_MUL(y1, h2))));

                _SSE_STORE(&q[i*ldq],q1);

                q2 = _SSE_LOAD(&q[(i*ldq)+offset]);

                q2 = _SSE_SUB(q2, _SSE_ADD(_SSE_ADD(_SSE_MUL(w2, h4), _SSE_MUL(z2, h3)), _SSE_ADD(_SSE_MUL(x2,h1), _SSE_MUL(y2, h2))));

                _SSE_STORE(&q[(i*ldq)+offset],q2);
        }

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-3]);
        h2 = _SSE_SET1(hh[ldh+nb-2]);
        h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-3], hh[nb-3]);
        h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
        h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif


        q1 = _SSE_LOAD(&q[nb*ldq]);
        q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);

        q1 = _SSE_SUB(q1, _SSE_ADD(_SSE_ADD(_SSE_MUL(z1, h3), _SSE_MUL(y1, h2)) , _SSE_MUL(x1, h1)));
        q2 = _SSE_SUB(q2, _SSE_ADD(_SSE_ADD(_SSE_MUL(z2, h3), _SSE_MUL(y2, h2)) , _SSE_MUL(x2, h1)));

        _SSE_STORE(&q[nb*ldq],q1);
        _SSE_STORE(&q[(nb*ldq)+offset],q2);

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-2]);
        h2 = _SSE_SET1(hh[ldh+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
        h2 = _SSE_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif


        q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
        q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);

        q1 = _SSE_SUB(q1, _SSE_ADD( _SSE_MUL(y1, h2) , _SSE_MUL(x1, h1)));
        q2 = _SSE_SUB(q2, _SSE_ADD( _SSE_MUL(y2, h2) , _SSE_MUL(x2, h1)));

        _SSE_STORE(&q[(nb+1)*ldq],q1);
        _SSE_STORE(&q[((nb+1)*ldq)+offset],q2);

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif

        q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
        q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);

        q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
        q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));

        _SSE_STORE(&q[(nb+2)*ldq],q1);
        _SSE_STORE(&q[((nb+2)*ldq)+offset],q2);
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
__forceinline void hh_trafo_kernel_2_SSE_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_4_SSE_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
#endif
#endif
#ifdef HAVE_SPARC64_SSE
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_2_SPARC64_4hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s_1_2, double s_1_3, double s_2_3, double s_1_4, double s_2_4, double s_3_4)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_4_SPARC64_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
#endif
#endif

{
        /////////////////////////////////////////////////////
        // Matrix Vector Multiplication, Q [2 x nb+3] * hh
        // hh contains four householder vectors
        /////////////////////////////////////////////////////
        int i;

        __SSE_DATATYPE a1_1 = _SSE_LOAD(&q[ldq*3]);
        __SSE_DATATYPE a2_1 = _SSE_LOAD(&q[ldq*2]);
        __SSE_DATATYPE a3_1 = _SSE_LOAD(&q[ldq]);
        __SSE_DATATYPE a4_1 = _SSE_LOAD(&q[0]);

#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE h_2_1 = _SSE_SET1(hh[ldh+1]);
        __SSE_DATATYPE h_3_2 = _SSE_SET1(hh[(ldh*2)+1]);
        __SSE_DATATYPE h_3_1 = _SSE_SET1(hh[(ldh*2)+2]);
        __SSE_DATATYPE h_4_3 = _SSE_SET1(hh[(ldh*3)+1]);
        __SSE_DATATYPE h_4_2 = _SSE_SET1(hh[(ldh*3)+2]);
        __SSE_DATATYPE h_4_1 = _SSE_SET1(hh[(ldh*3)+3]);
#endif

#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE h_2_1 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
        __SSE_DATATYPE h_3_2 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
        __SSE_DATATYPE h_3_1 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
        __SSE_DATATYPE h_4_3 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
        __SSE_DATATYPE h_4_2 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
        __SSE_DATATYPE h_4_1 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

        __SSE_DATATYPE w1 = _SSE_ADD(a4_1, _SSE_MUL(a3_1, h_4_3));
        w1 = _SSE_ADD(w1, _SSE_MUL(a2_1, h_4_2));
        w1 = _SSE_ADD(w1, _SSE_MUL(a1_1, h_4_1));
        __SSE_DATATYPE z1 = _SSE_ADD(a3_1, _SSE_MUL(a2_1, h_3_2));
        z1 = _SSE_ADD(z1, _SSE_MUL(a1_1, h_3_1));
        __SSE_DATATYPE y1 = _SSE_ADD(a2_1, _SSE_MUL(a1_1, h_2_1));
        __SSE_DATATYPE x1 = a1_1;

        __SSE_DATATYPE q1;

        __SSE_DATATYPE h1;
        __SSE_DATATYPE h2;
        __SSE_DATATYPE h3;
        __SSE_DATATYPE h4;

        for(i = 4; i < nb; i++)
        {
#ifdef HAVE_SSE_INTRINSICS
                h1 = _SSE_SET1(hh[i-3]);
                h2 = _SSE_SET1(hh[ldh+i-2]);
                h3 = _SSE_SET1(hh[(ldh*2)+i-1]);
                h4 = _SSE_SET1(hh[(ldh*3)+i]);
#endif

#ifdef HAVE_SPARC64_SSE
                h1 = _SSE_SET(hh[i-3], hh[i-3]);
                h2 = _SSE_SET(hh[ldh+i-2], hh[ldh+i-2]);
                h3 = _SSE_SET(hh[(ldh*2)+i-1], hh[(ldh*2)+i-1]);
                h4 = _SSE_SET(hh[(ldh*3)+i], hh[(ldh*3)+i]);
#endif

                q1 = _SSE_LOAD(&q[i*ldq]);

                x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
                y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
                z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
                w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
        }

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-3]);
        h2 = _SSE_SET1(hh[ldh+nb-2]);
        h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-3], hh[nb-3]);
        h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
        h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif

        q1 = _SSE_LOAD(&q[nb*ldq]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
        y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
        z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-2]);
        h2 = _SSE_SET1(hh[(ldh*1)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
        h2 = _SSE_SET(hh[(ldh*1)+nb-1], hh[(ldh*1)+nb-1]);
#endif

        q1 = _SSE_LOAD(&q[(nb+1)*ldq]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
        y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif

        q1 = _SSE_LOAD(&q[(nb+2)*ldq]);

        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
        /////////////////////////////////////////////////////
        // Rank-1 update of Q [2 x nb+3]
        /////////////////////////////////////////////////////

#ifdef HAVE_SSE_INTRINSICS
        __SSE_DATATYPE tau1 = _SSE_SET1(hh[0]);
        __SSE_DATATYPE tau2 = _SSE_SET1(hh[ldh]);
        __SSE_DATATYPE tau3 = _SSE_SET1(hh[ldh*2]);
        __SSE_DATATYPE tau4 = _SSE_SET1(hh[ldh*3]);

        __SSE_DATATYPE vs_1_2 = _SSE_SET1(s_1_2);
        __SSE_DATATYPE vs_1_3 = _SSE_SET1(s_1_3);
        __SSE_DATATYPE vs_2_3 = _SSE_SET1(s_2_3);
        __SSE_DATATYPE vs_1_4 = _SSE_SET1(s_1_4);
        __SSE_DATATYPE vs_2_4 = _SSE_SET1(s_2_4);
        __SSE_DATATYPE vs_3_4 = _SSE_SET1(s_3_4);
#endif

#ifdef HAVE_SPARC64_SSE
        __SSE_DATATYPE tau1 = _SSE_SET(hh[0], hh[0]);
        __SSE_DATATYPE tau2 = _SSE_SET(hh[ldh], hh[ldh]);
        __SSE_DATATYPE tau3 = _SSE_SET(hh[ldh*2], hh[ldh*2]);
        __SSE_DATATYPE tau4 = _SSE_SET(hh[ldh*3], hh[ldh*3]);

        __SSE_DATATYPE vs_1_2 = _SSE_SET(s_1_2, s_1_2);
        __SSE_DATATYPE vs_1_3 = _SSE_SET(s_1_3, s_1_3);
        __SSE_DATATYPE vs_2_3 = _SSE_SET(s_2_3, s_2_3);
        __SSE_DATATYPE vs_1_4 = _SSE_SET(s_1_4, s_1_4);
        __SSE_DATATYPE vs_2_4 = _SSE_SET(s_2_4, s_2_4);
        __SSE_DATATYPE vs_3_4 = _SSE_SET(s_3_4, s_3_4);
#endif


        h1 = tau1;
        x1 = _SSE_MUL(x1, h1);

        h1 = tau2;
        h2 = _SSE_MUL(h1, vs_1_2);

        y1 = _SSE_SUB(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));

        h1 = tau3;
        h2 = _SSE_MUL(h1, vs_1_3);
        h3 = _SSE_MUL(h1, vs_2_3);

        z1 = _SSE_SUB(_SSE_MUL(z1,h1), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)));

        h1 = tau4;
        h2 = _SSE_MUL(h1, vs_1_4);
        h3 = _SSE_MUL(h1, vs_2_4);
        h4 = _SSE_MUL(h1, vs_3_4);

        w1 = _SSE_SUB(_SSE_MUL(w1,h1), _SSE_ADD(_SSE_MUL(z1,h4), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));

        q1 = _SSE_LOAD(&q[0]);
        q1 = _SSE_SUB(q1, w1);
        _SSE_STORE(&q[0],q1);

#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+1]);
#endif

#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

        q1 = _SSE_LOAD(&q[ldq]);

        q1 = _SSE_SUB(q1, _SSE_ADD(z1, _SSE_MUL(w1, h4)));

        _SSE_STORE(&q[ldq],q1);

#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+1]);
        h4 = _SSE_SET1(hh[(ldh*3)+2]);
#endif

#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
        h4 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

        q1 = _SSE_LOAD(&q[ldq*2]);

        q1 = _SSE_SUB(q1, _SSE_ADD(y1, _SSE_ADD(_SSE_MUL(z1, h3), _SSE_MUL(w1, h4))));

        _SSE_STORE(&q[ldq*2],q1);

#ifdef HAVE_SSE_INTRINSICS
        h2 = _SSE_SET1(hh[ldh+1]);
        h3 = _SSE_SET1(hh[(ldh*2)+2]);
        h4 = _SSE_SET1(hh[(ldh*3)+3]);
#endif

#ifdef HAVE_SPARC64_SSE
        h2 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
        h3 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
        h4 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

        q1 = _SSE_LOAD(&q[ldq*3]);

        q1 = _SSE_SUB(q1, _SSE_ADD(x1, _SSE_ADD(_SSE_MUL(y1, h2), _SSE_ADD(_SSE_MUL(z1, h3), _SSE_MUL(w1, h4)))));

        _SSE_STORE(&q[ldq*3], q1);

        for (i = 4; i < nb; i++)
        {
#ifdef HAVE_SSE_INTRINSICS
                h1 = _SSE_SET1(hh[i-3]);
                h2 = _SSE_SET1(hh[ldh+i-2]);
                h3 = _SSE_SET1(hh[(ldh*2)+i-1]);
                h4 = _SSE_SET1(hh[(ldh*3)+i]);
#endif

#ifdef HAVE_SPARC64_SSE
                h1 = _SSE_SET(hh[i-3], hh[i-3]);
                h2 = _SSE_SET(hh[ldh+i-2], hh[ldh+i-2]);
                h3 = _SSE_SET(hh[(ldh*2)+i-1], hh[(ldh*2)+i-1]);
                h4 = _SSE_SET(hh[(ldh*3)+i], hh[(ldh*3)+i]);
#endif

                q1 = _SSE_LOAD(&q[i*ldq]);

                q1 = _SSE_SUB(q1, _SSE_ADD(_SSE_ADD(_SSE_MUL(w1, h4), _SSE_MUL(z1, h3)), _SSE_ADD(_SSE_MUL(x1,h1), _SSE_MUL(y1, h2))));

                _SSE_STORE(&q[i*ldq],q1);
        }

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-3]);
        h2 = _SSE_SET1(hh[ldh+nb-2]);
        h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-3], hh[nb-3]);
        h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
        h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif


        q1 = _SSE_LOAD(&q[nb*ldq]);
        q1 = _SSE_SUB(q1, _SSE_ADD(_SSE_ADD(_SSE_MUL(z1, h3), _SSE_MUL(y1, h2)) , _SSE_MUL(x1, h1)));

        _SSE_STORE(&q[nb*ldq],q1);

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-2]);
        h2 = _SSE_SET1(hh[ldh+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
        h2 = _SSE_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif

        q1 = _SSE_LOAD(&q[(nb+1)*ldq]);

        q1 = _SSE_SUB(q1, _SSE_ADD( _SSE_MUL(y1, h2) , _SSE_MUL(x1, h1)));

        _SSE_STORE(&q[(nb+1)*ldq],q1);

#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif

        q1 = _SSE_LOAD(&q[(nb+2)*ldq]);

        q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));

        _SSE_STORE(&q[(nb+2)*ldq],q1);
}
