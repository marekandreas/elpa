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
#define _AVX512_ADD _mm512_add_pd
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
#define _AVX512_ADD _mm512_add_ps
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



//Forward declaration
#ifdef DOUBLE_PRECISION_REAL
static void hh_trafo_kernel_8_AVX512_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);
static void hh_trafo_kernel_16_AVX512_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);
static void hh_trafo_kernel_24_AVX512_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);
static void hh_trafo_kernel_32_AVX512_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);

void hexa_hh_trafo_real_avx512_6hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif

#ifdef SINGLE_PRECISION_REAL
static void hh_trafo_kernel_16_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);
static void hh_trafo_kernel_32_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);
static void hh_trafo_kernel_48_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);
static void hh_trafo_kernel_64_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods);

void hexa_hh_trafo_real_avx512_6hv_single_(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);

#endif

/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine hexa_hh_trafo_real_avx512_6hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="hexa_hh_trafo_real_avx512_6hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_double)     :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/


/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine hexa_hh_trafo_real_avx512_6hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="hexa_hh_trafo_real_avx512_6hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)      :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/


#ifdef DOUBLE_PRECISION_REAL
void hexa_hh_trafo_real_avx512_6hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
#ifdef SINGLE_PRECISION_REAL
void hexa_hh_trafo_real_avx512_6hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
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
        for (i = 0; i < nq-24; i+=32)
        {
                hh_trafo_kernel_32_AVX512_6hv_double(&q[i], hh, nb, ldq, ldh, scalarprods);
                worked_on += 32;
        }
#endif
#ifdef SINGLE_PRECISION_REAL
        for (i = 0; i < nq-48; i+=64)
        {
                hh_trafo_kernel_64_AVX512_6hv_single(&q[i], hh, nb, ldq, ldh, scalarprods);
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
                hh_trafo_kernel_24_AVX512_6hv_double(&q[i], hh, nb, ldq, ldh, scalarprods);
                worked_on += 24;
        }
#endif

#ifdef SINGLE_PRECISION_REAL
        if (nq-i ==48)
        {
                hh_trafo_kernel_48_AVX512_6hv_single(&q[i], hh, nb, ldq, ldh, scalarprods);
                worked_on += 48;
        }
#endif

#ifdef DOUBLE_PRECISION_REAL
        if (nq-i == 16)
        {
                hh_trafo_kernel_16_AVX512_6hv_double(&q[i], hh, nb, ldq, ldh, scalarprods);
                worked_on += 16;
        }
#endif

#ifdef SINGLE_PRECISION_REAL
        if (nq-i ==32)
        {
                hh_trafo_kernel_32_AVX512_6hv_single(&q[i], hh, nb, ldq, ldh, scalarprods);
                worked_on += 32;
        }
#endif

#ifdef DOUBLE_PRECISION_REAL
        if (nq-i == 8)
        {
                hh_trafo_kernel_8_AVX512_6hv_double(&q[i], hh, nb, ldq, ldh, scalarprods);
                worked_on += 8;
        }
#endif

#ifdef SINGLE_PRECISION_REAL
        if (nq-i == 16)
        {
                hh_trafo_kernel_16_AVX512_6hv_single(&q[i], hh, nb, ldq, ldh, scalarprods);
                worked_on += 16;
        }
#endif

#ifdef WITH_DEBUG
        if (worked_on != nq)
        {
          printf("ERROR in avx512 kernel\n");
          abort();
        }
#endif
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
__forceinline void hh_trafo_kernel_8_AVX512_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_16_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods)
#endif

{
        /////////////////////////////////////////////////////
        // Matrix Vector Multiplication, Q [8 x nb+3] * hh
        // hh contains four householder vectors
        /////////////////////////////////////////////////////
        int i;

        __AVX512_DATATYPE a1_1 = _AVX512_LOAD(&q[ldq*5]);
        __AVX512_DATATYPE a2_1 = _AVX512_LOAD(&q[ldq*4]);
        __AVX512_DATATYPE a3_1 = _AVX512_LOAD(&q[ldq*3]);
        __AVX512_DATATYPE a4_1 = _AVX512_LOAD(&q[ldq*2]);
        __AVX512_DATATYPE a5_1 = _AVX512_LOAD(&q[ldq]);
        __AVX512_DATATYPE a6_1 = _AVX512_LOAD(&q[0]);

        __AVX512_DATATYPE h_6_5 = _AVX512_SET1(hh[(ldh*5)+1]);
        __AVX512_DATATYPE h_6_4 = _AVX512_SET1(hh[(ldh*5)+2]);
        __AVX512_DATATYPE h_6_3 = _AVX512_SET1(hh[(ldh*5)+3]);
        __AVX512_DATATYPE h_6_2 = _AVX512_SET1(hh[(ldh*5)+4]);
        __AVX512_DATATYPE h_6_1 = _AVX512_SET1(hh[(ldh*5)+5]);

//        register __AVX512_DATATYPE t1 = _AVX512_FMA(a5_1, h_6_5, a6_1);
        __AVX512_DATATYPE t1 = _AVX512_FMA(a5_1, h_6_5, a6_1);

        t1 = _AVX512_FMA(a4_1, h_6_4, t1);
        t1 = _AVX512_FMA(a3_1, h_6_3, t1);
        t1 = _AVX512_FMA(a2_1, h_6_2, t1);
        t1 = _AVX512_FMA(a1_1, h_6_1, t1);

        __AVX512_DATATYPE h_5_4 = _AVX512_SET1(hh[(ldh*4)+1]);
        __AVX512_DATATYPE h_5_3 = _AVX512_SET1(hh[(ldh*4)+2]);
        __AVX512_DATATYPE h_5_2 = _AVX512_SET1(hh[(ldh*4)+3]);
        __AVX512_DATATYPE h_5_1 = _AVX512_SET1(hh[(ldh*4)+4]);

//        register __AVX512_DATATYPE v1 = _AVX512_FMA(a4_1, h_5_4, a5_1);
        __AVX512_DATATYPE v1 = _AVX512_FMA(a4_1, h_5_4, a5_1);

        v1 = _AVX512_FMA(a3_1, h_5_3, v1);
        v1 = _AVX512_FMA(a2_1, h_5_2, v1);
        v1 = _AVX512_FMA(a1_1, h_5_1, v1);

        __AVX512_DATATYPE h_4_3 = _AVX512_SET1(hh[(ldh*3)+1]);
        __AVX512_DATATYPE h_4_2 = _AVX512_SET1(hh[(ldh*3)+2]);
        __AVX512_DATATYPE h_4_1 = _AVX512_SET1(hh[(ldh*3)+3]);

//        register __AVX512_DATATYPE w1 = _AVX512_FMA(a3_1, h_4_3, a4_1);
        __AVX512_DATATYPE w1 = _AVX512_FMA(a3_1, h_4_3, a4_1);

        w1 = _AVX512_FMA(a2_1, h_4_2, w1);
        w1 = _AVX512_FMA(a1_1, h_4_1, w1);

        __AVX512_DATATYPE h_2_1 = _AVX512_SET1(hh[ldh+1]);
        __AVX512_DATATYPE h_3_2 = _AVX512_SET1(hh[(ldh*2)+1]);
        __AVX512_DATATYPE h_3_1 = _AVX512_SET1(hh[(ldh*2)+2]);

//        register __AVX512_DATATYPE z1 = _AVX512_FMA(a2_1, h_3_2, a3_1);
        __AVX512_DATATYPE z1 = _AVX512_FMA(a2_1, h_3_2, a3_1);

        z1 = _AVX512_FMA(a1_1, h_3_1, z1);
//        register __AVX512_DATATYPE y1 = _AVX512_FMA(a1_1, h_2_1, a2_1);
         __AVX512_DATATYPE y1 = _AVX512_FMA(a1_1, h_2_1, a2_1);

//        register __AVX512_DATATYPE x1 = a1_1;
        __AVX512_DATATYPE x1 = a1_1;

        __AVX512_DATATYPE q1;

        __AVX512_DATATYPE h1;
        __AVX512_DATATYPE h2;
        __AVX512_DATATYPE h3;
        __AVX512_DATATYPE h4;
        __AVX512_DATATYPE h5;
        __AVX512_DATATYPE h6;

        for(i = 6; i < nb; i++)
        {
                h1 = _AVX512_SET1(hh[i-5]);
                q1 = _AVX512_LOAD(&q[i*ldq]);

                x1 = _AVX512_FMA(q1, h1, x1);

                h2 = _AVX512_SET1(hh[ldh+i-4]);

                y1 = _AVX512_FMA(q1, h2, y1);
                h3 = _AVX512_SET1(hh[(ldh*2)+i-3]);

                z1 = _AVX512_FMA(q1, h3, z1);

                h4 = _AVX512_SET1(hh[(ldh*3)+i-2]);

                w1 = _AVX512_FMA(q1, h4, w1);

                h5 = _AVX512_SET1(hh[(ldh*4)+i-1]);

                v1 = _AVX512_FMA(q1, h5, v1);

                h6 = _AVX512_SET1(hh[(ldh*5)+i]);

                t1 = _AVX512_FMA(q1, h6, t1);
        }

        h1 = _AVX512_SET1(hh[nb-5]);
        q1 = _AVX512_LOAD(&q[nb*ldq]);

        x1 = _AVX512_FMA(q1, h1, x1);

        h2 = _AVX512_SET1(hh[ldh+nb-4]);

        y1 = _AVX512_FMA(q1, h2, y1);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-3]);

        z1 = _AVX512_FMA(q1, h3, z1);

        h4 = _AVX512_SET1(hh[(ldh*3)+nb-2]);

        w1 = _AVX512_FMA(q1, h4, w1);

        h5 = _AVX512_SET1(hh[(ldh*4)+nb-1]);

        v1 = _AVX512_FMA(q1, h5, v1);

        h1 = _AVX512_SET1(hh[nb-4]);

        q1 = _AVX512_LOAD(&q[(nb+1)*ldq]);

        x1 = _AVX512_FMA(q1, h1, x1);

        h2 = _AVX512_SET1(hh[ldh+nb-3]);

        y1 = _AVX512_FMA(q1, h2, y1);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-2]);

        z1 = _AVX512_FMA(q1, h3, z1);

        h4 = _AVX512_SET1(hh[(ldh*3)+nb-1]);

        w1 = _AVX512_FMA(q1, h4, w1);

        h1 = _AVX512_SET1(hh[nb-3]);
        q1 = _AVX512_LOAD(&q[(nb+2)*ldq]);

        x1 = _AVX512_FMA(q1, h1, x1);

        h2 = _AVX512_SET1(hh[ldh+nb-2]);

        y1 = _AVX512_FMA(q1, h2, y1);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-1]);

        z1 = _AVX512_FMA(q1, h3, z1);

        h1 = _AVX512_SET1(hh[nb-2]);
        q1 = _AVX512_LOAD(&q[(nb+3)*ldq]);

        x1 = _AVX512_FMA(q1, h1, x1);

        h2 = _AVX512_SET1(hh[ldh+nb-1]);

        y1 = _AVX512_FMA(q1, h2, y1);

        h1 = _AVX512_SET1(hh[nb-1]);
        q1 = _AVX512_LOAD(&q[(nb+4)*ldq]);

        x1 = _AVX512_FMA(q1, h1, x1);

        /////////////////////////////////////////////////////
        // Apply tau, correct wrong calculation using pre-calculated scalar products
        /////////////////////////////////////////////////////

        __AVX512_DATATYPE tau1 = _AVX512_SET1(hh[0]);
        x1 = _AVX512_MUL(x1, tau1);

        __AVX512_DATATYPE tau2 = _AVX512_SET1(hh[ldh]);
        __AVX512_DATATYPE vs_1_2 = _AVX512_SET1(scalarprods[0]);
        h2 = _AVX512_MUL(tau2, vs_1_2);

        y1 = _AVX512_FMSUB(y1, tau2, _AVX512_MUL(x1,h2));

        __AVX512_DATATYPE tau3 = _AVX512_SET1(hh[ldh*2]);
        __AVX512_DATATYPE vs_1_3 = _AVX512_SET1(scalarprods[1]);
        __AVX512_DATATYPE vs_2_3 = _AVX512_SET1(scalarprods[2]);

        h2 = _AVX512_MUL(tau3, vs_1_3);
        h3 = _AVX512_MUL(tau3, vs_2_3);

        z1 = _AVX512_FMSUB(z1, tau3, _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2)));

        __AVX512_DATATYPE tau4 = _AVX512_SET1(hh[ldh*3]);
        __AVX512_DATATYPE vs_1_4 = _AVX512_SET1(scalarprods[3]);
        __AVX512_DATATYPE vs_2_4 = _AVX512_SET1(scalarprods[4]);

        h2 = _AVX512_MUL(tau4, vs_1_4);
        h3 = _AVX512_MUL(tau4, vs_2_4);

        __AVX512_DATATYPE vs_3_4 = _AVX512_SET1(scalarprods[5]);
        h4 = _AVX512_MUL(tau4, vs_3_4);

        w1 = _AVX512_FMSUB(w1, tau4, _AVX512_FMA(z1, h4, _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2))));

        __AVX512_DATATYPE tau5 = _AVX512_SET1(hh[ldh*4]);
        __AVX512_DATATYPE vs_1_5 = _AVX512_SET1(scalarprods[6]);
        __AVX512_DATATYPE vs_2_5 = _AVX512_SET1(scalarprods[7]);

        h2 = _AVX512_MUL(tau5, vs_1_5);
        h3 = _AVX512_MUL(tau5, vs_2_5);

        __AVX512_DATATYPE vs_3_5 = _AVX512_SET1(scalarprods[8]);
        __AVX512_DATATYPE vs_4_5 = _AVX512_SET1(scalarprods[9]);

        h4 = _AVX512_MUL(tau5, vs_3_5);
        h5 = _AVX512_MUL(tau5, vs_4_5);

        v1 = _AVX512_FMSUB(v1, tau5, _AVX512_ADD(_AVX512_FMA(w1, h5, _AVX512_MUL(z1,h4)), _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2))));

        __AVX512_DATATYPE tau6 = _AVX512_SET1(hh[ldh*5]);
        __AVX512_DATATYPE vs_1_6 = _AVX512_SET1(scalarprods[10]);
        __AVX512_DATATYPE vs_2_6 = _AVX512_SET1(scalarprods[11]);
        h2 = _AVX512_MUL(tau6, vs_1_6);
        h3 = _AVX512_MUL(tau6, vs_2_6);

        __AVX512_DATATYPE vs_3_6 = _AVX512_SET1(scalarprods[12]);
        __AVX512_DATATYPE vs_4_6 = _AVX512_SET1(scalarprods[13]);
        __AVX512_DATATYPE vs_5_6 = _AVX512_SET1(scalarprods[14]);

        h4 = _AVX512_MUL(tau6, vs_3_6);
        h5 = _AVX512_MUL(tau6, vs_4_6);
        h6 = _AVX512_MUL(tau6, vs_5_6);

        t1 = _AVX512_FMSUB(t1, tau6, _AVX512_FMA(v1, h6, _AVX512_ADD(_AVX512_FMA(w1, h5, _AVX512_MUL(z1,h4)), _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2)))));

        /////////////////////////////////////////////////////
        // Rank-1 update of Q [8 x nb+3]
        /////////////////////////////////////////////////////

        q1 = _AVX512_LOAD(&q[0]);
        q1 = _AVX512_SUB(q1, t1);
        _AVX512_STORE(&q[0],q1);

        h6 = _AVX512_SET1(hh[(ldh*5)+1]);
        q1 = _AVX512_LOAD(&q[ldq]);
        q1 = _AVX512_SUB(q1, v1);

        q1 = _AVX512_NFMA(t1, h6, q1);

        _AVX512_STORE(&q[ldq],q1);

        h5 = _AVX512_SET1(hh[(ldh*4)+1]);
        q1 = _AVX512_LOAD(&q[ldq*2]);
        q1 = _AVX512_SUB(q1, w1);

        q1 = _AVX512_NFMA(v1, h5, q1);

        h6 = _AVX512_SET1(hh[(ldh*5)+2]);

        q1 = _AVX512_NFMA(t1, h6, q1);

        _AVX512_STORE(&q[ldq*2],q1);

        h4 = _AVX512_SET1(hh[(ldh*3)+1]);
        q1 = _AVX512_LOAD(&q[ldq*3]);
        q1 = _AVX512_SUB(q1, z1);

        q1 = _AVX512_NFMA(w1, h4, q1);

        h5 = _AVX512_SET1(hh[(ldh*4)+2]);

        q1 = _AVX512_NFMA(v1, h5, q1);

        h6 = _AVX512_SET1(hh[(ldh*5)+3]);

        q1 = _AVX512_NFMA(t1, h6, q1);

        _AVX512_STORE(&q[ldq*3],q1);

        h3 = _AVX512_SET1(hh[(ldh*2)+1]);
        q1 = _AVX512_LOAD(&q[ldq*4]);
        q1 = _AVX512_SUB(q1, y1);

        q1 = _AVX512_NFMA(z1, h3, q1);

        h4 = _AVX512_SET1(hh[(ldh*3)+2]);

        q1 = _AVX512_NFMA(w1, h4, q1);

        h5 = _AVX512_SET1(hh[(ldh*4)+3]);

        q1 = _AVX512_NFMA(v1, h5, q1);

        h6 = _AVX512_SET1(hh[(ldh*5)+4]);

        q1 = _AVX512_NFMA(t1, h6, q1);

        _AVX512_STORE(&q[ldq*4],q1);

        h2 = _AVX512_SET1(hh[(ldh)+1]);
        q1 = _AVX512_LOAD(&q[ldq*5]);
        q1 = _AVX512_SUB(q1, x1);

        q1 = _AVX512_NFMA(y1, h2, q1);

        h3 = _AVX512_SET1(hh[(ldh*2)+2]);

        q1 = _AVX512_NFMA(z1, h3, q1);

        h4 = _AVX512_SET1(hh[(ldh*3)+3]);

        q1 = _AVX512_NFMA(w1, h4, q1);

        h5 = _AVX512_SET1(hh[(ldh*4)+4]);

        q1 = _AVX512_NFMA(v1, h5, q1);

        h6 = _AVX512_SET1(hh[(ldh*5)+5]);

        q1 = _AVX512_NFMA(t1, h6, q1);

        _AVX512_STORE(&q[ldq*5],q1);

        for (i = 6; i < nb; i++)
        {
                q1 = _AVX512_LOAD(&q[i*ldq]);
                h1 = _AVX512_SET1(hh[i-5]);

                q1 = _AVX512_NFMA(x1, h1, q1);

                h2 = _AVX512_SET1(hh[ldh+i-4]);

                q1 = _AVX512_NFMA(y1, h2, q1);

                h3 = _AVX512_SET1(hh[(ldh*2)+i-3]);

                q1 = _AVX512_NFMA(z1, h3, q1);

                h4 = _AVX512_SET1(hh[(ldh*3)+i-2]);

                q1 = _AVX512_NFMA(w1, h4, q1);

                h5 = _AVX512_SET1(hh[(ldh*4)+i-1]);

                q1 = _AVX512_NFMA(v1, h5, q1);

                h6 = _AVX512_SET1(hh[(ldh*5)+i]);

                q1 = _AVX512_NFMA(t1, h6, q1);

                _AVX512_STORE(&q[i*ldq],q1);
        }

        h1 = _AVX512_SET1(hh[nb-5]);
        q1 = _AVX512_LOAD(&q[nb*ldq]);

        q1 = _AVX512_NFMA(x1, h1, q1);

        h2 = _AVX512_SET1(hh[ldh+nb-4]);

        q1 = _AVX512_NFMA(y1, h2, q1);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-3]);

        q1 = _AVX512_NFMA(z1, h3, q1);

        h4 = _AVX512_SET1(hh[(ldh*3)+nb-2]);

        q1 = _AVX512_NFMA(w1, h4, q1);

        h5 = _AVX512_SET1(hh[(ldh*4)+nb-1]);

        q1 = _AVX512_NFMA(v1, h5, q1);

        _AVX512_STORE(&q[nb*ldq],q1);

        h1 = _AVX512_SET1(hh[nb-4]);
        q1 = _AVX512_LOAD(&q[(nb+1)*ldq]);

        q1 = _AVX512_NFMA(x1, h1, q1);

        h2 = _AVX512_SET1(hh[ldh+nb-3]);

        q1 = _AVX512_NFMA(y1, h2, q1);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-2]);

        q1 = _AVX512_NFMA(z1, h3, q1);

        h4 = _AVX512_SET1(hh[(ldh*3)+nb-1]);

        q1 = _AVX512_NFMA(w1, h4, q1);

        _AVX512_STORE(&q[(nb+1)*ldq],q1);

        h1 = _AVX512_SET1(hh[nb-3]);
        q1 = _AVX512_LOAD(&q[(nb+2)*ldq]);

        q1 = _AVX512_NFMA(x1, h1, q1);

        h2 = _AVX512_SET1(hh[ldh+nb-2]);

        q1 = _AVX512_NFMA(y1, h2, q1);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-1]);

        q1 = _AVX512_NFMA(z1, h3, q1);

        _AVX512_STORE(&q[(nb+2)*ldq],q1);

        h1 = _AVX512_SET1(hh[nb-2]);
        q1 = _AVX512_LOAD(&q[(nb+3)*ldq]);

        q1 = _AVX512_NFMA(x1, h1, q1);

        h2 = _AVX512_SET1(hh[ldh+nb-1]);

        q1 = _AVX512_NFMA(y1, h2, q1);

        _AVX512_STORE(&q[(nb+3)*ldq],q1);

        h1 = _AVX512_SET1(hh[nb-1]);
        q1 = _AVX512_LOAD(&q[(nb+4)*ldq]);

        q1 = _AVX512_NFMA(x1, h1, q1);

        _AVX512_STORE(&q[(nb+4)*ldq],q1);
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
__forceinline void hh_trafo_kernel_16_AVX512_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_32_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods)
#endif
{
        /////////////////////////////////////////////////////
        // Matrix Vector Multiplication, Q [8 x nb+3] * hh
        // hh contains four householder vectors
        /////////////////////////////////////////////////////
        int i;

        __AVX512_DATATYPE a1_1 = _AVX512_LOAD(&q[ldq*5]);
        __AVX512_DATATYPE a2_1 = _AVX512_LOAD(&q[ldq*4]);
        __AVX512_DATATYPE a3_1 = _AVX512_LOAD(&q[ldq*3]);
        __AVX512_DATATYPE a4_1 = _AVX512_LOAD(&q[ldq*2]);
        __AVX512_DATATYPE a5_1 = _AVX512_LOAD(&q[ldq]);
        __AVX512_DATATYPE a6_1 = _AVX512_LOAD(&q[0]);

        __AVX512_DATATYPE h_6_5 = _AVX512_SET1(hh[(ldh*5)+1]);
        __AVX512_DATATYPE h_6_4 = _AVX512_SET1(hh[(ldh*5)+2]);
        __AVX512_DATATYPE h_6_3 = _AVX512_SET1(hh[(ldh*5)+3]);
        __AVX512_DATATYPE h_6_2 = _AVX512_SET1(hh[(ldh*5)+4]);
        __AVX512_DATATYPE h_6_1 = _AVX512_SET1(hh[(ldh*5)+5]);

//        register__AVX512_DATATYPE t1 = _AVX512_FMA(a5_1, h_6_5, a6_1);
        __AVX512_DATATYPE t1 = _AVX512_FMA(a5_1, h_6_5, a6_1);

        t1 = _AVX512_FMA(a4_1, h_6_4, t1);
        t1 = _AVX512_FMA(a3_1, h_6_3, t1);
        t1 = _AVX512_FMA(a2_1, h_6_2, t1);
        t1 = _AVX512_FMA(a1_1, h_6_1, t1);

        __AVX512_DATATYPE h_5_4 = _AVX512_SET1(hh[(ldh*4)+1]);
        __AVX512_DATATYPE h_5_3 = _AVX512_SET1(hh[(ldh*4)+2]);
        __AVX512_DATATYPE h_5_2 = _AVX512_SET1(hh[(ldh*4)+3]);
        __AVX512_DATATYPE h_5_1 = _AVX512_SET1(hh[(ldh*4)+4]);

//        register __AVX512_DATATYPE v1 = _AVX512_FMA(a4_1, h_5_4, a5_1);
        __AVX512_DATATYPE v1 = _AVX512_FMA(a4_1, h_5_4, a5_1);

        v1 = _AVX512_FMA(a3_1, h_5_3, v1);
        v1 = _AVX512_FMA(a2_1, h_5_2, v1);
        v1 = _AVX512_FMA(a1_1, h_5_1, v1);

        __AVX512_DATATYPE h_4_3 = _AVX512_SET1(hh[(ldh*3)+1]);
        __AVX512_DATATYPE h_4_2 = _AVX512_SET1(hh[(ldh*3)+2]);
        __AVX512_DATATYPE h_4_1 = _AVX512_SET1(hh[(ldh*3)+3]);

//        register __AVX512_DATATYPE w1 = _AVX512_FMA(a3_1, h_4_3, a4_1);
        __AVX512_DATATYPE w1 = _AVX512_FMA(a3_1, h_4_3, a4_1);

        w1 = _AVX512_FMA(a2_1, h_4_2, w1);
        w1 = _AVX512_FMA(a1_1, h_4_1, w1);

        __AVX512_DATATYPE h_2_1 = _AVX512_SET1(hh[ldh+1]);
        __AVX512_DATATYPE h_3_2 = _AVX512_SET1(hh[(ldh*2)+1]);
        __AVX512_DATATYPE h_3_1 = _AVX512_SET1(hh[(ldh*2)+2]);

//        register __AVX512_DATATYPE z1 = _AVX512_FMA(a2_1, h_3_2, a3_1);
        __AVX512_DATATYPE z1 = _AVX512_FMA(a2_1, h_3_2, a3_1);

        z1 = _AVX512_FMA(a1_1, h_3_1, z1);
//        register __AVX512_DATATYPE y1 = _AVX512_FMA(a1_1, h_2_1, a2_1);
        __AVX512_DATATYPE y1 = _AVX512_FMA(a1_1, h_2_1, a2_1);


//        register __AVX512_DATATYPE x1 = a1_1;
        __AVX512_DATATYPE x1 = a1_1;


        __AVX512_DATATYPE a1_2 = _AVX512_LOAD(&q[(ldq*5)+offset]);
        __AVX512_DATATYPE a2_2 = _AVX512_LOAD(&q[(ldq*4)+offset]);
        __AVX512_DATATYPE a3_2 = _AVX512_LOAD(&q[(ldq*3)+offset]);
        __AVX512_DATATYPE a4_2 = _AVX512_LOAD(&q[(ldq*2)+offset]);
        __AVX512_DATATYPE a5_2 = _AVX512_LOAD(&q[(ldq)+offset]);
        __AVX512_DATATYPE a6_2 = _AVX512_LOAD(&q[0+offset]);

//        register __AVX512_DATATYPE t2 = _AVX512_FMA(a5_2, h_6_5, a6_2);
        __AVX512_DATATYPE t2 = _AVX512_FMA(a5_2, h_6_5, a6_2);

        t2 = _AVX512_FMA(a4_2, h_6_4, t2);
        t2 = _AVX512_FMA(a3_2, h_6_3, t2);
        t2 = _AVX512_FMA(a2_2, h_6_2, t2);
        t2 = _AVX512_FMA(a1_2, h_6_1, t2);

//        register __AVX512_DATATYPE v2 = _AVX512_FMA(a4_2, h_5_4, a5_2);
        __AVX512_DATATYPE v2 = _AVX512_FMA(a4_2, h_5_4, a5_2);

        v2 = _AVX512_FMA(a3_2, h_5_3, v2);
        v2 = _AVX512_FMA(a2_2, h_5_2, v2);
        v2 = _AVX512_FMA(a1_2, h_5_1, v2);

//        register __AVX512_DATATYPE w2 = _AVX512_FMA(a3_2, h_4_3, a4_2);
        __AVX512_DATATYPE w2 = _AVX512_FMA(a3_2, h_4_3, a4_2);

        w2 = _AVX512_FMA(a2_2, h_4_2, w2);
        w2 = _AVX512_FMA(a1_2, h_4_1, w2);

//        register __AVX512_DATATYPE z2 = _AVX512_FMA(a2_2, h_3_2, a3_2);
        __AVX512_DATATYPE z2 = _AVX512_FMA(a2_2, h_3_2, a3_2);

        z2 = _AVX512_FMA(a1_2, h_3_1, z2);
//        register __AVX512_DATATYPE y2 = _AVX512_FMA(a1_2, h_2_1, a2_2);
        __AVX512_DATATYPE y2 = _AVX512_FMA(a1_2, h_2_1, a2_2);


//        register __AVX512_DATATYPE x2 = a1_2;
        __AVX512_DATATYPE x2 = a1_2;

        __AVX512_DATATYPE q1;
        __AVX512_DATATYPE q2;

        __AVX512_DATATYPE h1;
        __AVX512_DATATYPE h2;
        __AVX512_DATATYPE h3;
        __AVX512_DATATYPE h4;
        __AVX512_DATATYPE h5;
        __AVX512_DATATYPE h6;

        for(i = 6; i < nb; i++)
        {
                h1 = _AVX512_SET1(hh[i-5]);
                q1 = _AVX512_LOAD(&q[i*ldq]);
                q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);

                x1 = _AVX512_FMA(q1, h1, x1);
                x2 = _AVX512_FMA(q2, h1, x2);

                h2 = _AVX512_SET1(hh[ldh+i-4]);

                y1 = _AVX512_FMA(q1, h2, y1);
                y2 = _AVX512_FMA(q2, h2, y2);

                h3 = _AVX512_SET1(hh[(ldh*2)+i-3]);

                z1 = _AVX512_FMA(q1, h3, z1);
                z2 = _AVX512_FMA(q2, h3, z2);

                h4 = _AVX512_SET1(hh[(ldh*3)+i-2]);

                w1 = _AVX512_FMA(q1, h4, w1);
                w2 = _AVX512_FMA(q2, h4, w2);

                h5 = _AVX512_SET1(hh[(ldh*4)+i-1]);

                v1 = _AVX512_FMA(q1, h5, v1);
                v2 = _AVX512_FMA(q2, h5, v2);

                h6 = _AVX512_SET1(hh[(ldh*5)+i]);

                t1 = _AVX512_FMA(q1, h6, t1);
                t2 = _AVX512_FMA(q2, h6, t2);
        }

        h1 = _AVX512_SET1(hh[nb-5]);
        q1 = _AVX512_LOAD(&q[nb*ldq]);
        q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);

        x1 = _AVX512_FMA(q1, h1, x1);
        x2 = _AVX512_FMA(q2, h1, x2);

        h2 = _AVX512_SET1(hh[ldh+nb-4]);

        y1 = _AVX512_FMA(q1, h2, y1);
        y2 = _AVX512_FMA(q2, h2, y2);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-3]);

        z1 = _AVX512_FMA(q1, h3, z1);
        z2 = _AVX512_FMA(q2, h3, z2);

        h4 = _AVX512_SET1(hh[(ldh*3)+nb-2]);

        w1 = _AVX512_FMA(q1, h4, w1);
        w2 = _AVX512_FMA(q2, h4, w2);

        h5 = _AVX512_SET1(hh[(ldh*4)+nb-1]);

        v1 = _AVX512_FMA(q1, h5, v1);
        v2 = _AVX512_FMA(q2, h5, v2);

        h1 = _AVX512_SET1(hh[nb-4]);

        q1 = _AVX512_LOAD(&q[(nb+1)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+1)*ldq)+offset]);

        x1 = _AVX512_FMA(q1, h1, x1);
        x2 = _AVX512_FMA(q2, h1, x2);

        h2 = _AVX512_SET1(hh[ldh+nb-3]);

        y1 = _AVX512_FMA(q1, h2, y1);
        y2 = _AVX512_FMA(q2, h2, y2);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-2]);

        z1 = _AVX512_FMA(q1, h3, z1);
        z2 = _AVX512_FMA(q2, h3, z2);

        h4 = _AVX512_SET1(hh[(ldh*3)+nb-1]);

        w1 = _AVX512_FMA(q1, h4, w1);
        w2 = _AVX512_FMA(q2, h4, w2);

        h1 = _AVX512_SET1(hh[nb-3]);
        q1 = _AVX512_LOAD(&q[(nb+2)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+2)*ldq)+offset]);

        x1 = _AVX512_FMA(q1, h1, x1);
        x2 = _AVX512_FMA(q2, h1, x2);

        h2 = _AVX512_SET1(hh[ldh+nb-2]);

        y1 = _AVX512_FMA(q1, h2, y1);
        y2 = _AVX512_FMA(q2, h2, y2);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-1]);

        z1 = _AVX512_FMA(q1, h3, z1);
        z2 = _AVX512_FMA(q2, h3, z2);

        h1 = _AVX512_SET1(hh[nb-2]);
        q1 = _AVX512_LOAD(&q[(nb+3)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+3)*ldq)+offset]);

        x1 = _AVX512_FMA(q1, h1, x1);
        x2 = _AVX512_FMA(q2, h1, x2);

        h2 = _AVX512_SET1(hh[ldh+nb-1]);

        y1 = _AVX512_FMA(q1, h2, y1);
        y2 = _AVX512_FMA(q2, h2, y2);

        h1 = _AVX512_SET1(hh[nb-1]);
        q1 = _AVX512_LOAD(&q[(nb+4)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+4)*ldq)+offset]);

        x1 = _AVX512_FMA(q1, h1, x1);
        x2 = _AVX512_FMA(q2, h1, x2);

        /////////////////////////////////////////////////////
        // Apply tau, correct wrong calculation using pre-calculated scalar products
        /////////////////////////////////////////////////////

        __AVX512_DATATYPE tau1 = _AVX512_SET1(hh[0]);
        x1 = _AVX512_MUL(x1, tau1);
        x2 = _AVX512_MUL(x2, tau1);

        __AVX512_DATATYPE tau2 = _AVX512_SET1(hh[ldh]);
        __AVX512_DATATYPE vs_1_2 = _AVX512_SET1(scalarprods[0]);
        h2 = _AVX512_MUL(tau2, vs_1_2);

        y1 = _AVX512_FMSUB(y1, tau2, _AVX512_MUL(x1,h2));
        y2 = _AVX512_FMSUB(y2, tau2, _AVX512_MUL(x2,h2));

        __AVX512_DATATYPE tau3 = _AVX512_SET1(hh[ldh*2]);
        __AVX512_DATATYPE vs_1_3 = _AVX512_SET1(scalarprods[1]);
        __AVX512_DATATYPE vs_2_3 = _AVX512_SET1(scalarprods[2]);

        h2 = _AVX512_MUL(tau3, vs_1_3);
        h3 = _AVX512_MUL(tau3, vs_2_3);

        z1 = _AVX512_FMSUB(z1, tau3, _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2)));
        z2 = _AVX512_FMSUB(z2, tau3, _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2)));

        __AVX512_DATATYPE tau4 = _AVX512_SET1(hh[ldh*3]);
        __AVX512_DATATYPE vs_1_4 = _AVX512_SET1(scalarprods[3]);
        __AVX512_DATATYPE vs_2_4 = _AVX512_SET1(scalarprods[4]);

        h2 = _AVX512_MUL(tau4, vs_1_4);
        h3 = _AVX512_MUL(tau4, vs_2_4);

        __AVX512_DATATYPE vs_3_4 = _AVX512_SET1(scalarprods[5]);
        h4 = _AVX512_MUL(tau4, vs_3_4);

        w1 = _AVX512_FMSUB(w1, tau4, _AVX512_FMA(z1, h4, _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2))));
        w2 = _AVX512_FMSUB(w2, tau4, _AVX512_FMA(z2, h4, _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2))));

        __AVX512_DATATYPE tau5 = _AVX512_SET1(hh[ldh*4]);
        __AVX512_DATATYPE vs_1_5 = _AVX512_SET1(scalarprods[6]);
        __AVX512_DATATYPE vs_2_5 = _AVX512_SET1(scalarprods[7]);

        h2 = _AVX512_MUL(tau5, vs_1_5);
        h3 = _AVX512_MUL(tau5, vs_2_5);

        __AVX512_DATATYPE vs_3_5 = _AVX512_SET1(scalarprods[8]);
        __AVX512_DATATYPE vs_4_5 = _AVX512_SET1(scalarprods[9]);

        h4 = _AVX512_MUL(tau5, vs_3_5);
        h5 = _AVX512_MUL(tau5, vs_4_5);

        v1 = _AVX512_FMSUB(v1, tau5, _AVX512_ADD(_AVX512_FMA(w1, h5, _AVX512_MUL(z1,h4)), _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2))));
        v2 = _AVX512_FMSUB(v2, tau5, _AVX512_ADD(_AVX512_FMA(w2, h5, _AVX512_MUL(z2,h4)), _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2))));

        __AVX512_DATATYPE tau6 = _AVX512_SET1(hh[ldh*5]);
        __AVX512_DATATYPE vs_1_6 = _AVX512_SET1(scalarprods[10]);
        __AVX512_DATATYPE vs_2_6 = _AVX512_SET1(scalarprods[11]);
        h2 = _AVX512_MUL(tau6, vs_1_6);
        h3 = _AVX512_MUL(tau6, vs_2_6);

        __AVX512_DATATYPE vs_3_6 = _AVX512_SET1(scalarprods[12]);
        __AVX512_DATATYPE vs_4_6 = _AVX512_SET1(scalarprods[13]);
        __AVX512_DATATYPE vs_5_6 = _AVX512_SET1(scalarprods[14]);

        h4 = _AVX512_MUL(tau6, vs_3_6);
        h5 = _AVX512_MUL(tau6, vs_4_6);
        h6 = _AVX512_MUL(tau6, vs_5_6);

        t1 = _AVX512_FMSUB(t1, tau6, _AVX512_FMA(v1, h6, _AVX512_ADD(_AVX512_FMA(w1, h5, _AVX512_MUL(z1,h4)), _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2)))));
        t2 = _AVX512_FMSUB(t2, tau6, _AVX512_FMA(v2, h6, _AVX512_ADD(_AVX512_FMA(w2, h5, _AVX512_MUL(z2,h4)), _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2)))));

        /////////////////////////////////////////////////////
        // Rank-1 update of Q [8 x nb+3]
        /////////////////////////////////////////////////////

        q1 = _AVX512_LOAD(&q[0]);
        q2 = _AVX512_LOAD(&q[0+offset]);

        q1 = _AVX512_SUB(q1, t1);
        q2 = _AVX512_SUB(q2, t2);

        _AVX512_STORE(&q[0],q1);
        _AVX512_STORE(&q[0+offset],q2);

        h6 = _AVX512_SET1(hh[(ldh*5)+1]);
        q1 = _AVX512_LOAD(&q[ldq]);
        q2 = _AVX512_LOAD(&q[ldq+offset]);

        q1 = _AVX512_SUB(q1, v1);
        q2 = _AVX512_SUB(q2, v2);

        q1 = _AVX512_NFMA(t1, h6, q1);
        q2 = _AVX512_NFMA(t2, h6, q2);

        _AVX512_STORE(&q[ldq],q1);
        _AVX512_STORE(&q[ldq+offset],q2);

        h5 = _AVX512_SET1(hh[(ldh*4)+1]);
        q1 = _AVX512_LOAD(&q[ldq*2]);
        q2 = _AVX512_LOAD(&q[(ldq*2)+offset]);

        q1 = _AVX512_SUB(q1, w1);
        q2 = _AVX512_SUB(q2, w2);

        q1 = _AVX512_NFMA(v1, h5, q1);
        q2 = _AVX512_NFMA(v2, h5, q2);

        h6 = _AVX512_SET1(hh[(ldh*5)+2]);

        q1 = _AVX512_NFMA(t1, h6, q1);
        q2 = _AVX512_NFMA(t2, h6, q2);

        _AVX512_STORE(&q[ldq*2],q1);
        _AVX512_STORE(&q[(ldq*2)+offset],q2);

        h4 = _AVX512_SET1(hh[(ldh*3)+1]);
        q1 = _AVX512_LOAD(&q[ldq*3]);
        q2 = _AVX512_LOAD(&q[(ldq*3)+offset]);

        q1 = _AVX512_SUB(q1, z1);
        q2 = _AVX512_SUB(q2, z2);

        q1 = _AVX512_NFMA(w1, h4, q1);
        q2 = _AVX512_NFMA(w2, h4, q2);

        h5 = _AVX512_SET1(hh[(ldh*4)+2]);

        q1 = _AVX512_NFMA(v1, h5, q1);
        q2 = _AVX512_NFMA(v2, h5, q2);

        h6 = _AVX512_SET1(hh[(ldh*5)+3]);

        q1 = _AVX512_NFMA(t1, h6, q1);
        q2 = _AVX512_NFMA(t2, h6, q2);

        _AVX512_STORE(&q[ldq*3],q1);
        _AVX512_STORE(&q[(ldq*3)+offset],q2);

        h3 = _AVX512_SET1(hh[(ldh*2)+1]);
        q1 = _AVX512_LOAD(&q[ldq*4]);
        q2 = _AVX512_LOAD(&q[(ldq*4)+offset]);

        q1 = _AVX512_SUB(q1, y1);
        q2 = _AVX512_SUB(q2, y2);

        q1 = _AVX512_NFMA(z1, h3, q1);
        q2 = _AVX512_NFMA(z2, h3, q2);

        h4 = _AVX512_SET1(hh[(ldh*3)+2]);

        q1 = _AVX512_NFMA(w1, h4, q1);
        q2 = _AVX512_NFMA(w2, h4, q2);

        h5 = _AVX512_SET1(hh[(ldh*4)+3]);

        q1 = _AVX512_NFMA(v1, h5, q1);
        q2 = _AVX512_NFMA(v2, h5, q2);

        h6 = _AVX512_SET1(hh[(ldh*5)+4]);

        q1 = _AVX512_NFMA(t1, h6, q1);
        q2 = _AVX512_NFMA(t2, h6, q2);

        _AVX512_STORE(&q[ldq*4],q1);
        _AVX512_STORE(&q[(ldq*4)+offset],q2);

        h2 = _AVX512_SET1(hh[(ldh)+1]);
        q1 = _AVX512_LOAD(&q[ldq*5]);
        q2 = _AVX512_LOAD(&q[(ldq*5)+offset]);

        q1 = _AVX512_SUB(q1, x1);
        q2 = _AVX512_SUB(q2, x2);

        q1 = _AVX512_NFMA(y1, h2, q1);
        q2 = _AVX512_NFMA(y2, h2, q2);

        h3 = _AVX512_SET1(hh[(ldh*2)+2]);

        q1 = _AVX512_NFMA(z1, h3, q1);
        q2 = _AVX512_NFMA(z2, h3, q2);

        h4 = _AVX512_SET1(hh[(ldh*3)+3]);

        q1 = _AVX512_NFMA(w1, h4, q1);
        q2 = _AVX512_NFMA(w2, h4, q2);

        h5 = _AVX512_SET1(hh[(ldh*4)+4]);

        q1 = _AVX512_NFMA(v1, h5, q1);
        q2 = _AVX512_NFMA(v2, h5, q2);

        h6 = _AVX512_SET1(hh[(ldh*5)+5]);

        q1 = _AVX512_NFMA(t1, h6, q1);
        q2 = _AVX512_NFMA(t2, h6, q2);

        _AVX512_STORE(&q[ldq*5],q1);
        _AVX512_STORE(&q[(ldq*5)+offset],q2);

        for (i = 6; i < nb; i++)
        {
                q1 = _AVX512_LOAD(&q[i*ldq]);
                q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);

                h1 = _AVX512_SET1(hh[i-5]);

                q1 = _AVX512_NFMA(x1, h1, q1);
                q2 = _AVX512_NFMA(x2, h1, q2);

                h2 = _AVX512_SET1(hh[ldh+i-4]);

                q1 = _AVX512_NFMA(y1, h2, q1);
                q2 = _AVX512_NFMA(y2, h2, q2);

                h3 = _AVX512_SET1(hh[(ldh*2)+i-3]);

                q1 = _AVX512_NFMA(z1, h3, q1);
                q2 = _AVX512_NFMA(z2, h3, q2);

                h4 = _AVX512_SET1(hh[(ldh*3)+i-2]);

                q1 = _AVX512_NFMA(w1, h4, q1);
                q2 = _AVX512_NFMA(w2, h4, q2);

                h5 = _AVX512_SET1(hh[(ldh*4)+i-1]);

                q1 = _AVX512_NFMA(v1, h5, q1);
                q2 = _AVX512_NFMA(v2, h5, q2);

                h6 = _AVX512_SET1(hh[(ldh*5)+i]);

                q1 = _AVX512_NFMA(t1, h6, q1);
                q2 = _AVX512_NFMA(t2, h6, q2);

                _AVX512_STORE(&q[i*ldq],q1);
                _AVX512_STORE(&q[(i*ldq)+offset],q2);

        }

        h1 = _AVX512_SET1(hh[nb-5]);
        q1 = _AVX512_LOAD(&q[nb*ldq]);
        q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);

        q1 = _AVX512_NFMA(x1, h1, q1);
        q2 = _AVX512_NFMA(x2, h1, q2);

        h2 = _AVX512_SET1(hh[ldh+nb-4]);

        q1 = _AVX512_NFMA(y1, h2, q1);
        q2 = _AVX512_NFMA(y2, h2, q2);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-3]);

        q1 = _AVX512_NFMA(z1, h3, q1);
        q2 = _AVX512_NFMA(z2, h3, q2);

        h4 = _AVX512_SET1(hh[(ldh*3)+nb-2]);

        q1 = _AVX512_NFMA(w1, h4, q1);
        q2 = _AVX512_NFMA(w2, h4, q2);

        h5 = _AVX512_SET1(hh[(ldh*4)+nb-1]);

        q1 = _AVX512_NFMA(v1, h5, q1);
        q2 = _AVX512_NFMA(v2, h5, q2);

        _AVX512_STORE(&q[nb*ldq],q1);
        _AVX512_STORE(&q[(nb*ldq)+offset],q2);

        h1 = _AVX512_SET1(hh[nb-4]);
        q1 = _AVX512_LOAD(&q[(nb+1)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+1)*ldq)+offset]);

        q1 = _AVX512_NFMA(x1, h1, q1);
        q2 = _AVX512_NFMA(x2, h1, q2);

        h2 = _AVX512_SET1(hh[ldh+nb-3]);

        q1 = _AVX512_NFMA(y1, h2, q1);
        q2 = _AVX512_NFMA(y2, h2, q2);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-2]);

        q1 = _AVX512_NFMA(z1, h3, q1);
        q2 = _AVX512_NFMA(z2, h3, q2);

        h4 = _AVX512_SET1(hh[(ldh*3)+nb-1]);

        q1 = _AVX512_NFMA(w1, h4, q1);
        q2 = _AVX512_NFMA(w2, h4, q2);

        _AVX512_STORE(&q[(nb+1)*ldq],q1);
        _AVX512_STORE(&q[((nb+1)*ldq)+offset],q2);

        h1 = _AVX512_SET1(hh[nb-3]);
        q1 = _AVX512_LOAD(&q[(nb+2)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+2)*ldq)+offset]);

        q1 = _AVX512_NFMA(x1, h1, q1);
        q2 = _AVX512_NFMA(x2, h1, q2);

        h2 = _AVX512_SET1(hh[ldh+nb-2]);

        q1 = _AVX512_NFMA(y1, h2, q1);
        q2 = _AVX512_NFMA(y2, h2, q2);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-1]);

        q1 = _AVX512_NFMA(z1, h3, q1);
        q2 = _AVX512_NFMA(z2, h3, q2);

        _AVX512_STORE(&q[(nb+2)*ldq],q1);
        _AVX512_STORE(&q[((nb+2)*ldq)+offset],q2);

        h1 = _AVX512_SET1(hh[nb-2]);
        q1 = _AVX512_LOAD(&q[(nb+3)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+3)*ldq)+offset]);

        q1 = _AVX512_NFMA(x1, h1, q1);
        q2 = _AVX512_NFMA(x2, h1, q2);

        h2 = _AVX512_SET1(hh[ldh+nb-1]);

        q1 = _AVX512_NFMA(y1, h2, q1);
        q2 = _AVX512_NFMA(y2, h2, q2);

        _AVX512_STORE(&q[(nb+3)*ldq],q1);
        _AVX512_STORE(&q[((nb+3)*ldq)+offset],q2);

        h1 = _AVX512_SET1(hh[nb-1]);
        q1 = _AVX512_LOAD(&q[(nb+4)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+4)*ldq)+offset]);

        q1 = _AVX512_NFMA(x1, h1, q1);
        q2 = _AVX512_NFMA(x2, h1, q2);

        _AVX512_STORE(&q[(nb+4)*ldq],q1);
        _AVX512_STORE(&q[((nb+4)*ldq)+offset],q2);

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
__forceinline void hh_trafo_kernel_24_AVX512_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_48_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods)
#endif
{
        /////////////////////////////////////////////////////
        // Matrix Vector Multiplication, Q [8 x nb+3] * hh
        // hh contains four householder vectors
        /////////////////////////////////////////////////////
        int i;

        __AVX512_DATATYPE a1_1 = _AVX512_LOAD(&q[ldq*5]);
        __AVX512_DATATYPE a2_1 = _AVX512_LOAD(&q[ldq*4]);
        __AVX512_DATATYPE a3_1 = _AVX512_LOAD(&q[ldq*3]);
        __AVX512_DATATYPE a4_1 = _AVX512_LOAD(&q[ldq*2]);
        __AVX512_DATATYPE a5_1 = _AVX512_LOAD(&q[ldq]);
        __AVX512_DATATYPE a6_1 = _AVX512_LOAD(&q[0]);

        __AVX512_DATATYPE h_6_5 = _AVX512_SET1(hh[(ldh*5)+1]);
        __AVX512_DATATYPE h_6_4 = _AVX512_SET1(hh[(ldh*5)+2]);
        __AVX512_DATATYPE h_6_3 = _AVX512_SET1(hh[(ldh*5)+3]);
        __AVX512_DATATYPE h_6_2 = _AVX512_SET1(hh[(ldh*5)+4]);
        __AVX512_DATATYPE h_6_1 = _AVX512_SET1(hh[(ldh*5)+5]);

//        register __AVX512_DATATYPE t1 = _AVX512_FMA(a5_1, h_6_5, a6_1);
         __AVX512_DATATYPE t1 = _AVX512_FMA(a5_1, h_6_5, a6_1);

        t1 = _AVX512_FMA(a4_1, h_6_4, t1);
        t1 = _AVX512_FMA(a3_1, h_6_3, t1);
        t1 = _AVX512_FMA(a2_1, h_6_2, t1);
        t1 = _AVX512_FMA(a1_1, h_6_1, t1);

        __AVX512_DATATYPE h_5_4 = _AVX512_SET1(hh[(ldh*4)+1]);
        __AVX512_DATATYPE h_5_3 = _AVX512_SET1(hh[(ldh*4)+2]);
        __AVX512_DATATYPE h_5_2 = _AVX512_SET1(hh[(ldh*4)+3]);
        __AVX512_DATATYPE h_5_1 = _AVX512_SET1(hh[(ldh*4)+4]);

//        register __AVX512_DATATYPE v1 = _AVX512_FMA(a4_1, h_5_4, a5_1);
         __AVX512_DATATYPE v1 = _AVX512_FMA(a4_1, h_5_4, a5_1);

        v1 = _AVX512_FMA(a3_1, h_5_3, v1);
        v1 = _AVX512_FMA(a2_1, h_5_2, v1);
        v1 = _AVX512_FMA(a1_1, h_5_1, v1);

        __AVX512_DATATYPE h_4_3 = _AVX512_SET1(hh[(ldh*3)+1]);
        __AVX512_DATATYPE h_4_2 = _AVX512_SET1(hh[(ldh*3)+2]);
        __AVX512_DATATYPE h_4_1 = _AVX512_SET1(hh[(ldh*3)+3]);

//        register __AVX512_DATATYPE w1 = _AVX512_FMA(a3_1, h_4_3, a4_1);
        __AVX512_DATATYPE w1 = _AVX512_FMA(a3_1, h_4_3, a4_1);

        w1 = _AVX512_FMA(a2_1, h_4_2, w1);
        w1 = _AVX512_FMA(a1_1, h_4_1, w1);

        __AVX512_DATATYPE h_2_1 = _AVX512_SET1(hh[ldh+1]);
        __AVX512_DATATYPE h_3_2 = _AVX512_SET1(hh[(ldh*2)+1]);
        __AVX512_DATATYPE h_3_1 = _AVX512_SET1(hh[(ldh*2)+2]);

//        register __AVX512_DATATYPE z1 = _AVX512_FMA(a2_1, h_3_2, a3_1);
        __AVX512_DATATYPE z1 = _AVX512_FMA(a2_1, h_3_2, a3_1);

        z1 = _AVX512_FMA(a1_1, h_3_1, z1);
//        register __AVX512_DATATYPE y1 = _AVX512_FMA(a1_1, h_2_1, a2_1);
        __AVX512_DATATYPE y1 = _AVX512_FMA(a1_1, h_2_1, a2_1);


//        register __AVX512_DATATYPE x1 = a1_1;
        __AVX512_DATATYPE x1 = a1_1;


        __AVX512_DATATYPE a1_2 = _AVX512_LOAD(&q[(ldq*5)+offset]);
        __AVX512_DATATYPE a2_2 = _AVX512_LOAD(&q[(ldq*4)+offset]);
        __AVX512_DATATYPE a3_2 = _AVX512_LOAD(&q[(ldq*3)+offset]);
        __AVX512_DATATYPE a4_2 = _AVX512_LOAD(&q[(ldq*2)+offset]);
        __AVX512_DATATYPE a5_2 = _AVX512_LOAD(&q[(ldq)+offset]);
        __AVX512_DATATYPE a6_2 = _AVX512_LOAD(&q[0+offset]);

//        register __AVX512_DATATYPE t2 = _AVX512_FMA(a5_2, h_6_5, a6_2);
        __AVX512_DATATYPE t2 = _AVX512_FMA(a5_2, h_6_5, a6_2);

        t2 = _AVX512_FMA(a4_2, h_6_4, t2);
        t2 = _AVX512_FMA(a3_2, h_6_3, t2);
        t2 = _AVX512_FMA(a2_2, h_6_2, t2);
        t2 = _AVX512_FMA(a1_2, h_6_1, t2);

//        register __AVX512_DATATYPE v2 = _AVX512_FMA(a4_2, h_5_4, a5_2);
        __AVX512_DATATYPE v2 = _AVX512_FMA(a4_2, h_5_4, a5_2);

        v2 = _AVX512_FMA(a3_2, h_5_3, v2);
        v2 = _AVX512_FMA(a2_2, h_5_2, v2);
        v2 = _AVX512_FMA(a1_2, h_5_1, v2);

//        register __AVX512_DATATYPE w2 = _AVX512_FMA(a3_2, h_4_3, a4_2);
        __AVX512_DATATYPE w2 = _AVX512_FMA(a3_2, h_4_3, a4_2);

        w2 = _AVX512_FMA(a2_2, h_4_2, w2);
        w2 = _AVX512_FMA(a1_2, h_4_1, w2);

//        register __AVX512_DATATYPE z2 = _AVX512_FMA(a2_2, h_3_2, a3_2);
         __AVX512_DATATYPE z2 = _AVX512_FMA(a2_2, h_3_2, a3_2);

        z2 = _AVX512_FMA(a1_2, h_3_1, z2);
//        register __AVX512_DATATYPE y2 = _AVX512_FMA(a1_2, h_2_1, a2_2);
        __AVX512_DATATYPE y2 = _AVX512_FMA(a1_2, h_2_1, a2_2);


//        register __AVX512_DATATYPE x2 = a1_2;
        __AVX512_DATATYPE x2 = a1_2;


        __AVX512_DATATYPE a1_3 = _AVX512_LOAD(&q[(ldq*5)+2*offset]);
        __AVX512_DATATYPE a2_3 = _AVX512_LOAD(&q[(ldq*4)+2*offset]);
        __AVX512_DATATYPE a3_3 = _AVX512_LOAD(&q[(ldq*3)+2*offset]);
        __AVX512_DATATYPE a4_3 = _AVX512_LOAD(&q[(ldq*2)+2*offset]);
        __AVX512_DATATYPE a5_3 = _AVX512_LOAD(&q[(ldq)+2*offset]);
        __AVX512_DATATYPE a6_3 = _AVX512_LOAD(&q[0+2*offset]);

//        register __AVX512_DATATYPE t3 = _AVX512_FMA(a5_3, h_6_5, a6_3);
        __AVX512_DATATYPE t3 = _AVX512_FMA(a5_3, h_6_5, a6_3);

        t3 = _AVX512_FMA(a4_3, h_6_4, t3);
        t3 = _AVX512_FMA(a3_3, h_6_3, t3);
        t3 = _AVX512_FMA(a2_3, h_6_2, t3);
        t3 = _AVX512_FMA(a1_3, h_6_1, t3);

//        register __AVX512_DATATYPE v3 = _AVX512_FMA(a4_3, h_5_4, a5_3);
        __AVX512_DATATYPE v3 = _AVX512_FMA(a4_3, h_5_4, a5_3);

        v3 = _AVX512_FMA(a3_3, h_5_3, v3);
        v3 = _AVX512_FMA(a2_3, h_5_2, v3);
        v3 = _AVX512_FMA(a1_3, h_5_1, v3);

//        register __AVX512_DATATYPE w3 = _AVX512_FMA(a3_3, h_4_3, a4_3);
        __AVX512_DATATYPE w3 = _AVX512_FMA(a3_3, h_4_3, a4_3);

        w3 = _AVX512_FMA(a2_3, h_4_2, w3);
        w3 = _AVX512_FMA(a1_3, h_4_1, w3);

//        register __AVX512_DATATYPE z3 = _AVX512_FMA(a2_3, h_3_2, a3_3);
        __AVX512_DATATYPE z3 = _AVX512_FMA(a2_3, h_3_2, a3_3);

        z3 = _AVX512_FMA(a1_3, h_3_1, z3);
//        register __AVX512_DATATYPE y3 = _AVX512_FMA(a1_3, h_2_1, a2_3);
        __AVX512_DATATYPE y3 = _AVX512_FMA(a1_3, h_2_1, a2_3);


//        register __AVX512_DATATYPE x3 = a1_3;
        __AVX512_DATATYPE x3 = a1_3;


        __AVX512_DATATYPE q1;
        __AVX512_DATATYPE q2;
        __AVX512_DATATYPE q3;

        __AVX512_DATATYPE h1;
        __AVX512_DATATYPE h2;
        __AVX512_DATATYPE h3;
        __AVX512_DATATYPE h4;
        __AVX512_DATATYPE h5;
        __AVX512_DATATYPE h6;

        for(i = 6; i < nb; i++)
        {
                h1 = _AVX512_SET1(hh[i-5]);
                q1 = _AVX512_LOAD(&q[i*ldq]);
                q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);
                q3 = _AVX512_LOAD(&q[(i*ldq)+2*offset]);

                x1 = _AVX512_FMA(q1, h1, x1);
                x2 = _AVX512_FMA(q2, h1, x2);
                x3 = _AVX512_FMA(q3, h1, x3);

                h2 = _AVX512_SET1(hh[ldh+i-4]);

                y1 = _AVX512_FMA(q1, h2, y1);
                y2 = _AVX512_FMA(q2, h2, y2);
                y3 = _AVX512_FMA(q3, h2, y3);

                h3 = _AVX512_SET1(hh[(ldh*2)+i-3]);

                z1 = _AVX512_FMA(q1, h3, z1);
                z2 = _AVX512_FMA(q2, h3, z2);
                z3 = _AVX512_FMA(q3, h3, z3);

                h4 = _AVX512_SET1(hh[(ldh*3)+i-2]);

                w1 = _AVX512_FMA(q1, h4, w1);
                w2 = _AVX512_FMA(q2, h4, w2);
                w3 = _AVX512_FMA(q3, h4, w3);

                h5 = _AVX512_SET1(hh[(ldh*4)+i-1]);

                v1 = _AVX512_FMA(q1, h5, v1);
                v2 = _AVX512_FMA(q2, h5, v2);
                v3 = _AVX512_FMA(q3, h5, v3);

                h6 = _AVX512_SET1(hh[(ldh*5)+i]);

                t1 = _AVX512_FMA(q1, h6, t1);
                t2 = _AVX512_FMA(q2, h6, t2);
                t3 = _AVX512_FMA(q3, h6, t3);
        }

        h1 = _AVX512_SET1(hh[nb-5]);
        q1 = _AVX512_LOAD(&q[nb*ldq]);
        q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[(nb*ldq)+2*offset]);

        x1 = _AVX512_FMA(q1, h1, x1);
        x2 = _AVX512_FMA(q2, h1, x2);
        x3 = _AVX512_FMA(q3, h1, x3);

        h2 = _AVX512_SET1(hh[ldh+nb-4]);

        y1 = _AVX512_FMA(q1, h2, y1);
        y2 = _AVX512_FMA(q2, h2, y2);
        y3 = _AVX512_FMA(q3, h2, y3);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-3]);

        z1 = _AVX512_FMA(q1, h3, z1);
        z2 = _AVX512_FMA(q2, h3, z2);
        z3 = _AVX512_FMA(q3, h3, z3);

        h4 = _AVX512_SET1(hh[(ldh*3)+nb-2]);

        w1 = _AVX512_FMA(q1, h4, w1);
        w2 = _AVX512_FMA(q2, h4, w2);
        w3 = _AVX512_FMA(q3, h4, w3);

        h5 = _AVX512_SET1(hh[(ldh*4)+nb-1]);

        v1 = _AVX512_FMA(q1, h5, v1);
        v2 = _AVX512_FMA(q2, h5, v2);
        v3 = _AVX512_FMA(q3, h5, v3);

        h1 = _AVX512_SET1(hh[nb-4]);

        q1 = _AVX512_LOAD(&q[(nb+1)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+1)*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[((nb+1)*ldq)+2*offset]);

        x1 = _AVX512_FMA(q1, h1, x1);
        x2 = _AVX512_FMA(q2, h1, x2);
        x3 = _AVX512_FMA(q3, h1, x3);

        h2 = _AVX512_SET1(hh[ldh+nb-3]);

        y1 = _AVX512_FMA(q1, h2, y1);
        y2 = _AVX512_FMA(q2, h2, y2);
        y3 = _AVX512_FMA(q3, h2, y3);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-2]);

        z1 = _AVX512_FMA(q1, h3, z1);
        z2 = _AVX512_FMA(q2, h3, z2);
        z3 = _AVX512_FMA(q3, h3, z3);

        h4 = _AVX512_SET1(hh[(ldh*3)+nb-1]);

        w1 = _AVX512_FMA(q1, h4, w1);
        w2 = _AVX512_FMA(q2, h4, w2);
        w3 = _AVX512_FMA(q3, h4, w3);

        h1 = _AVX512_SET1(hh[nb-3]);
        q1 = _AVX512_LOAD(&q[(nb+2)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+2)*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[((nb+2)*ldq)+2*offset]);

        x1 = _AVX512_FMA(q1, h1, x1);
        x2 = _AVX512_FMA(q2, h1, x2);
        x3 = _AVX512_FMA(q3, h1, x3);

        h2 = _AVX512_SET1(hh[ldh+nb-2]);

        y1 = _AVX512_FMA(q1, h2, y1);
        y2 = _AVX512_FMA(q2, h2, y2);
        y3 = _AVX512_FMA(q3, h2, y3);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-1]);

        z1 = _AVX512_FMA(q1, h3, z1);
        z2 = _AVX512_FMA(q2, h3, z2);
        z3 = _AVX512_FMA(q3, h3, z3);

        h1 = _AVX512_SET1(hh[nb-2]);
        q1 = _AVX512_LOAD(&q[(nb+3)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+3)*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[((nb+3)*ldq)+2*offset]);

        x1 = _AVX512_FMA(q1, h1, x1);
        x2 = _AVX512_FMA(q2, h1, x2);
        x3 = _AVX512_FMA(q3, h1, x3);

        h2 = _AVX512_SET1(hh[ldh+nb-1]);

        y1 = _AVX512_FMA(q1, h2, y1);
        y2 = _AVX512_FMA(q2, h2, y2);
        y3 = _AVX512_FMA(q3, h2, y3);

        h1 = _AVX512_SET1(hh[nb-1]);
        q1 = _AVX512_LOAD(&q[(nb+4)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+4)*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[((nb+4)*ldq)+2*offset]);

        x1 = _AVX512_FMA(q1, h1, x1);
        x2 = _AVX512_FMA(q2, h1, x2);
        x3 = _AVX512_FMA(q3, h1, x3);

        /////////////////////////////////////////////////////
        // Apply tau, correct wrong calculation using pre-calculated scalar products
        /////////////////////////////////////////////////////

        __AVX512_DATATYPE tau1 = _AVX512_SET1(hh[0]);
        x1 = _AVX512_MUL(x1, tau1);
        x2 = _AVX512_MUL(x2, tau1);
        x3 = _AVX512_MUL(x3, tau1);

        __AVX512_DATATYPE tau2 = _AVX512_SET1(hh[ldh]);
        __AVX512_DATATYPE vs_1_2 = _AVX512_SET1(scalarprods[0]);
        h2 = _AVX512_MUL(tau2, vs_1_2);

        y1 = _AVX512_FMSUB(y1, tau2, _AVX512_MUL(x1,h2));
        y2 = _AVX512_FMSUB(y2, tau2, _AVX512_MUL(x2,h2));
        y3 = _AVX512_FMSUB(y3, tau2, _AVX512_MUL(x3,h2));

        __AVX512_DATATYPE tau3 = _AVX512_SET1(hh[ldh*2]);
        __AVX512_DATATYPE vs_1_3 = _AVX512_SET1(scalarprods[1]);
        __AVX512_DATATYPE vs_2_3 = _AVX512_SET1(scalarprods[2]);

        h2 = _AVX512_MUL(tau3, vs_1_3);
        h3 = _AVX512_MUL(tau3, vs_2_3);

        z1 = _AVX512_FMSUB(z1, tau3, _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2)));
        z2 = _AVX512_FMSUB(z2, tau3, _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2)));
        z3 = _AVX512_FMSUB(z3, tau3, _AVX512_FMA(y3, h3, _AVX512_MUL(x3,h2)));

        __AVX512_DATATYPE tau4 = _AVX512_SET1(hh[ldh*3]);
        __AVX512_DATATYPE vs_1_4 = _AVX512_SET1(scalarprods[3]);
        __AVX512_DATATYPE vs_2_4 = _AVX512_SET1(scalarprods[4]);

        h2 = _AVX512_MUL(tau4, vs_1_4);
        h3 = _AVX512_MUL(tau4, vs_2_4);

        __AVX512_DATATYPE vs_3_4 = _AVX512_SET1(scalarprods[5]);
        h4 = _AVX512_MUL(tau4, vs_3_4);

        w1 = _AVX512_FMSUB(w1, tau4, _AVX512_FMA(z1, h4, _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2))));
        w2 = _AVX512_FMSUB(w2, tau4, _AVX512_FMA(z2, h4, _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2))));
        w3 = _AVX512_FMSUB(w3, tau4, _AVX512_FMA(z3, h4, _AVX512_FMA(y3, h3, _AVX512_MUL(x3,h2))));

        __AVX512_DATATYPE tau5 = _AVX512_SET1(hh[ldh*4]);
        __AVX512_DATATYPE vs_1_5 = _AVX512_SET1(scalarprods[6]);
        __AVX512_DATATYPE vs_2_5 = _AVX512_SET1(scalarprods[7]);

        h2 = _AVX512_MUL(tau5, vs_1_5);
        h3 = _AVX512_MUL(tau5, vs_2_5);

        __AVX512_DATATYPE vs_3_5 = _AVX512_SET1(scalarprods[8]);
        __AVX512_DATATYPE vs_4_5 = _AVX512_SET1(scalarprods[9]);

        h4 = _AVX512_MUL(tau5, vs_3_5);
        h5 = _AVX512_MUL(tau5, vs_4_5);

        v1 = _AVX512_FMSUB(v1, tau5, _AVX512_ADD(_AVX512_FMA(w1, h5, _AVX512_MUL(z1,h4)), _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2))));
        v2 = _AVX512_FMSUB(v2, tau5, _AVX512_ADD(_AVX512_FMA(w2, h5, _AVX512_MUL(z2,h4)), _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2))));
        v3 = _AVX512_FMSUB(v3, tau5, _AVX512_ADD(_AVX512_FMA(w3, h5, _AVX512_MUL(z3,h4)), _AVX512_FMA(y3, h3, _AVX512_MUL(x3,h2))));

        __AVX512_DATATYPE tau6 = _AVX512_SET1(hh[ldh*5]);
        __AVX512_DATATYPE vs_1_6 = _AVX512_SET1(scalarprods[10]);
        __AVX512_DATATYPE vs_2_6 = _AVX512_SET1(scalarprods[11]);
        h2 = _AVX512_MUL(tau6, vs_1_6);
        h3 = _AVX512_MUL(tau6, vs_2_6);

        __AVX512_DATATYPE vs_3_6 = _AVX512_SET1(scalarprods[12]);
        __AVX512_DATATYPE vs_4_6 = _AVX512_SET1(scalarprods[13]);
        __AVX512_DATATYPE vs_5_6 = _AVX512_SET1(scalarprods[14]);

        h4 = _AVX512_MUL(tau6, vs_3_6);
        h5 = _AVX512_MUL(tau6, vs_4_6);
        h6 = _AVX512_MUL(tau6, vs_5_6);

        t1 = _AVX512_FMSUB(t1, tau6, _AVX512_FMA(v1, h6, _AVX512_ADD(_AVX512_FMA(w1, h5, _AVX512_MUL(z1,h4)), _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2)))));
        t2 = _AVX512_FMSUB(t2, tau6, _AVX512_FMA(v2, h6, _AVX512_ADD(_AVX512_FMA(w2, h5, _AVX512_MUL(z2,h4)), _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2)))));
        t3 = _AVX512_FMSUB(t3, tau6, _AVX512_FMA(v3, h6, _AVX512_ADD(_AVX512_FMA(w3, h5, _AVX512_MUL(z3,h4)), _AVX512_FMA(y3, h3, _AVX512_MUL(x3,h2)))));

        /////////////////////////////////////////////////////
        // Rank-1 update of Q [8 x nb+3]
        /////////////////////////////////////////////////////

        q1 = _AVX512_LOAD(&q[0]);
        q2 = _AVX512_LOAD(&q[0+offset]);
        q3 = _AVX512_LOAD(&q[0+2*offset]);

        q1 = _AVX512_SUB(q1, t1);
        q2 = _AVX512_SUB(q2, t2);
        q3 = _AVX512_SUB(q3, t3);

        _AVX512_STORE(&q[0],q1);
        _AVX512_STORE(&q[0+offset],q2);
        _AVX512_STORE(&q[0+2*offset],q3);

        h6 = _AVX512_SET1(hh[(ldh*5)+1]);
        q1 = _AVX512_LOAD(&q[ldq]);
        q2 = _AVX512_LOAD(&q[ldq+offset]);
        q3 = _AVX512_LOAD(&q[ldq+2*offset]);

        q1 = _AVX512_SUB(q1, v1);
        q2 = _AVX512_SUB(q2, v2);
        q3 = _AVX512_SUB(q3, v3);

        q1 = _AVX512_NFMA(t1, h6, q1);
        q2 = _AVX512_NFMA(t2, h6, q2);
        q3 = _AVX512_NFMA(t3, h6, q3);

        _AVX512_STORE(&q[ldq],q1);
        _AVX512_STORE(&q[ldq+offset],q2);
        _AVX512_STORE(&q[ldq+2*offset],q3);

        h5 = _AVX512_SET1(hh[(ldh*4)+1]);
        q1 = _AVX512_LOAD(&q[ldq*2]);
        q2 = _AVX512_LOAD(&q[(ldq*2)+offset]);
        q3 = _AVX512_LOAD(&q[(ldq*2)+2*offset]);

        q1 = _AVX512_SUB(q1, w1);
        q2 = _AVX512_SUB(q2, w2);
        q3 = _AVX512_SUB(q3, w3);

        q1 = _AVX512_NFMA(v1, h5, q1);
        q2 = _AVX512_NFMA(v2, h5, q2);
        q3 = _AVX512_NFMA(v3, h5, q3);

        h6 = _AVX512_SET1(hh[(ldh*5)+2]);

        q1 = _AVX512_NFMA(t1, h6, q1);
        q2 = _AVX512_NFMA(t2, h6, q2);
        q3 = _AVX512_NFMA(t3, h6, q3);

        _AVX512_STORE(&q[ldq*2],q1);
        _AVX512_STORE(&q[(ldq*2)+offset],q2);
        _AVX512_STORE(&q[(ldq*2)+2*offset],q3);

        h4 = _AVX512_SET1(hh[(ldh*3)+1]);
        q1 = _AVX512_LOAD(&q[ldq*3]);
        q2 = _AVX512_LOAD(&q[(ldq*3)+offset]);
        q3 = _AVX512_LOAD(&q[(ldq*3)+2*offset]);

        q1 = _AVX512_SUB(q1, z1);
        q2 = _AVX512_SUB(q2, z2);
        q3 = _AVX512_SUB(q3, z3);

        q1 = _AVX512_NFMA(w1, h4, q1);
        q2 = _AVX512_NFMA(w2, h4, q2);
        q3 = _AVX512_NFMA(w3, h4, q3);

        h5 = _AVX512_SET1(hh[(ldh*4)+2]);

        q1 = _AVX512_NFMA(v1, h5, q1);
        q2 = _AVX512_NFMA(v2, h5, q2);
        q3 = _AVX512_NFMA(v3, h5, q3);

        h6 = _AVX512_SET1(hh[(ldh*5)+3]);

        q1 = _AVX512_NFMA(t1, h6, q1);
        q2 = _AVX512_NFMA(t2, h6, q2);
        q3 = _AVX512_NFMA(t3, h6, q3);

        _AVX512_STORE(&q[ldq*3],q1);
        _AVX512_STORE(&q[(ldq*3)+offset],q2);
        _AVX512_STORE(&q[(ldq*3)+2*offset],q3);

        h3 = _AVX512_SET1(hh[(ldh*2)+1]);
        q1 = _AVX512_LOAD(&q[ldq*4]);
        q2 = _AVX512_LOAD(&q[(ldq*4)+offset]);
        q3 = _AVX512_LOAD(&q[(ldq*4)+2*offset]);

        q1 = _AVX512_SUB(q1, y1);
        q2 = _AVX512_SUB(q2, y2);
        q3 = _AVX512_SUB(q3, y3);

        q1 = _AVX512_NFMA(z1, h3, q1);
        q2 = _AVX512_NFMA(z2, h3, q2);
        q3 = _AVX512_NFMA(z3, h3, q3);

        h4 = _AVX512_SET1(hh[(ldh*3)+2]);

        q1 = _AVX512_NFMA(w1, h4, q1);
        q2 = _AVX512_NFMA(w2, h4, q2);
        q3 = _AVX512_NFMA(w3, h4, q3);

        h5 = _AVX512_SET1(hh[(ldh*4)+3]);

        q1 = _AVX512_NFMA(v1, h5, q1);
        q2 = _AVX512_NFMA(v2, h5, q2);
        q3 = _AVX512_NFMA(v3, h5, q3);

        h6 = _AVX512_SET1(hh[(ldh*5)+4]);

        q1 = _AVX512_NFMA(t1, h6, q1);
        q2 = _AVX512_NFMA(t2, h6, q2);
        q3 = _AVX512_NFMA(t3, h6, q3);

        _AVX512_STORE(&q[ldq*4],q1);
        _AVX512_STORE(&q[(ldq*4)+offset],q2);
        _AVX512_STORE(&q[(ldq*4)+2*offset],q3);

        h2 = _AVX512_SET1(hh[(ldh)+1]);
        q1 = _AVX512_LOAD(&q[ldq*5]);
        q2 = _AVX512_LOAD(&q[(ldq*5)+offset]);
        q3 = _AVX512_LOAD(&q[(ldq*5)+2*offset]);

        q1 = _AVX512_SUB(q1, x1);
        q2 = _AVX512_SUB(q2, x2);
        q3 = _AVX512_SUB(q3, x3);

        q1 = _AVX512_NFMA(y1, h2, q1);
        q2 = _AVX512_NFMA(y2, h2, q2);
        q3 = _AVX512_NFMA(y3, h2, q3);

        h3 = _AVX512_SET1(hh[(ldh*2)+2]);

        q1 = _AVX512_NFMA(z1, h3, q1);
        q2 = _AVX512_NFMA(z2, h3, q2);
        q3 = _AVX512_NFMA(z3, h3, q3);

        h4 = _AVX512_SET1(hh[(ldh*3)+3]);

        q1 = _AVX512_NFMA(w1, h4, q1);
        q2 = _AVX512_NFMA(w2, h4, q2);
        q3 = _AVX512_NFMA(w3, h4, q3);

        h5 = _AVX512_SET1(hh[(ldh*4)+4]);

        q1 = _AVX512_NFMA(v1, h5, q1);
        q2 = _AVX512_NFMA(v2, h5, q2);
        q3 = _AVX512_NFMA(v3, h5, q3);

        h6 = _AVX512_SET1(hh[(ldh*5)+5]);

        q1 = _AVX512_NFMA(t1, h6, q1);
        q2 = _AVX512_NFMA(t2, h6, q2);
        q3 = _AVX512_NFMA(t3, h6, q3);

        _AVX512_STORE(&q[ldq*5],q1);
        _AVX512_STORE(&q[(ldq*5)+offset],q2);
        _AVX512_STORE(&q[(ldq*5)+2*offset],q3);

        for (i = 6; i < nb; i++)
        {
                q1 = _AVX512_LOAD(&q[i*ldq]);
                q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);
                q3 = _AVX512_LOAD(&q[(i*ldq)+2*offset]);

                h1 = _AVX512_SET1(hh[i-5]);

                q1 = _AVX512_NFMA(x1, h1, q1);
                q2 = _AVX512_NFMA(x2, h1, q2);
                q3 = _AVX512_NFMA(x3, h1, q3);

                h2 = _AVX512_SET1(hh[ldh+i-4]);

                q1 = _AVX512_NFMA(y1, h2, q1);
                q2 = _AVX512_NFMA(y2, h2, q2);
                q3 = _AVX512_NFMA(y3, h2, q3);

                h3 = _AVX512_SET1(hh[(ldh*2)+i-3]);

                q1 = _AVX512_NFMA(z1, h3, q1);
                q2 = _AVX512_NFMA(z2, h3, q2);
                q3 = _AVX512_NFMA(z3, h3, q3);

                h4 = _AVX512_SET1(hh[(ldh*3)+i-2]);

                q1 = _AVX512_NFMA(w1, h4, q1);
                q2 = _AVX512_NFMA(w2, h4, q2);
                q3 = _AVX512_NFMA(w3, h4, q3);

                h5 = _AVX512_SET1(hh[(ldh*4)+i-1]);

                q1 = _AVX512_NFMA(v1, h5, q1);
                q2 = _AVX512_NFMA(v2, h5, q2);
                q3 = _AVX512_NFMA(v3, h5, q3);

                h6 = _AVX512_SET1(hh[(ldh*5)+i]);

                q1 = _AVX512_NFMA(t1, h6, q1);
                q2 = _AVX512_NFMA(t2, h6, q2);
                q3 = _AVX512_NFMA(t3, h6, q3);

                _AVX512_STORE(&q[i*ldq],q1);
                _AVX512_STORE(&q[(i*ldq)+offset],q2);
                _AVX512_STORE(&q[(i*ldq)+2*offset],q3);

        }

        h1 = _AVX512_SET1(hh[nb-5]);
        q1 = _AVX512_LOAD(&q[nb*ldq]);
        q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[(nb*ldq)+2*offset]);

        q1 = _AVX512_NFMA(x1, h1, q1);
        q2 = _AVX512_NFMA(x2, h1, q2);
        q3 = _AVX512_NFMA(x3, h1, q3);

        h2 = _AVX512_SET1(hh[ldh+nb-4]);

        q1 = _AVX512_NFMA(y1, h2, q1);
        q2 = _AVX512_NFMA(y2, h2, q2);
        q3 = _AVX512_NFMA(y3, h2, q3);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-3]);

        q1 = _AVX512_NFMA(z1, h3, q1);
        q2 = _AVX512_NFMA(z2, h3, q2);
        q3 = _AVX512_NFMA(z3, h3, q3);

        h4 = _AVX512_SET1(hh[(ldh*3)+nb-2]);

        q1 = _AVX512_NFMA(w1, h4, q1);
        q2 = _AVX512_NFMA(w2, h4, q2);
        q3 = _AVX512_NFMA(w3, h4, q3);

        h5 = _AVX512_SET1(hh[(ldh*4)+nb-1]);

        q1 = _AVX512_NFMA(v1, h5, q1);
        q2 = _AVX512_NFMA(v2, h5, q2);
        q3 = _AVX512_NFMA(v3, h5, q3);

        _AVX512_STORE(&q[nb*ldq],q1);
        _AVX512_STORE(&q[(nb*ldq)+offset],q2);
        _AVX512_STORE(&q[(nb*ldq)+2*offset],q3);

        h1 = _AVX512_SET1(hh[nb-4]);
        q1 = _AVX512_LOAD(&q[(nb+1)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+1)*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[((nb+1)*ldq)+2*offset]);

        q1 = _AVX512_NFMA(x1, h1, q1);
        q2 = _AVX512_NFMA(x2, h1, q2);
        q3 = _AVX512_NFMA(x3, h1, q3);

        h2 = _AVX512_SET1(hh[ldh+nb-3]);

        q1 = _AVX512_NFMA(y1, h2, q1);
        q2 = _AVX512_NFMA(y2, h2, q2);
        q3 = _AVX512_NFMA(y3, h2, q3);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-2]);

        q1 = _AVX512_NFMA(z1, h3, q1);
        q2 = _AVX512_NFMA(z2, h3, q2);
        q3 = _AVX512_NFMA(z3, h3, q3);

        h4 = _AVX512_SET1(hh[(ldh*3)+nb-1]);

        q1 = _AVX512_NFMA(w1, h4, q1);
        q2 = _AVX512_NFMA(w2, h4, q2);
        q3 = _AVX512_NFMA(w3, h4, q3);

        _AVX512_STORE(&q[(nb+1)*ldq],q1);
        _AVX512_STORE(&q[((nb+1)*ldq)+offset],q2);
        _AVX512_STORE(&q[((nb+1)*ldq)+2*offset],q3);

        h1 = _AVX512_SET1(hh[nb-3]);
        q1 = _AVX512_LOAD(&q[(nb+2)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+2)*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[((nb+2)*ldq)+2*offset]);

        q1 = _AVX512_NFMA(x1, h1, q1);
        q2 = _AVX512_NFMA(x2, h1, q2);
        q3 = _AVX512_NFMA(x3, h1, q3);

        h2 = _AVX512_SET1(hh[ldh+nb-2]);

        q1 = _AVX512_NFMA(y1, h2, q1);
        q2 = _AVX512_NFMA(y2, h2, q2);
        q3 = _AVX512_NFMA(y3, h2, q3);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-1]);

        q1 = _AVX512_NFMA(z1, h3, q1);
        q2 = _AVX512_NFMA(z2, h3, q2);
        q3 = _AVX512_NFMA(z3, h3, q3);

        _AVX512_STORE(&q[(nb+2)*ldq],q1);
        _AVX512_STORE(&q[((nb+2)*ldq)+offset],q2);
        _AVX512_STORE(&q[((nb+2)*ldq)+2*offset],q3);

        h1 = _AVX512_SET1(hh[nb-2]);
        q1 = _AVX512_LOAD(&q[(nb+3)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+3)*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[((nb+3)*ldq)+2*offset]);

        q1 = _AVX512_NFMA(x1, h1, q1);
        q2 = _AVX512_NFMA(x2, h1, q2);
        q3 = _AVX512_NFMA(x3, h1, q3);

        h2 = _AVX512_SET1(hh[ldh+nb-1]);

        q1 = _AVX512_NFMA(y1, h2, q1);
        q2 = _AVX512_NFMA(y2, h2, q2);
        q3 = _AVX512_NFMA(y3, h2, q3);

        _AVX512_STORE(&q[(nb+3)*ldq],q1);
        _AVX512_STORE(&q[((nb+3)*ldq)+offset],q2);
        _AVX512_STORE(&q[((nb+3)*ldq)+2*offset],q3);

        h1 = _AVX512_SET1(hh[nb-1]);
        q1 = _AVX512_LOAD(&q[(nb+4)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+4)*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[((nb+4)*ldq)+2*offset]);

        q1 = _AVX512_NFMA(x1, h1, q1);
        q2 = _AVX512_NFMA(x2, h1, q2);
        q3 = _AVX512_NFMA(x3, h1, q3);

        _AVX512_STORE(&q[(nb+4)*ldq],q1);
        _AVX512_STORE(&q[((nb+4)*ldq)+offset],q2);
        _AVX512_STORE(&q[((nb+4)*ldq)+2*offset],q3);

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
__forceinline void hh_trafo_kernel_32_AVX512_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_64_AVX512_6hv_single(float* q, float* hh, int nb, int ldq, int ldh, float* scalarprods)
#endif
{
        /////////////////////////////////////////////////////
        // Matrix Vector Multiplication, Q [8 x nb+3] * hh
        // hh contains four householder vectors
        /////////////////////////////////////////////////////
        int i;

        __AVX512_DATATYPE a1_1 = _AVX512_LOAD(&q[ldq*5]);
        __AVX512_DATATYPE a2_1 = _AVX512_LOAD(&q[ldq*4]);
        __AVX512_DATATYPE a3_1 = _AVX512_LOAD(&q[ldq*3]);
        __AVX512_DATATYPE a4_1 = _AVX512_LOAD(&q[ldq*2]);
        __AVX512_DATATYPE a5_1 = _AVX512_LOAD(&q[ldq]);
        __AVX512_DATATYPE a6_1 = _AVX512_LOAD(&q[0]);

        __AVX512_DATATYPE h_6_5 = _AVX512_SET1(hh[(ldh*5)+1]);
        __AVX512_DATATYPE h_6_4 = _AVX512_SET1(hh[(ldh*5)+2]);
        __AVX512_DATATYPE h_6_3 = _AVX512_SET1(hh[(ldh*5)+3]);
        __AVX512_DATATYPE h_6_2 = _AVX512_SET1(hh[(ldh*5)+4]);
        __AVX512_DATATYPE h_6_1 = _AVX512_SET1(hh[(ldh*5)+5]);

//        register __AVX512_DATATYPE t1 = _AVX512_FMA(a5_1, h_6_5, a6_1);
        __AVX512_DATATYPE t1 = _AVX512_FMA(a5_1, h_6_5, a6_1);

        t1 = _AVX512_FMA(a4_1, h_6_4, t1);
        t1 = _AVX512_FMA(a3_1, h_6_3, t1);
        t1 = _AVX512_FMA(a2_1, h_6_2, t1);
        t1 = _AVX512_FMA(a1_1, h_6_1, t1);

        __AVX512_DATATYPE h_5_4 = _AVX512_SET1(hh[(ldh*4)+1]);
        __AVX512_DATATYPE h_5_3 = _AVX512_SET1(hh[(ldh*4)+2]);
        __AVX512_DATATYPE h_5_2 = _AVX512_SET1(hh[(ldh*4)+3]);
        __AVX512_DATATYPE h_5_1 = _AVX512_SET1(hh[(ldh*4)+4]);

//        register __AVX512_DATATYPE v1 = _AVX512_FMA(a4_1, h_5_4, a5_1);
        __AVX512_DATATYPE v1 = _AVX512_FMA(a4_1, h_5_4, a5_1);

        v1 = _AVX512_FMA(a3_1, h_5_3, v1);
        v1 = _AVX512_FMA(a2_1, h_5_2, v1);
        v1 = _AVX512_FMA(a1_1, h_5_1, v1);

        __AVX512_DATATYPE h_4_3 = _AVX512_SET1(hh[(ldh*3)+1]);
        __AVX512_DATATYPE h_4_2 = _AVX512_SET1(hh[(ldh*3)+2]);
        __AVX512_DATATYPE h_4_1 = _AVX512_SET1(hh[(ldh*3)+3]);

//        register __AVX512_DATATYPE w1 = _AVX512_FMA(a3_1, h_4_3, a4_1);
        __AVX512_DATATYPE w1 = _AVX512_FMA(a3_1, h_4_3, a4_1);

        w1 = _AVX512_FMA(a2_1, h_4_2, w1);
        w1 = _AVX512_FMA(a1_1, h_4_1, w1);

        __AVX512_DATATYPE h_2_1 = _AVX512_SET1(hh[ldh+1]);
        __AVX512_DATATYPE h_3_2 = _AVX512_SET1(hh[(ldh*2)+1]);
        __AVX512_DATATYPE h_3_1 = _AVX512_SET1(hh[(ldh*2)+2]);

//        register __AVX512_DATATYPE z1 = _AVX512_FMA(a2_1, h_3_2, a3_1);
        __AVX512_DATATYPE z1 = _AVX512_FMA(a2_1, h_3_2, a3_1);

        z1 = _AVX512_FMA(a1_1, h_3_1, z1);
//        register __AVX512_DATATYPE y1 = _AVX512_FMA(a1_1, h_2_1, a2_1);
        __AVX512_DATATYPE y1 = _AVX512_FMA(a1_1, h_2_1, a2_1);


//        register __AVX512_DATATYPE x1 = a1_1;
        __AVX512_DATATYPE x1 = a1_1;



        __AVX512_DATATYPE a1_2 = _AVX512_LOAD(&q[(ldq*5)+offset]);
        __AVX512_DATATYPE a2_2 = _AVX512_LOAD(&q[(ldq*4)+offset]);
        __AVX512_DATATYPE a3_2 = _AVX512_LOAD(&q[(ldq*3)+offset]);
        __AVX512_DATATYPE a4_2 = _AVX512_LOAD(&q[(ldq*2)+offset]);
        __AVX512_DATATYPE a5_2 = _AVX512_LOAD(&q[(ldq)+offset]);
        __AVX512_DATATYPE a6_2 = _AVX512_LOAD(&q[0+offset]);

//        register __AVX512_DATATYPE t2 = _AVX512_FMA(a5_2, h_6_5, a6_2);
         __AVX512_DATATYPE t2 = _AVX512_FMA(a5_2, h_6_5, a6_2);

        t2 = _AVX512_FMA(a4_2, h_6_4, t2);
        t2 = _AVX512_FMA(a3_2, h_6_3, t2);
        t2 = _AVX512_FMA(a2_2, h_6_2, t2);
        t2 = _AVX512_FMA(a1_2, h_6_1, t2);

//        register __AVX512_DATATYPE v2 = _AVX512_FMA(a4_2, h_5_4, a5_2);
        __AVX512_DATATYPE v2 = _AVX512_FMA(a4_2, h_5_4, a5_2);

        v2 = _AVX512_FMA(a3_2, h_5_3, v2);
        v2 = _AVX512_FMA(a2_2, h_5_2, v2);
        v2 = _AVX512_FMA(a1_2, h_5_1, v2);

//        register __AVX512_DATATYPE w2 = _AVX512_FMA(a3_2, h_4_3, a4_2);
        __AVX512_DATATYPE w2 = _AVX512_FMA(a3_2, h_4_3, a4_2);

        w2 = _AVX512_FMA(a2_2, h_4_2, w2);
        w2 = _AVX512_FMA(a1_2, h_4_1, w2);

//        register __AVX512_DATATYPE z2 = _AVX512_FMA(a2_2, h_3_2, a3_2);
         __AVX512_DATATYPE z2 = _AVX512_FMA(a2_2, h_3_2, a3_2);

        z2 = _AVX512_FMA(a1_2, h_3_1, z2);
//        register __AVX512_DATATYPE y2 = _AVX512_FMA(a1_2, h_2_1, a2_2);
        __AVX512_DATATYPE y2 = _AVX512_FMA(a1_2, h_2_1, a2_2);


//        register __AVX512_DATATYPE x2 = a1_2;
        __AVX512_DATATYPE x2 = a1_2;


        __AVX512_DATATYPE a1_3 = _AVX512_LOAD(&q[(ldq*5)+2*offset]);
        __AVX512_DATATYPE a2_3 = _AVX512_LOAD(&q[(ldq*4)+2*offset]);
        __AVX512_DATATYPE a3_3 = _AVX512_LOAD(&q[(ldq*3)+2*offset]);
        __AVX512_DATATYPE a4_3 = _AVX512_LOAD(&q[(ldq*2)+2*offset]);
        __AVX512_DATATYPE a5_3 = _AVX512_LOAD(&q[(ldq)+2*offset]);
        __AVX512_DATATYPE a6_3 = _AVX512_LOAD(&q[0+2*offset]);

//        register __AVX512_DATATYPE t3 = _AVX512_FMA(a5_3, h_6_5, a6_3);
        __AVX512_DATATYPE t3 = _AVX512_FMA(a5_3, h_6_5, a6_3);

        t3 = _AVX512_FMA(a4_3, h_6_4, t3);
        t3 = _AVX512_FMA(a3_3, h_6_3, t3);
        t3 = _AVX512_FMA(a2_3, h_6_2, t3);
        t3 = _AVX512_FMA(a1_3, h_6_1, t3);

//        register __AVX512_DATATYPE v3 = _AVX512_FMA(a4_3, h_5_4, a5_3);
        __AVX512_DATATYPE v3 = _AVX512_FMA(a4_3, h_5_4, a5_3);

        v3 = _AVX512_FMA(a3_3, h_5_3, v3);
        v3 = _AVX512_FMA(a2_3, h_5_2, v3);
        v3 = _AVX512_FMA(a1_3, h_5_1, v3);

//        register __AVX512_DATATYPE w3 = _AVX512_FMA(a3_3, h_4_3, a4_3);
        __AVX512_DATATYPE w3 = _AVX512_FMA(a3_3, h_4_3, a4_3);

        w3 = _AVX512_FMA(a2_3, h_4_2, w3);
        w3 = _AVX512_FMA(a1_3, h_4_1, w3);

//        register __AVX512_DATATYPE z3 = _AVX512_FMA(a2_3, h_3_2, a3_3);
        __AVX512_DATATYPE z3 = _AVX512_FMA(a2_3, h_3_2, a3_3);

        z3 = _AVX512_FMA(a1_3, h_3_1, z3);
//        register __AVX512_DATATYPE y3 = _AVX512_FMA(a1_3, h_2_1, a2_3);
        __AVX512_DATATYPE y3 = _AVX512_FMA(a1_3, h_2_1, a2_3);


//        register __AVX512_DATATYPE x3 = a1_3;
        __AVX512_DATATYPE x3 = a1_3;


        __AVX512_DATATYPE a1_4 = _AVX512_LOAD(&q[(ldq*5)+3*offset]);
        __AVX512_DATATYPE a2_4 = _AVX512_LOAD(&q[(ldq*4)+3*offset]);
        __AVX512_DATATYPE a3_4 = _AVX512_LOAD(&q[(ldq*3)+3*offset]);
        __AVX512_DATATYPE a4_4 = _AVX512_LOAD(&q[(ldq*2)+3*offset]);
        __AVX512_DATATYPE a5_4 = _AVX512_LOAD(&q[(ldq)+3*offset]);
        __AVX512_DATATYPE a6_4 = _AVX512_LOAD(&q[0+3*offset]);

//        register __AVX512_DATATYPE t4 = _AVX512_FMA(a5_4, h_6_5, a6_4);
        __AVX512_DATATYPE t4 = _AVX512_FMA(a5_4, h_6_5, a6_4);

        t4 = _AVX512_FMA(a4_4, h_6_4, t4);
        t4 = _AVX512_FMA(a3_4, h_6_3, t4);
        t4 = _AVX512_FMA(a2_4, h_6_2, t4);
        t4 = _AVX512_FMA(a1_4, h_6_1, t4);

//        register __AVX512_DATATYPE v4 = _AVX512_FMA(a4_4, h_5_4, a5_4);
        __AVX512_DATATYPE v4 = _AVX512_FMA(a4_4, h_5_4, a5_4);

        v4 = _AVX512_FMA(a3_4, h_5_3, v4);
        v4 = _AVX512_FMA(a2_4, h_5_2, v4);
        v4 = _AVX512_FMA(a1_4, h_5_1, v4);

//        register __AVX512_DATATYPE w4 = _AVX512_FMA(a3_4, h_4_3, a4_4);
        __AVX512_DATATYPE w4 = _AVX512_FMA(a3_4, h_4_3, a4_4);

        w4 = _AVX512_FMA(a2_4, h_4_2, w4);
        w4 = _AVX512_FMA(a1_4, h_4_1, w4);

//        register __AVX512_DATATYPE z4 = _AVX512_FMA(a2_4, h_3_2, a3_4);
        __AVX512_DATATYPE z4 = _AVX512_FMA(a2_4, h_3_2, a3_4);

        z4 = _AVX512_FMA(a1_4, h_3_1, z4);
//        register __AVX512_DATATYPE y4 = _AVX512_FMA(a1_4, h_2_1, a2_4);
         __AVX512_DATATYPE y4 = _AVX512_FMA(a1_4, h_2_1, a2_4);


//        register __AVX512_DATATYPE x4 = a1_4;
        __AVX512_DATATYPE x4 = a1_4;


        __AVX512_DATATYPE q1;
        __AVX512_DATATYPE q2;
        __AVX512_DATATYPE q3;
        __AVX512_DATATYPE q4;

        __AVX512_DATATYPE h1;
        __AVX512_DATATYPE h2;
        __AVX512_DATATYPE h3;
        __AVX512_DATATYPE h4;
        __AVX512_DATATYPE h5;
        __AVX512_DATATYPE h6;

        for(i = 6; i < nb; i++)
        {
                h1 = _AVX512_SET1(hh[i-5]);
                q1 = _AVX512_LOAD(&q[i*ldq]);
                q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);
                q3 = _AVX512_LOAD(&q[(i*ldq)+2*offset]);
                q4 = _AVX512_LOAD(&q[(i*ldq)+3*offset]);

                x1 = _AVX512_FMA(q1, h1, x1);
                x2 = _AVX512_FMA(q2, h1, x2);
                x3 = _AVX512_FMA(q3, h1, x3);
                x4 = _AVX512_FMA(q4, h1, x4);

                h2 = _AVX512_SET1(hh[ldh+i-4]);

                y1 = _AVX512_FMA(q1, h2, y1);
                y2 = _AVX512_FMA(q2, h2, y2);
                y3 = _AVX512_FMA(q3, h2, y3);
                y4 = _AVX512_FMA(q4, h2, y4);

                h3 = _AVX512_SET1(hh[(ldh*2)+i-3]);

                z1 = _AVX512_FMA(q1, h3, z1);
                z2 = _AVX512_FMA(q2, h3, z2);
                z3 = _AVX512_FMA(q3, h3, z3);
                z4 = _AVX512_FMA(q4, h3, z4);

                h4 = _AVX512_SET1(hh[(ldh*3)+i-2]);

                w1 = _AVX512_FMA(q1, h4, w1);
                w2 = _AVX512_FMA(q2, h4, w2);
                w3 = _AVX512_FMA(q3, h4, w3);
                w4 = _AVX512_FMA(q4, h4, w4);

                h5 = _AVX512_SET1(hh[(ldh*4)+i-1]);

                v1 = _AVX512_FMA(q1, h5, v1);
                v2 = _AVX512_FMA(q2, h5, v2);
                v3 = _AVX512_FMA(q3, h5, v3);
                v4 = _AVX512_FMA(q4, h5, v4);

                h6 = _AVX512_SET1(hh[(ldh*5)+i]);

                t1 = _AVX512_FMA(q1, h6, t1);
                t2 = _AVX512_FMA(q2, h6, t2);
                t3 = _AVX512_FMA(q3, h6, t3);
                t4 = _AVX512_FMA(q4, h6, t4);
        }

        h1 = _AVX512_SET1(hh[nb-5]);
        q1 = _AVX512_LOAD(&q[nb*ldq]);
        q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[(nb*ldq)+2*offset]);
        q4 = _AVX512_LOAD(&q[(nb*ldq)+3*offset]);

        x1 = _AVX512_FMA(q1, h1, x1);
        x2 = _AVX512_FMA(q2, h1, x2);
        x3 = _AVX512_FMA(q3, h1, x3);
        x4 = _AVX512_FMA(q4, h1, x4);

        h2 = _AVX512_SET1(hh[ldh+nb-4]);

        y1 = _AVX512_FMA(q1, h2, y1);
        y2 = _AVX512_FMA(q2, h2, y2);
        y3 = _AVX512_FMA(q3, h2, y3);
        y4 = _AVX512_FMA(q4, h2, y4);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-3]);

        z1 = _AVX512_FMA(q1, h3, z1);
        z2 = _AVX512_FMA(q2, h3, z2);
        z3 = _AVX512_FMA(q3, h3, z3);
        z4 = _AVX512_FMA(q4, h3, z4);

        h4 = _AVX512_SET1(hh[(ldh*3)+nb-2]);

        w1 = _AVX512_FMA(q1, h4, w1);
        w2 = _AVX512_FMA(q2, h4, w2);
        w3 = _AVX512_FMA(q3, h4, w3);
        w4 = _AVX512_FMA(q4, h4, w4);

        h5 = _AVX512_SET1(hh[(ldh*4)+nb-1]);

        v1 = _AVX512_FMA(q1, h5, v1);
        v2 = _AVX512_FMA(q2, h5, v2);
        v3 = _AVX512_FMA(q3, h5, v3);
        v4 = _AVX512_FMA(q4, h5, v4);

        h1 = _AVX512_SET1(hh[nb-4]);

        q1 = _AVX512_LOAD(&q[(nb+1)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+1)*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[((nb+1)*ldq)+2*offset]);
        q4 = _AVX512_LOAD(&q[((nb+1)*ldq)+3*offset]);

        x1 = _AVX512_FMA(q1, h1, x1);
        x2 = _AVX512_FMA(q2, h1, x2);
        x3 = _AVX512_FMA(q3, h1, x3);
        x4 = _AVX512_FMA(q4, h1, x4);

        h2 = _AVX512_SET1(hh[ldh+nb-3]);

        y1 = _AVX512_FMA(q1, h2, y1);
        y2 = _AVX512_FMA(q2, h2, y2);
        y3 = _AVX512_FMA(q3, h2, y3);
        y4 = _AVX512_FMA(q4, h2, y4);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-2]);

        z1 = _AVX512_FMA(q1, h3, z1);
        z2 = _AVX512_FMA(q2, h3, z2);
        z3 = _AVX512_FMA(q3, h3, z3);
        z4 = _AVX512_FMA(q4, h3, z4);

        h4 = _AVX512_SET1(hh[(ldh*3)+nb-1]);

        w1 = _AVX512_FMA(q1, h4, w1);
        w2 = _AVX512_FMA(q2, h4, w2);
        w3 = _AVX512_FMA(q3, h4, w3);
        w4 = _AVX512_FMA(q4, h4, w4);

        h1 = _AVX512_SET1(hh[nb-3]);
        q1 = _AVX512_LOAD(&q[(nb+2)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+2)*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[((nb+2)*ldq)+2*offset]);
        q4 = _AVX512_LOAD(&q[((nb+2)*ldq)+3*offset]);

        x1 = _AVX512_FMA(q1, h1, x1);
        x2 = _AVX512_FMA(q2, h1, x2);
        x3 = _AVX512_FMA(q3, h1, x3);
        x4 = _AVX512_FMA(q4, h1, x4);

        h2 = _AVX512_SET1(hh[ldh+nb-2]);

        y1 = _AVX512_FMA(q1, h2, y1);
        y2 = _AVX512_FMA(q2, h2, y2);
        y3 = _AVX512_FMA(q3, h2, y3);
        y4 = _AVX512_FMA(q4, h2, y4);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-1]);

        z1 = _AVX512_FMA(q1, h3, z1);
        z2 = _AVX512_FMA(q2, h3, z2);
        z3 = _AVX512_FMA(q3, h3, z3);
        z4 = _AVX512_FMA(q4, h3, z4);

        h1 = _AVX512_SET1(hh[nb-2]);
        q1 = _AVX512_LOAD(&q[(nb+3)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+3)*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[((nb+3)*ldq)+2*offset]);
        q4 = _AVX512_LOAD(&q[((nb+3)*ldq)+3*offset]);

        x1 = _AVX512_FMA(q1, h1, x1);
        x2 = _AVX512_FMA(q2, h1, x2);
        x3 = _AVX512_FMA(q3, h1, x3);
        x4 = _AVX512_FMA(q4, h1, x4);

        h2 = _AVX512_SET1(hh[ldh+nb-1]);

        y1 = _AVX512_FMA(q1, h2, y1);
        y2 = _AVX512_FMA(q2, h2, y2);
        y3 = _AVX512_FMA(q3, h2, y3);
        y4 = _AVX512_FMA(q4, h2, y4);

        h1 = _AVX512_SET1(hh[nb-1]);
        q1 = _AVX512_LOAD(&q[(nb+4)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+4)*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[((nb+4)*ldq)+2*offset]);
        q4 = _AVX512_LOAD(&q[((nb+4)*ldq)+3*offset]);

        x1 = _AVX512_FMA(q1, h1, x1);
        x2 = _AVX512_FMA(q2, h1, x2);
        x3 = _AVX512_FMA(q3, h1, x3);
        x4 = _AVX512_FMA(q4, h1, x4);

        /////////////////////////////////////////////////////
        // Apply tau, correct wrong calculation using pre-calculated scalar products
        /////////////////////////////////////////////////////

        __AVX512_DATATYPE tau1 = _AVX512_SET1(hh[0]);
        x1 = _AVX512_MUL(x1, tau1);
        x2 = _AVX512_MUL(x2, tau1);
        x3 = _AVX512_MUL(x3, tau1);
        x4 = _AVX512_MUL(x4, tau1);

        __AVX512_DATATYPE tau2 = _AVX512_SET1(hh[ldh]);
        __AVX512_DATATYPE vs_1_2 = _AVX512_SET1(scalarprods[0]);
        h2 = _AVX512_MUL(tau2, vs_1_2);

        y1 = _AVX512_FMSUB(y1, tau2, _AVX512_MUL(x1,h2));
        y2 = _AVX512_FMSUB(y2, tau2, _AVX512_MUL(x2,h2));
        y3 = _AVX512_FMSUB(y3, tau2, _AVX512_MUL(x3,h2));
        y4 = _AVX512_FMSUB(y4, tau2, _AVX512_MUL(x4,h2));

        __AVX512_DATATYPE tau3 = _AVX512_SET1(hh[ldh*2]);
        __AVX512_DATATYPE vs_1_3 = _AVX512_SET1(scalarprods[1]);
        __AVX512_DATATYPE vs_2_3 = _AVX512_SET1(scalarprods[2]);

        h2 = _AVX512_MUL(tau3, vs_1_3);
        h3 = _AVX512_MUL(tau3, vs_2_3);

        z1 = _AVX512_FMSUB(z1, tau3, _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2)));
        z2 = _AVX512_FMSUB(z2, tau3, _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2)));
        z3 = _AVX512_FMSUB(z3, tau3, _AVX512_FMA(y3, h3, _AVX512_MUL(x3,h2)));
        z4 = _AVX512_FMSUB(z4, tau3, _AVX512_FMA(y4, h3, _AVX512_MUL(x4,h2)));

        __AVX512_DATATYPE tau4 = _AVX512_SET1(hh[ldh*3]);
        __AVX512_DATATYPE vs_1_4 = _AVX512_SET1(scalarprods[3]);
        __AVX512_DATATYPE vs_2_4 = _AVX512_SET1(scalarprods[4]);

        h2 = _AVX512_MUL(tau4, vs_1_4);
        h3 = _AVX512_MUL(tau4, vs_2_4);

        __AVX512_DATATYPE vs_3_4 = _AVX512_SET1(scalarprods[5]);
        h4 = _AVX512_MUL(tau4, vs_3_4);

        w1 = _AVX512_FMSUB(w1, tau4, _AVX512_FMA(z1, h4, _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2))));
        w2 = _AVX512_FMSUB(w2, tau4, _AVX512_FMA(z2, h4, _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2))));
        w3 = _AVX512_FMSUB(w3, tau4, _AVX512_FMA(z3, h4, _AVX512_FMA(y3, h3, _AVX512_MUL(x3,h2))));
        w4 = _AVX512_FMSUB(w4, tau4, _AVX512_FMA(z4, h4, _AVX512_FMA(y4, h3, _AVX512_MUL(x4,h2))));

        __AVX512_DATATYPE tau5 = _AVX512_SET1(hh[ldh*4]);
        __AVX512_DATATYPE vs_1_5 = _AVX512_SET1(scalarprods[6]);
        __AVX512_DATATYPE vs_2_5 = _AVX512_SET1(scalarprods[7]);

        h2 = _AVX512_MUL(tau5, vs_1_5);
        h3 = _AVX512_MUL(tau5, vs_2_5);

        __AVX512_DATATYPE vs_3_5 = _AVX512_SET1(scalarprods[8]);
        __AVX512_DATATYPE vs_4_5 = _AVX512_SET1(scalarprods[9]);

        h4 = _AVX512_MUL(tau5, vs_3_5);
        h5 = _AVX512_MUL(tau5, vs_4_5);

        v1 = _AVX512_FMSUB(v1, tau5, _AVX512_ADD(_AVX512_FMA(w1, h5, _AVX512_MUL(z1,h4)), _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2))));
        v2 = _AVX512_FMSUB(v2, tau5, _AVX512_ADD(_AVX512_FMA(w2, h5, _AVX512_MUL(z2,h4)), _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2))));
        v3 = _AVX512_FMSUB(v3, tau5, _AVX512_ADD(_AVX512_FMA(w3, h5, _AVX512_MUL(z3,h4)), _AVX512_FMA(y3, h3, _AVX512_MUL(x3,h2))));
        v4 = _AVX512_FMSUB(v4, tau5, _AVX512_ADD(_AVX512_FMA(w4, h5, _AVX512_MUL(z4,h4)), _AVX512_FMA(y4, h3, _AVX512_MUL(x4,h2))));

        __AVX512_DATATYPE tau6 = _AVX512_SET1(hh[ldh*5]);
        __AVX512_DATATYPE vs_1_6 = _AVX512_SET1(scalarprods[10]);
        __AVX512_DATATYPE vs_2_6 = _AVX512_SET1(scalarprods[11]);
        h2 = _AVX512_MUL(tau6, vs_1_6);
        h3 = _AVX512_MUL(tau6, vs_2_6);

        __AVX512_DATATYPE vs_3_6 = _AVX512_SET1(scalarprods[12]);
        __AVX512_DATATYPE vs_4_6 = _AVX512_SET1(scalarprods[13]);
        __AVX512_DATATYPE vs_5_6 = _AVX512_SET1(scalarprods[14]);

        h4 = _AVX512_MUL(tau6, vs_3_6);
        h5 = _AVX512_MUL(tau6, vs_4_6);
        h6 = _AVX512_MUL(tau6, vs_5_6);

        t1 = _AVX512_FMSUB(t1, tau6, _AVX512_FMA(v1, h6, _AVX512_ADD(_AVX512_FMA(w1, h5, _AVX512_MUL(z1,h4)), _AVX512_FMA(y1, h3, _AVX512_MUL(x1,h2)))));
        t2 = _AVX512_FMSUB(t2, tau6, _AVX512_FMA(v2, h6, _AVX512_ADD(_AVX512_FMA(w2, h5, _AVX512_MUL(z2,h4)), _AVX512_FMA(y2, h3, _AVX512_MUL(x2,h2)))));
        t3 = _AVX512_FMSUB(t3, tau6, _AVX512_FMA(v3, h6, _AVX512_ADD(_AVX512_FMA(w3, h5, _AVX512_MUL(z3,h4)), _AVX512_FMA(y3, h3, _AVX512_MUL(x3,h2)))));
        t4 = _AVX512_FMSUB(t4, tau6, _AVX512_FMA(v4, h6, _AVX512_ADD(_AVX512_FMA(w4, h5, _AVX512_MUL(z4,h4)), _AVX512_FMA(y4, h3, _AVX512_MUL(x4,h2)))));


        /////////////////////////////////////////////////////
        // Rank-1 update of Q [8 x nb+3]
        /////////////////////////////////////////////////////

        q1 = _AVX512_LOAD(&q[0]);
        q2 = _AVX512_LOAD(&q[0+offset]);
        q3 = _AVX512_LOAD(&q[0+2*offset]);
        q4 = _AVX512_LOAD(&q[0+3*offset]);

        q1 = _AVX512_SUB(q1, t1);
        q2 = _AVX512_SUB(q2, t2);
        q3 = _AVX512_SUB(q3, t3);
        q4 = _AVX512_SUB(q4, t4);

        _AVX512_STORE(&q[0],q1);
        _AVX512_STORE(&q[0+offset],q2);
        _AVX512_STORE(&q[0+2*offset],q3);
        _AVX512_STORE(&q[0+3*offset],q4);

        h6 = _AVX512_SET1(hh[(ldh*5)+1]);
        q1 = _AVX512_LOAD(&q[ldq]);
        q2 = _AVX512_LOAD(&q[ldq+offset]);
        q3 = _AVX512_LOAD(&q[ldq+2*offset]);
        q4 = _AVX512_LOAD(&q[ldq+3*offset]);

        q1 = _AVX512_SUB(q1, v1);
        q2 = _AVX512_SUB(q2, v2);
        q3 = _AVX512_SUB(q3, v3);
        q4 = _AVX512_SUB(q4, v4);

        q1 = _AVX512_NFMA(t1, h6, q1);
        q2 = _AVX512_NFMA(t2, h6, q2);
        q3 = _AVX512_NFMA(t3, h6, q3);
        q4 = _AVX512_NFMA(t4, h6, q4);

        _AVX512_STORE(&q[ldq],q1);
        _AVX512_STORE(&q[ldq+offset],q2);
        _AVX512_STORE(&q[ldq+2*offset],q3);
        _AVX512_STORE(&q[ldq+3*offset],q4);

        h5 = _AVX512_SET1(hh[(ldh*4)+1]);
        q1 = _AVX512_LOAD(&q[ldq*2]);
        q2 = _AVX512_LOAD(&q[(ldq*2)+offset]);
        q3 = _AVX512_LOAD(&q[(ldq*2)+2*offset]);
        q4 = _AVX512_LOAD(&q[(ldq*2)+3*offset]);

        q1 = _AVX512_SUB(q1, w1);
        q2 = _AVX512_SUB(q2, w2);
        q3 = _AVX512_SUB(q3, w3);
        q4 = _AVX512_SUB(q4, w4);

        q1 = _AVX512_NFMA(v1, h5, q1);
        q2 = _AVX512_NFMA(v2, h5, q2);
        q3 = _AVX512_NFMA(v3, h5, q3);
        q4 = _AVX512_NFMA(v4, h5, q4);

        h6 = _AVX512_SET1(hh[(ldh*5)+2]);

        q1 = _AVX512_NFMA(t1, h6, q1);
        q2 = _AVX512_NFMA(t2, h6, q2);
        q3 = _AVX512_NFMA(t3, h6, q3);
        q4 = _AVX512_NFMA(t4, h6, q4);

        _AVX512_STORE(&q[ldq*2],q1);
        _AVX512_STORE(&q[(ldq*2)+offset],q2);
        _AVX512_STORE(&q[(ldq*2)+2*offset],q3);
        _AVX512_STORE(&q[(ldq*2)+3*offset],q4);

        h4 = _AVX512_SET1(hh[(ldh*3)+1]);
        q1 = _AVX512_LOAD(&q[ldq*3]);
        q2 = _AVX512_LOAD(&q[(ldq*3)+offset]);
        q3 = _AVX512_LOAD(&q[(ldq*3)+2*offset]);
        q4 = _AVX512_LOAD(&q[(ldq*3)+3*offset]);

        q1 = _AVX512_SUB(q1, z1);
        q2 = _AVX512_SUB(q2, z2);
        q3 = _AVX512_SUB(q3, z3);
        q4 = _AVX512_SUB(q4, z4);

        q1 = _AVX512_NFMA(w1, h4, q1);
        q2 = _AVX512_NFMA(w2, h4, q2);
        q3 = _AVX512_NFMA(w3, h4, q3);
        q4 = _AVX512_NFMA(w4, h4, q4);

        h5 = _AVX512_SET1(hh[(ldh*4)+2]);

        q1 = _AVX512_NFMA(v1, h5, q1);
        q2 = _AVX512_NFMA(v2, h5, q2);
        q3 = _AVX512_NFMA(v3, h5, q3);
        q4 = _AVX512_NFMA(v4, h5, q4);

        h6 = _AVX512_SET1(hh[(ldh*5)+3]);

        q1 = _AVX512_NFMA(t1, h6, q1);
        q2 = _AVX512_NFMA(t2, h6, q2);
        q3 = _AVX512_NFMA(t3, h6, q3);
        q4 = _AVX512_NFMA(t4, h6, q4);

        _AVX512_STORE(&q[ldq*3],q1);
        _AVX512_STORE(&q[(ldq*3)+offset],q2);
        _AVX512_STORE(&q[(ldq*3)+2*offset],q3);
        _AVX512_STORE(&q[(ldq*3)+3*offset],q4);

        h3 = _AVX512_SET1(hh[(ldh*2)+1]);
        q1 = _AVX512_LOAD(&q[ldq*4]);
        q2 = _AVX512_LOAD(&q[(ldq*4)+offset]);
        q3 = _AVX512_LOAD(&q[(ldq*4)+2*offset]);
        q4 = _AVX512_LOAD(&q[(ldq*4)+3*offset]);

        q1 = _AVX512_SUB(q1, y1);
        q2 = _AVX512_SUB(q2, y2);
        q3 = _AVX512_SUB(q3, y3);
        q4 = _AVX512_SUB(q4, y4);

        q1 = _AVX512_NFMA(z1, h3, q1);
        q2 = _AVX512_NFMA(z2, h3, q2);
        q3 = _AVX512_NFMA(z3, h3, q3);
        q4 = _AVX512_NFMA(z4, h3, q4);

        h4 = _AVX512_SET1(hh[(ldh*3)+2]);

        q1 = _AVX512_NFMA(w1, h4, q1);
        q2 = _AVX512_NFMA(w2, h4, q2);
        q3 = _AVX512_NFMA(w3, h4, q3);
        q4 = _AVX512_NFMA(w4, h4, q4);

        h5 = _AVX512_SET1(hh[(ldh*4)+3]);

        q1 = _AVX512_NFMA(v1, h5, q1);
        q2 = _AVX512_NFMA(v2, h5, q2);
        q3 = _AVX512_NFMA(v3, h5, q3);
        q4 = _AVX512_NFMA(v4, h5, q4);

        h6 = _AVX512_SET1(hh[(ldh*5)+4]);

        q1 = _AVX512_NFMA(t1, h6, q1);
        q2 = _AVX512_NFMA(t2, h6, q2);
        q3 = _AVX512_NFMA(t3, h6, q3);
        q4 = _AVX512_NFMA(t4, h6, q4);

        _AVX512_STORE(&q[ldq*4],q1);
        _AVX512_STORE(&q[(ldq*4)+offset],q2);
        _AVX512_STORE(&q[(ldq*4)+2*offset],q3);
        _AVX512_STORE(&q[(ldq*4)+3*offset],q4);

        h2 = _AVX512_SET1(hh[(ldh)+1]);
        q1 = _AVX512_LOAD(&q[ldq*5]);
        q2 = _AVX512_LOAD(&q[(ldq*5)+offset]);
        q3 = _AVX512_LOAD(&q[(ldq*5)+2*offset]);
        q4 = _AVX512_LOAD(&q[(ldq*5)+3*offset]);

        q1 = _AVX512_SUB(q1, x1);
        q2 = _AVX512_SUB(q2, x2);
        q3 = _AVX512_SUB(q3, x3);
        q4 = _AVX512_SUB(q4, x4);

        q1 = _AVX512_NFMA(y1, h2, q1);
        q2 = _AVX512_NFMA(y2, h2, q2);
        q3 = _AVX512_NFMA(y3, h2, q3);
        q4 = _AVX512_NFMA(y4, h2, q4);

        h3 = _AVX512_SET1(hh[(ldh*2)+2]);

        q1 = _AVX512_NFMA(z1, h3, q1);
        q2 = _AVX512_NFMA(z2, h3, q2);
        q3 = _AVX512_NFMA(z3, h3, q3);
        q4 = _AVX512_NFMA(z4, h3, q4);

        h4 = _AVX512_SET1(hh[(ldh*3)+3]);

        q1 = _AVX512_NFMA(w1, h4, q1);
        q2 = _AVX512_NFMA(w2, h4, q2);
        q3 = _AVX512_NFMA(w3, h4, q3);
        q4 = _AVX512_NFMA(w4, h4, q4);

        h5 = _AVX512_SET1(hh[(ldh*4)+4]);

        q1 = _AVX512_NFMA(v1, h5, q1);
        q2 = _AVX512_NFMA(v2, h5, q2);
        q3 = _AVX512_NFMA(v3, h5, q3);
        q4 = _AVX512_NFMA(v4, h5, q4);

        h6 = _AVX512_SET1(hh[(ldh*5)+5]);

        q1 = _AVX512_NFMA(t1, h6, q1);
        q2 = _AVX512_NFMA(t2, h6, q2);
        q3 = _AVX512_NFMA(t3, h6, q3);
        q4 = _AVX512_NFMA(t4, h6, q4);

        _AVX512_STORE(&q[ldq*5],q1);
        _AVX512_STORE(&q[(ldq*5)+offset],q2);
        _AVX512_STORE(&q[(ldq*5)+2*offset],q3);
        _AVX512_STORE(&q[(ldq*5)+3*offset],q4);

        for (i = 6; i < nb; i++)
        {
                q1 = _AVX512_LOAD(&q[i*ldq]);
                q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);
                q3 = _AVX512_LOAD(&q[(i*ldq)+2*offset]);
                q4 = _AVX512_LOAD(&q[(i*ldq)+3*offset]);

                h1 = _AVX512_SET1(hh[i-5]);

                q1 = _AVX512_NFMA(x1, h1, q1);
                q2 = _AVX512_NFMA(x2, h1, q2);
                q3 = _AVX512_NFMA(x3, h1, q3);
                q4 = _AVX512_NFMA(x4, h1, q4);

                h2 = _AVX512_SET1(hh[ldh+i-4]);

                q1 = _AVX512_NFMA(y1, h2, q1);
                q2 = _AVX512_NFMA(y2, h2, q2);
                q3 = _AVX512_NFMA(y3, h2, q3);
                q4 = _AVX512_NFMA(y4, h2, q4);

                h3 = _AVX512_SET1(hh[(ldh*2)+i-3]);

                q1 = _AVX512_NFMA(z1, h3, q1);
                q2 = _AVX512_NFMA(z2, h3, q2);
                q3 = _AVX512_NFMA(z3, h3, q3);
                q4 = _AVX512_NFMA(z4, h3, q4);

                h4 = _AVX512_SET1(hh[(ldh*3)+i-2]);

                q1 = _AVX512_NFMA(w1, h4, q1);
                q2 = _AVX512_NFMA(w2, h4, q2);
                q3 = _AVX512_NFMA(w3, h4, q3);
                q4 = _AVX512_NFMA(w4, h4, q4);

                h5 = _AVX512_SET1(hh[(ldh*4)+i-1]);

                q1 = _AVX512_NFMA(v1, h5, q1);
                q2 = _AVX512_NFMA(v2, h5, q2);
                q3 = _AVX512_NFMA(v3, h5, q3);
                q4 = _AVX512_NFMA(v4, h5, q4);

                h6 = _AVX512_SET1(hh[(ldh*5)+i]);

                q1 = _AVX512_NFMA(t1, h6, q1);
                q2 = _AVX512_NFMA(t2, h6, q2);
                q3 = _AVX512_NFMA(t3, h6, q3);
                q4 = _AVX512_NFMA(t4, h6, q4);

                _AVX512_STORE(&q[i*ldq],q1);
                _AVX512_STORE(&q[(i*ldq)+offset],q2);
                _AVX512_STORE(&q[(i*ldq)+2*offset],q3);
                _AVX512_STORE(&q[(i*ldq)+3*offset],q4);

        }

        h1 = _AVX512_SET1(hh[nb-5]);
        q1 = _AVX512_LOAD(&q[nb*ldq]);
        q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[(nb*ldq)+2*offset]);
        q4 = _AVX512_LOAD(&q[(nb*ldq)+3*offset]);

        q1 = _AVX512_NFMA(x1, h1, q1);
        q2 = _AVX512_NFMA(x2, h1, q2);
        q3 = _AVX512_NFMA(x3, h1, q3);
        q4 = _AVX512_NFMA(x4, h1, q4);

        h2 = _AVX512_SET1(hh[ldh+nb-4]);

        q1 = _AVX512_NFMA(y1, h2, q1);
        q2 = _AVX512_NFMA(y2, h2, q2);
        q3 = _AVX512_NFMA(y3, h2, q3);
        q4 = _AVX512_NFMA(y4, h2, q4);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-3]);

        q1 = _AVX512_NFMA(z1, h3, q1);
        q2 = _AVX512_NFMA(z2, h3, q2);
        q3 = _AVX512_NFMA(z3, h3, q3);
        q4 = _AVX512_NFMA(z4, h3, q4);

        h4 = _AVX512_SET1(hh[(ldh*3)+nb-2]);

        q1 = _AVX512_NFMA(w1, h4, q1);
        q2 = _AVX512_NFMA(w2, h4, q2);
        q3 = _AVX512_NFMA(w3, h4, q3);
        q4 = _AVX512_NFMA(w4, h4, q4);

        h5 = _AVX512_SET1(hh[(ldh*4)+nb-1]);

        q1 = _AVX512_NFMA(v1, h5, q1);
        q2 = _AVX512_NFMA(v2, h5, q2);
        q3 = _AVX512_NFMA(v3, h5, q3);
        q4 = _AVX512_NFMA(v4, h5, q4);

        _AVX512_STORE(&q[nb*ldq],q1);
        _AVX512_STORE(&q[(nb*ldq)+offset],q2);
        _AVX512_STORE(&q[(nb*ldq)+2*offset],q3);
        _AVX512_STORE(&q[(nb*ldq)+3*offset],q4);

        h1 = _AVX512_SET1(hh[nb-4]);
        q1 = _AVX512_LOAD(&q[(nb+1)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+1)*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[((nb+1)*ldq)+2*offset]);
        q4 = _AVX512_LOAD(&q[((nb+1)*ldq)+3*offset]);

        q1 = _AVX512_NFMA(x1, h1, q1);
        q2 = _AVX512_NFMA(x2, h1, q2);
        q3 = _AVX512_NFMA(x3, h1, q3);
        q4 = _AVX512_NFMA(x4, h1, q4);

        h2 = _AVX512_SET1(hh[ldh+nb-3]);

        q1 = _AVX512_NFMA(y1, h2, q1);
        q2 = _AVX512_NFMA(y2, h2, q2);
        q3 = _AVX512_NFMA(y3, h2, q3);
        q4 = _AVX512_NFMA(y4, h2, q4);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-2]);

        q1 = _AVX512_NFMA(z1, h3, q1);
        q2 = _AVX512_NFMA(z2, h3, q2);
        q3 = _AVX512_NFMA(z3, h3, q3);
        q4 = _AVX512_NFMA(z4, h3, q4);

        h4 = _AVX512_SET1(hh[(ldh*3)+nb-1]);

        q1 = _AVX512_NFMA(w1, h4, q1);
        q2 = _AVX512_NFMA(w2, h4, q2);
        q3 = _AVX512_NFMA(w3, h4, q3);
        q4 = _AVX512_NFMA(w4, h4, q4);

        _AVX512_STORE(&q[(nb+1)*ldq],q1);
        _AVX512_STORE(&q[((nb+1)*ldq)+offset],q2);
        _AVX512_STORE(&q[((nb+1)*ldq)+2*offset],q3);
        _AVX512_STORE(&q[((nb+1)*ldq)+3*offset],q4);

        h1 = _AVX512_SET1(hh[nb-3]);
        q1 = _AVX512_LOAD(&q[(nb+2)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+2)*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[((nb+2)*ldq)+2*offset]);
        q4 = _AVX512_LOAD(&q[((nb+2)*ldq)+3*offset]);

        q1 = _AVX512_NFMA(x1, h1, q1);
        q2 = _AVX512_NFMA(x2, h1, q2);
        q3 = _AVX512_NFMA(x3, h1, q3);
        q4 = _AVX512_NFMA(x4, h1, q4);

        h2 = _AVX512_SET1(hh[ldh+nb-2]);

        q1 = _AVX512_NFMA(y1, h2, q1);
        q2 = _AVX512_NFMA(y2, h2, q2);
        q3 = _AVX512_NFMA(y3, h2, q3);
        q4 = _AVX512_NFMA(y4, h2, q4);

        h3 = _AVX512_SET1(hh[(ldh*2)+nb-1]);

        q1 = _AVX512_NFMA(z1, h3, q1);
        q2 = _AVX512_NFMA(z2, h3, q2);
        q3 = _AVX512_NFMA(z3, h3, q3);
        q4 = _AVX512_NFMA(z4, h3, q4);

        _AVX512_STORE(&q[(nb+2)*ldq],q1);
        _AVX512_STORE(&q[((nb+2)*ldq)+offset],q2);
        _AVX512_STORE(&q[((nb+2)*ldq)+2*offset],q3);
        _AVX512_STORE(&q[((nb+2)*ldq)+3*offset],q4);

        h1 = _AVX512_SET1(hh[nb-2]);
        q1 = _AVX512_LOAD(&q[(nb+3)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+3)*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[((nb+3)*ldq)+2*offset]);
        q4 = _AVX512_LOAD(&q[((nb+3)*ldq)+3*offset]);

        q1 = _AVX512_NFMA(x1, h1, q1);
        q2 = _AVX512_NFMA(x2, h1, q2);
        q3 = _AVX512_NFMA(x3, h1, q3);
        q4 = _AVX512_NFMA(x4, h1, q4);

        h2 = _AVX512_SET1(hh[ldh+nb-1]);

        q1 = _AVX512_NFMA(y1, h2, q1);
        q2 = _AVX512_NFMA(y2, h2, q2);
        q3 = _AVX512_NFMA(y3, h2, q3);
        q4 = _AVX512_NFMA(y4, h2, q4);

        _AVX512_STORE(&q[(nb+3)*ldq],q1);
        _AVX512_STORE(&q[((nb+3)*ldq)+offset],q2);
        _AVX512_STORE(&q[((nb+3)*ldq)+2*offset],q3);
        _AVX512_STORE(&q[((nb+3)*ldq)+3*offset],q4);

        h1 = _AVX512_SET1(hh[nb-1]);
        q1 = _AVX512_LOAD(&q[(nb+4)*ldq]);
        q2 = _AVX512_LOAD(&q[((nb+4)*ldq)+offset]);
        q3 = _AVX512_LOAD(&q[((nb+4)*ldq)+2*offset]);
        q4 = _AVX512_LOAD(&q[((nb+4)*ldq)+3*offset]);

        q1 = _AVX512_NFMA(x1, h1, q1);
        q2 = _AVX512_NFMA(x2, h1, q2);
        q3 = _AVX512_NFMA(x3, h1, q3);
        q4 = _AVX512_NFMA(x4, h1, q4);

        _AVX512_STORE(&q[(nb+4)*ldq],q1);
        _AVX512_STORE(&q[((nb+4)*ldq)+offset],q2);
        _AVX512_STORE(&q[((nb+4)*ldq)+2*offset],q3);
        _AVX512_STORE(&q[((nb+4)*ldq)+3*offset],q4);

}

