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
//    along with ELPA. If not, see <http://www.gnu.org/licenses/>
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
#include "config-f90.h"

#define CONCAT_8ARGS(a, b, c, d, e, f, g, h) CONCAT2_8ARGS(a, b, c, d, e, f, g, h)
#define CONCAT2_8ARGS(a, b, c, d, e, f, g, h) a ## b ## c ## d ## e ## f ## g ## h

#define CONCAT_7ARGS(a, b, c, d, e, f, g) CONCAT2_7ARGS(a, b, c, d, e, f, g)
#define CONCAT2_7ARGS(a, b, c, d, e, f, g) a ## b ## c ## d ## e ## f ## g

#define CONCAT_6ARGS(a, b, c, d, e, f) CONCAT2_6ARGS(a, b, c, d, e, f)
#define CONCAT2_6ARGS(a, b, c, d, e, f) a ## b ## c ## d ## e ## f

#define CONCAT_5ARGS(a, b, c, d, e) CONCAT2_5ARGS(a, b, c, d, e)
#define CONCAT2_5ARGS(a, b, c, d, e) a ## b ## c ## d ## e

#define CONCAT_4ARGS(a, b, c, d) CONCAT2_4ARGS(a, b, c, d)
#define CONCAT2_4ARGS(a, b, c, d) a ## b ## c ## d

#define CONCAT_3ARGS(a, b, c) CONCAT2_3ARGS(a, b, c)
#define CONCAT2_3ARGS(a, b, c) a ## b ## c

//define instruction set numbers
#define SSE_128 128
#define AVX_256 256
#define AVX2_256 2562
#define AVX_512 512
#define NEON_ARCH64_128 1285

#if VEC_SET == SSE_128 || VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == AVX_512
#include <x86intrin.h>
#ifdef BLOCK2
#if VEC_SET == SSE_128 
#include <pmmintrin.h>
#endif
#endif

#define __forceinline __attribute__((always_inline))

#endif /* VEC_SET == SSE_128 || VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == AVX_512 */

#if VEC_SET == NEON_ARCH64_128
#include <arm_neon.h>
#endif

#include <complex.h>

#include <stdio.h>
#include <stdlib.h>

#ifdef BLOCK2
#define PREFIX double
#define BLOCK 2
#endif

#ifdef BLOCK1
#define PREFIX single
#define BLOCK 1
#endif

#if VEC_SET == SSE_128
#define SIMD_SET SSE
#endif

#if VEC_SET == NEON_ARCH64_128
#define SIMD_SET NEON_ARCH64
#endif

#if VEC_SET == AVX_256
#define SIMD_SET AVX
#endif

#if VEC_SET == AVX2_256
#define SIMD_SET AVX2
#endif

#if VEC_SET == AVX_512
#define SIMD_SET AVX512
#endif


#if VEC_SET == SSE_128

#ifdef DOUBLE_PRECISION_COMPLEX
#define offset 2
#define __SIMD_DATATYPE __m128d
#define _SIMD_LOAD _mm_load_pd
#define _SIMD_LOADU _mm_loadu_pd
#define _SIMD_STORE _mm_store_pd
#define _SIMD_STOREU _mm_storeu_pd
#define _SIMD_MUL _mm_mul_pd
#define _SIMD_ADD _mm_add_pd
#define _SIMD_XOR _mm_xor_pd
#define _SIMD_ADDSUB _mm_addsub_pd
#define _SIMD_SHUFFLE _mm_shuffle_pd
#define _SHUFFLE _MM_SHUFFLE2(0,1)

#ifdef __ELPA_USE_FMA__
#define _SIMD_FMSUBADD _mm_maddsub_pd
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */

#ifdef SINGLE_PRECISION_COMPLEX
#define offset 4
#define __SIMD_DATATYPE __m128
#define _SIMD_LOAD _mm_load_ps
#define _SIMD_LOADU _mm_loadu_ps
#define _SIMD_STORE _mm_store_ps
#define _SIMD_STOREU _mm_storeu_ps
#define _SIMD_MUL _mm_mul_ps
#define _SIMD_ADD _mm_add_ps
#define _SIMD_XOR _mm_xor_ps
#define _SIMD_ADDSUB _mm_addsub_ps
#define _SIMD_SHUFFLE _mm_shuffle_ps
#define _SHUFFLE 0xb1

#ifdef __ELPA_USE_FMA__
#define _SIMD_FMSUBADD _mm_maddsub_ps
#endif

#endif /* SINGLE_PRECISION_COMPLEX */

#endif /* VEC_SET == SSE_128 */

#if VEC_SET == NEON_128

#ifdef DOUBLE_PRECISION_COMPLEX
#define offset 2
#define __SIMD_DATATYPE __Float64x2_t
#define _SIMD_LOAD vld1q_f64
#define _SIMD_LOADU _mm_loadu_pd
#define _SIMD_STORE vst1q_f64
#define _SIMD_STOREU _mm_storeu_pd
#define _SIMD_MUL vmulq_f64
#define _SIMD_ADD vaddq_f64
#define _SIMD_XOR _mm_xor_pd
#define _SIMD_ADDSUB _mm_addsub_pd
#define _SIMD_SHUFFLE _mm_shuffle_pd
#define _SHUFFLE _MM_SHUFFLE2(0,1)

#ifdef __ELPA_USE_FMA__
#define _SIMD_FMSUBADD _mm_maddsub_pd
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */

#ifdef SINGLE_PRECISION_COMPLEX
#define offset 4
#define __SIMD_DATATYPE __m128
#define _SIMD_LOAD _mm_load_ps
#define _SIMD_LOADU _mm_loadu_ps
#define _SIMD_STORE _mm_store_ps
#define _SIMD_STOREU _mm_storeu_ps
#define _SIMD_MUL _mm_mul_ps
#define _SIMD_ADD _mm_add_ps
#define _SIMD_XOR _mm_xor_ps
#define _SIMD_ADDSUB _mm_addsub_ps
#define _SIMD_SHUFFLE _mm_shuffle_ps
#define _SHUFFLE 0xb1

#ifdef __ELPA_USE_FMA__
#define _SIMD_FMSUBADD _mm_maddsub_ps
#endif

#endif /* SINGLE_PRECISION_COMPLEX */

#endif /* VEC_SET == NEON_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256

#ifdef DOUBLE_PRECISION_COMPLEX
#define offset 4
#define __SIMD_DATATYPE __m256d
#define _SIMD_LOAD _mm256_load_pd
#define _SIMD_LOADU 1
#define _SIMD_STORE _mm256_store_pd
#define _SIMD_STOREU 1
#define _SIMD_MUL _mm256_mul_pd
#define _SIMD_ADD _mm256_add_pd
#define _SIMD_XOR _mm256_xor_pd
#define _SIMD_BROADCAST _mm256_broadcast_sd
#define _SIMD_SET1 _mm256_set1_pd
#define _SIMD_ADDSUB _mm256_addsub_pd
#define _SIMD_SHUFFLE _mm256_shuffle_pd
#define _SHUFFLE 0x5

#ifdef HAVE_AVX2
#if VEC_SET == AVX2_256
#ifdef __FMA4__
#define __ELPA_USE_FMA__
#define _mm256_FMADDSUB_pd(a,b,c) _mm256_maddsub_pd(a,b,c)
#define _mm256_FMSUBADD_pd(a,b,c) _mm256_msubadd_pd(a,b,c)
#endif

#ifdef __AVX2__
#define __ELPA_USE_FMA__
#define _mm256_FMADDSUB_pd(a,b,c) _mm256_fmaddsub_pd(a,b,c)
#define _mm256_FMSUBADD_pd(a,b,c) _mm256_fmsubadd_pd(a,b,c)
#endif

#define _SIMD_FMADDSUB _mm256_FMADDSUB_pd
#define _SIMD_FMSUBADD _mm256_FMSUBADD_pd
#endif /* VEC_SET == AVX2_256 */
#endif /* HAVE_AVX2 */

#endif /* DOUBLE_PRECISION_COMPLEX */

#ifdef SINGLE_PRECISION_COMPLEX
#define offset 8
#define __SIMD_DATATYPE __m256
#define _SIMD_LOAD _mm256_load_ps
#define _SIMD_LOADU 1
#define _SIMD_STORE _mm256_store_ps
#define _SIMD_STOREU 1
#define _SIMD_MUL _mm256_mul_ps
#define _SIMD_ADD _mm256_add_ps
#define _SIMD_XOR _mm256_xor_ps
#define _SIMD_BROADCAST  _mm256_broadcast_ss
#define _SIMD_SET1 _mm256_set1_ps
#define _SIMD_ADDSUB _mm256_addsub_ps
#define _SIMD_SHUFFLE _mm256_shuffle_ps
#define _SHUFFLE 0xb1

#ifdef HAVE_AVX2
#if VEC_SET == AVX2_256

#ifdef __FMA4__
#define __ELPA_USE_FMA__
#define _mm256_FMADDSUB_ps(a,b,c) _mm256_maddsub_ps(a,b,c)
#define _mm256_FMSUBADD_ps(a,b,c) _mm256_msubadd_ps(a,b,c)
#endif

#ifdef __AVX2__
#define __ELPA_USE_FMA__
#define _mm256_FMADDSUB_ps(a,b,c) _mm256_fmaddsub_ps(a,b,c)
#define _mm256_FMSUBADD_ps(a,b,c) _mm256_fmsubadd_ps(a,b,c)
#endif

#define _SIMD_FMADDSUB _mm256_FMADDSUB_ps
#define _SIMD_FMSUBADD _mm256_FMSUBADD_ps
#endif /* VEC_SET == AVX2_256 */
#endif /* HAVE_AVX2 */

#endif /* SINGLE_PRECISION_COMPLEX */

#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512

#ifdef DOUBLE_PRECISION_COMPLEX
#define offset 8
#define __SIMD_DATATYPE __m512d
#define _SIMD_LOAD _mm512_load_pd
#define _SIMD_LOADU 1
#define _SIMD_STORE _mm512_store_pd
#define _SIMD_STOREU 1
#define _SIMD_MUL _mm512_mul_pd
#define _SIMD_ADD _mm512_add_pd
#ifdef HAVE_AVX512_XEON
#define _SIMD_XOR _mm512_xor_pd
#endif
#define _SIMD_BROADCAST 1
#define _SIMD_SET1 _mm512_set1_pd
#define _SIMD_SET _mm512_set_pd
#define _SIMD_XOR_EPI _mm512_xor_epi64
#define _SIMD_ADDSUB 1
#define _SIMD_SHUFFLE _mm512_shuffle_pd
#define _SIMD_MASK_STOREU _mm512_mask_storeu_pd
#define _SHUFFLE 0x55

#ifdef HAVE_AVX512
#define __ELPA_USE_FMA__
#define _mm512_FMADDSUB_pd(a,b,c) _mm512_fmaddsub_pd(a,b,c)
#define _mm512_FMSUBADD_pd(a,b,c) _mm512_fmsubadd_pd(a,b,c)

#define _SIMD_FMADDSUB _mm512_FMADDSUB_pd
#define _SIMD_FMSUBADD _mm512_FMSUBADD_pd
#endif /* HAVE_AVX512 */

#endif /* DOUBLE_PRECISION_COMPLEX */

#ifdef SINGLE_PRECISION_COMPLEX
#define offset 16
#define __SIMD_DATATYPE __m512
#define _SIMD_LOAD _mm512_load_ps
#define _SIMD_LOADU 1
#define _SIMD_STORE _mm512_store_ps
#define _SIMD_STOREU 1
#define _SIMD_MUL _mm512_mul_ps
#define _SIMD_ADD _mm512_add_ps
#ifdef HAVE_AVX512_XEON
#define _SIMD_XOR _mm512_xor_ps
#endif
#define _SIMD_BROADCAST 1
#define _SIMD_SET1 _mm512_set1_ps
#define _SIMD_SET _mm512_set_ps
#define _SIMD_ADDSUB 1
#define _SIMD_SHUFFLE _mm512_shuffle_ps
#define _SIMD_MASK_STOREU _mm512_mask_storeu_ps
#define _SIMD_XOR_EPI _mm512_xor_epi32
#define _SHUFFLE 0xb1

#ifdef HAVE_AVX512

#define __ELPA_USE_FMA__
#define _mm512_FMADDSUB_ps(a,b,c) _mm512_fmaddsub_ps(a,b,c)
#define _mm512_FMSUBADD_ps(a,b,c) _mm512_fmsubadd_ps(a,b,c)

#define _SIMD_FMADDSUB _mm512_FMADDSUB_ps
#define _SIMD_FMSUBADD _mm512_FMSUBADD_ps
#endif /* HAVE_AVX512 */

#endif /* SINGLE_PRECISION_COMPLEX */

#endif /* VEC_SET == AVX_512 */




#define __forceinline __attribute__((always_inline))

#ifdef HAVE_SSE_INTRINSICS
#undef __AVX__
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
#define WORD_LENGTH double
#define DATA_TYPE double complex
#define DATA_TYPE_PTR double complex*
#define DATA_TYPE_REAL double
#define DATA_TYPE_REAL_PTR double*
#endif

#ifdef SINGLE_PRECISION_COMPLEX
#define WORD_LENGTH single
#define DATA_TYPE float complex
#define DATA_TYPE_PTR float complex*
#define DATA_TYPE_REAL float
#define DATA_TYPE_REAL_PTR float*
#endif


//Forward declaration

#if VEC_SET  == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 6
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 12
#endif
#endif /* VEC_SET  == SSE_128 */

#if VEC_SET  == AVX_256 || VEC_SET  == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 12
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 24
#endif
#endif /* VEC_SET  == AVX_256 || VEC_SET  == AVX2_256 */

#if VEC_SET  == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 24
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 48
#endif
#endif /* VEC_SET  == AVX_512 */
static __forceinline void CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH)(DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq 
#ifdef BLOCK1
		                       );
#endif
#ifdef BLOCK2
                                       ,int ldh, DATA_TYPE s);
#endif

#if VEC_SET  == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 5
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 10
#endif
#endif /* VEC_SET  == SSE_128 */

#if VEC_SET  == AVX_256 || VEC_SET  == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 10
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 20
#endif
#endif /* VEC_SET  == AVX_256 || VEC_SET  == AVX2_256 */

#if VEC_SET  == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 20
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 40
#endif
#endif /* VEC_SET  == AVX_512 */

static __forceinline void CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH)(DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq
#ifdef BLOCK1
		                       );
#endif
#ifdef BLOCK2
                                       ,int ldh, DATA_TYPE s);
#endif


#if VEC_SET  == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 8
#endif
#endif /* VEC_SET  == SSE_128 */

#if VEC_SET  == AVX_256 || VEC_SET  == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 16
#endif
#endif /* VEC_SET  == AVX_256 || VEC_SET  == AVX2_256 */

#if VEC_SET  == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 16
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 32
#endif
#endif /* VEC_SET  == AVX_512 */

static __forceinline void CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH)(DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq
#ifdef BLOCK1
		                       );
#endif
#ifdef BLOCK2
                                       ,int ldh, DATA_TYPE s);
#endif

#if VEC_SET  == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 3
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 6
#endif
#endif /* VEC_SET  == SSE_128 */

#if VEC_SET  == AVX_256 || VEC_SET  == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 6
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 12
#endif
#endif /* VEC_SET  == AVX_256 || VEC_SET  == AVX2_256 */

#if VEC_SET  == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 12
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 24
#endif
#endif /* VEC_SET  == AVX_512 */

static __forceinline void CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH)(DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq
#ifdef BLOCK1
		                       );
#endif
#ifdef BLOCK2
                                       ,int ldh, DATA_TYPE s);
#endif

#if VEC_SET  == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 2
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 4
#endif
#endif /* VEC_SET  == SSE_128 */

#if VEC_SET  == AVX_256 || VEC_SET  == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 8
#endif
#endif /* VEC_SET  == AVX_256 || VEC_SET  == AVX2_256 */

#if VEC_SET  == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 16
#endif
#endif /* VEC_SET  == AVX_512 */

static __forceinline void CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH)(DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq
#ifdef BLOCK1
		                       );
#endif
#ifdef BLOCK2
                                       ,int ldh, DATA_TYPE s);
#endif

#if VEC_SET  == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 1
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 2
#endif
#endif /* VEC_SET  == SSE_128 */

#if VEC_SET  == AVX_256 || VEC_SET  == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 2
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 4
#endif
#endif /* VEC_SET  == AVX_256 || VEC_SET  == AVX2_256 */

#if VEC_SET  == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#undef ROW_LENGTH 
#define ROW_LENGTH 8
#endif
#endif /* VEC_SET  == AVX_512 */

static __forceinline void CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH)(DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq
#ifdef BLOCK1
		                       );
#endif
#ifdef BLOCK2
                                       ,int ldh, DATA_TYPE s);
#endif


/*
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine single_hh_trafo_complex_SSE_1hv_double(q, hh, pnb, pnq, pldq) &
!f>                             bind(C, name="single_hh_trafo_complex_SSE_1hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq
!f>     ! complex(kind=c_double_complex)     :: q(*)
!f>     type(c_ptr), value                   :: q
!f>     complex(kind=c_double_complex)     :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine single_hh_trafo_complex_SSE_1hv_single(q, hh, pnb, pnq, pldq) &
!f>                             bind(C, name="single_hh_trafo_complex_SSE_1hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq
!f>     ! complex(kind=c_float_complex)   :: q(*)
!f>     type(c_ptr), value                :: q
!f>     complex(kind=c_float_complex)   :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/


/*
!f>#if defined(HAVE_AVX)
!f> interface
!f>   subroutine single_hh_trafo_complex_AVX_1hv_double(q, hh, pnb, pnq, pldq) &
!f>                             bind(C, name="single_hh_trafo_complex_AVX_1hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq
!f>     ! complex(kind=c_double_complex)     :: q(*)
!f>     type(c_ptr), value                   :: q
!f>     complex(kind=c_double_complex)       :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_AVX)
!f> interface
!f>   subroutine single_hh_trafo_complex_AVX_1hv_single(q, hh, pnb, pnq, pldq) &
!f>                             bind(C, name="single_hh_trafo_complex_AVX_1hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq
!f>     ! complex(kind=c_float_complex)   :: q(*)
!f>     type(c_ptr), value              :: q
!f>     complex(kind=c_float_complex)   :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/


/*
!f>#if defined(HAVE_AVX2)
!f> interface
!f>   subroutine single_hh_trafo_complex_AVX2_1hv_double(q, hh, pnb, pnq, pldq) &
!f>                             bind(C, name="single_hh_trafo_complex_AVX2_1hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq
!f>     ! complex(kind=c_double_complex)     :: q(*)
!f>     type(c_ptr), value                   :: q
!f>     complex(kind=c_double_complex)       :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_AVX2)
!f> interface
!f>   subroutine single_hh_trafo_complex_AVX2_1hv_single(q, hh, pnb, pnq, pldq) &
!f>                             bind(C, name="single_hh_trafo_complex_AVX2_1hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq
!f>     ! complex(kind=c_float_complex)   :: q(*)
!f>     type(c_ptr), value              :: q
!f>     complex(kind=c_float_complex)   :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine single_hh_trafo_complex_AVX512_1hv_double(q, hh, pnb, pnq, pldq) &
!f>                             bind(C, name="single_hh_trafo_complex_AVX512_1hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq
!f>     ! complex(kind=c_double_complex)     :: q(*)
!f>     type(c_ptr), value                 :: q
!f>     complex(kind=c_double_complex)     :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine single_hh_trafo_complex_AVX512_1hv_single(q, hh, pnb, pnq, pldq) &
!f>                             bind(C, name="single_hh_trafo_complex_AVX512_1hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq
!f>     ! complex(kind=c_float_complex)     :: q(*)
!f>     type(c_ptr), value                  :: q
!f>     complex(kind=c_float_complex)     :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/


/*
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine double_hh_trafo_complex_SSE_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="double_hh_trafo_complex_SSE_2hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     ! complex(kind=c_double_complex)     :: q(*)
!f>     type(c_ptr), value                   :: q
!f>     complex(kind=c_double_complex)     :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine double_hh_trafo_complex_SSE_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="double_hh_trafo_complex_SSE_2hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     ! complex(kind=c_float_complex)   :: q(*)
!f>     type(c_ptr), value                :: q
!f>     complex(kind=c_float_complex)   :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_AVX)
!f> interface
!f>   subroutine double_hh_trafo_complex_AVX_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="double_hh_trafo_complex_AVX_2hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        ! complex(kind=c_double_complex)     :: q(*)
!f>        type(c_ptr), value                     :: q
!f>        complex(kind=c_double_complex)           :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_AVX)
!f> interface
!f>   subroutine double_hh_trafo_complex_AVX_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="double_hh_trafo_complex_AVX_2hv_single")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        ! complex(kind=c_float_complex)   :: q(*)
!f>        type(c_ptr), value                  :: q
!f>        complex(kind=c_float_complex)        :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_AVX2)
!f> interface
!f>   subroutine double_hh_trafo_complex_AVX2_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="double_hh_trafo_complex_AVX2_2hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        ! complex(kind=c_double_complex)     :: q(*)
!f>        type(c_ptr), value                     :: q
!f>        complex(kind=c_double_complex)           :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_AVX2)
!f> interface
!f>   subroutine double_hh_trafo_complex_AVX2_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="double_hh_trafo_complex_AVX2_2hv_single")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        ! complex(kind=c_float_complex)   :: q(*)
!f>        type(c_ptr), value                  :: q
!f>        complex(kind=c_float_complex)        :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine double_hh_trafo_complex_AVX512_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="double_hh_trafo_complex_AVX512_2hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     ! complex(kind=c_double_complex)     :: q(*)
!f>     type(c_ptr), value                   :: q
!f>     complex(kind=c_double_complex)     :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine double_hh_trafo_complex_AVX512_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="double_hh_trafo_complex_AVX512_2hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     ! complex(kind=c_float_complex)     :: q(*)
!f>     type(c_ptr), value                  :: q
!f>     complex(kind=c_float_complex)     :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/


void CONCAT_7ARGS(PREFIX,_hh_trafo_complex_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int* pnb, int* pnq, int* pldq
#ifdef BLOCK1
		  )
#endif
#ifdef BLOCK2
                  ,int* pldh)
#endif
{

     int i, worked_on;
     int nb = *pnb;
     int nq = *pldq;
     int ldq = *pldq;
#ifdef BLOCK2
     int ldh = *pldh;

     DATA_TYPE s = conj(hh[(ldh)+1])*1.0;

     for (i = 2; i < nb; i++)
     {
             s += hh[i-1] * conj(hh[(i+ldh)]);
     }
#endif

     worked_on = 0;

#ifdef BLOCK1

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 6
#define STEP_SIZE 6
#define UPPER_BOUND 5
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 12
#define STEP_SIZE 12
#define UPPER_BOUND 10
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 12
#define STEP_SIZE 12
#define UPPER_BOUND 10
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 24
#define STEP_SIZE 24
#define UPPER_BOUND 20
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 24
#define STEP_SIZE 24
#define UPPER_BOUND 20
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 48
#define STEP_SIZE 48
#define UPPER_BOUND 40
#endif
#endif /* VEC_SET == AVX_512 */


        for (i = 0; i < nq - UPPER_BOUND; i+= STEP_SIZE)
        {

            CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (&q[i], hh, nb, ldq);
	    worked_on += ROW_LENGTH;
        }

        if (nq == i) {
          return;
        }

#if VEC_SET == SSE_128
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 5
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 10
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 10
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 20
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 20
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 40
#endif
#endif /* VEC_SET == AVX_512 */

        if (nq-i == ROW_LENGTH)
        {
            CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (&q[i], hh, nb, ldq);
	    worked_on += ROW_LENGTH;
        }

#if VEC_SET == SSE_128
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 8
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 16
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 16
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 32
#endif
#endif /* VEC_SET == AVX_512 */

        if (nq-i == ROW_LENGTH)
        {
            CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (&q[i], hh, nb, ldq);
	    worked_on += ROW_LENGTH;
        }

#if VEC_SET == SSE_128
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 3
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 6
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 6
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 12
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 12
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 24
#endif
#endif /* VEC_SET == AVX_512 */

        if (nq-i == ROW_LENGTH)
        {
            CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (&q[i], hh, nb, ldq);
	    worked_on += ROW_LENGTH;
        }

#if VEC_SET == SSE_128
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 2
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 4
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 8
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 16
#endif
#endif /* VEC_SET == AVX_512 */

        if (nq-i == ROW_LENGTH)
        {
            CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (&q[i], hh, nb, ldq);
	    worked_on += ROW_LENGTH;
        }

#if VEC_SET == SSE_128
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 1
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 2
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 2
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 4
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 8
#endif
#endif /* VEC_SET == AVX_512 */

        if (nq-i == ROW_LENGTH)
        {
            CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (&q[i], hh, nb, ldq);
	    worked_on += ROW_LENGTH;
        }

#endif /* BLOCK1 */

#ifdef BLOCK2

#if VEC_SET == SSE_128
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 4
#define STEP_SIZE 4
#define UPPER_BOUND 3
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 8
#define STEP_SIZE 8
#define UPPER_BOUND 6
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 8
#define STEP_SIZE 8
#define UPPER_BOUND 6
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 16
#define STEP_SIZE 16
#define UPPER_BOUND 12
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 16
#define STEP_SIZE 16
#define UPPER_BOUND 12
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 32
#define STEP_SIZE 32
#define UPPER_BOUND 24
#endif
#endif /* VEC_SET == AVX_512 */

    for (i = 0; i < nq - UPPER_BOUND; i+=STEP_SIZE)
    {
         CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
	 worked_on +=ROW_LENGTH;
    }
 
    if (nq == i)
    {
      return;
    }
    
#if VEC_SET == SSE_128
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 3
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 6
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 6
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 12
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 12
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 24
#endif
#endif /* VEC_SET == AVX_512 */

    if (nq-i == ROW_LENGTH)
    {
        CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
        worked_on += ROW_LENGTH;
    }

#if VEC_SET == SSE_128
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 2
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 4
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 8
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 16
#endif
#endif /* VEC_SET == AVX_512 */

    if (nq-i == ROW_LENGTH)
    {
        CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
        worked_on += ROW_LENGTH;
    }

#if VEC_SET == SSE_128
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 1
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 2
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 2
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 4
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#undef ROW_LENGTH
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 8
#endif
#endif /* VEC_SET == AVX_512 */

    if (nq-i == ROW_LENGTH)
    {
        CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
        worked_on += ROW_LENGTH;
    }

#endif /* BLOCK2 */

#ifdef WITH_DEBUG
    if (worked_on != nq)
    {
      printf("Error in complex SIMD_SET BLOCK BLOCK kernel %d %d\n", worked_on, nq);
      abort();
    }
#endif

}

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 6
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 12
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 12
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 24
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 24
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 48
#endif
#endif /* VEC_SET == AVX_512 */

static __forceinline void CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq
#ifdef BLOCK1
		)
#endif
#ifdef BLOCK2
                ,int ldh, DATA_TYPE s)
#endif
{

    DATA_TYPE_REAL_PTR q_dbl = (DATA_TYPE_REAL_PTR)q;
    DATA_TYPE_REAL_PTR hh_dbl = (DATA_TYPE_REAL_PTR)hh;
#ifdef BLOCK2
    DATA_TYPE_REAL_PTR s_dbl = (DATA_TYPE_REAL_PTR)(&s);
#endif

    __SIMD_DATATYPE x1, x2, x3, x4, x5, x6;
    __SIMD_DATATYPE q1, q2, q3, q4, q5, q6;
#ifdef BLOCK2
    __SIMD_DATATYPE y1, y2, y3, y4, y5, y6;
    __SIMD_DATATYPE h2_real, h2_imag;
#endif
    __SIMD_DATATYPE h1_real, h1_imag;
    __SIMD_DATATYPE tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
    int i=0;

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
    __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set_epi64x(0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
    __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set_epi64x(0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_512 */

#ifdef BLOCK2
     x1 = _SIMD_LOAD(&q_dbl[(2*ldq)+0]);
     x2 = _SIMD_LOAD(&q_dbl[(2*ldq)+offset]);
     x3 = _SIMD_LOAD(&q_dbl[(2*ldq)+2*offset]);
     x4 = _SIMD_LOAD(&q_dbl[(2*ldq)+3*offset]);
     x5 = _SIMD_LOAD(&q_dbl[(2*ldq)+4*offset]);
     x6 = _SIMD_LOAD(&q_dbl[(2*ldq)+5*offset]);

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+1)*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+1)*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h2_real = _SIMD_SET1(hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_SET1(hh_dbl[((ldh+1)*2)+1]);
#endif /*  VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
     // conjugate
     h2_imag = _SIMD_XOR(h2_imag, sign);
#endif

     y1 = _SIMD_LOAD(&q_dbl[0]);
     y2 = _SIMD_LOAD(&q_dbl[offset]);
     y3 = _SIMD_LOAD(&q_dbl[2*offset]);
     y4 = _SIMD_LOAD(&q_dbl[3*offset]);
     y5 = _SIMD_LOAD(&q_dbl[4*offset]);
     y6 = _SIMD_LOAD(&q_dbl[5*offset]);

     tmp1 = _SIMD_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_ADD(y1, _SIMD_FMSUBADD(h2_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
     tmp2 = _SIMD_MUL(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
     y2 = _SIMD_ADD(y2, _SIMD_FMSUBADD(h2_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     y2 = _SIMD_ADD(y2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h2_imag, x3);
#ifdef __ELPA_USE_FMA__
     y3 = _SIMD_ADD(y3, _SIMD_FMSUBADD(h2_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     y3 = _SIMD_ADD(y3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
     tmp4 = _SIMD_MUL(h2_imag, x4);
#ifdef __ELPA_USE_FMA__
     y4 = _SIMD_ADD(y4, _SIMD_FMSUBADD(h2_real, x4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
     y4 = _SIMD_ADD(y4, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

     tmp5 = _SIMD_MUL(h2_imag, x5);
#ifdef __ELPA_USE_FMA__
     y5 = _SIMD_ADD(y5, _SIMD_FMSUBADD(h2_real, x5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
     y5 = _SIMD_ADD(y5, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif
     tmp6 = _SIMD_MUL(h2_imag, x6);
#ifdef __ELPA_USE_FMA__
     y6 = _SIMD_ADD(y6, _SIMD_FMSUBADD(h2_real, x6, _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#else
     y6 = _SIMD_ADD(y6, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x6), _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

#ifdef BLOCK1
    x1 = _SIMD_LOAD(&q_dbl[0]);
    x2 = _SIMD_LOAD(&q_dbl[offset]);
    x3 = _SIMD_LOAD(&q_dbl[2*offset]);
    x4 = _SIMD_LOAD(&q_dbl[3*offset]);
    x5 = _SIMD_LOAD(&q_dbl[4*offset]);
    x6 = _SIMD_LOAD(&q_dbl[5*offset]);
#endif

    for (i = BLOCK; i < nb; i++)
    {

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = _mm_loaddup_pd(&hh_dbl[(i-BLOCK+1)*2]);
        h1_imag = _mm_loaddup_pd(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i-BLOCK+1)*2]) )));
        h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((i-BLOCK+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
       h1_real = _SIMD_BROADCAST(&hh_dbl[(i-BLOCK+1)*2]);
       h1_imag = _SIMD_BROADCAST(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
       h1_real = _SIMD_SET1(hh_dbl[(i-BLOCK+1)*2]);
       h1_imag = _SIMD_SET1(hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
        // conjugate
        h1_imag = _SIMD_XOR(h1_imag, sign);
#endif

        q1 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+0]);
        q2 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+offset]);
        q3 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+2*offset]);
        q4 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+3*offset]);
        q5 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+4*offset]);
        q6 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+5*offset]);

        tmp1 = _SIMD_MUL(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
        x1 = _SIMD_ADD(x1, _SIMD_FMSUBADD(h1_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
        x1 = _SIMD_ADD(x1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
        tmp2 = _SIMD_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
        x2 = _SIMD_ADD(x2, _SIMD_FMSUBADD(h1_real, q2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
        x2 = _SIMD_ADD(x2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif
        tmp3 = _SIMD_MUL(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
        x3 = _SIMD_ADD(x3, _SIMD_FMSUBADD(h1_real, q3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
        x3 = _SIMD_ADD(x3, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

        tmp4 = _SIMD_MUL(h1_imag, q4);
#ifdef __ELPA_USE_FMA__
        x4 = _SIMD_ADD(x4, _SIMD_FMSUBADD(h1_real, q4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
        x4 = _SIMD_ADD(x4, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif
        tmp5 = _SIMD_MUL(h1_imag, q5);
#ifdef __ELPA_USE_FMA__
        x5 = _SIMD_ADD(x5, _SIMD_FMSUBADD(h1_real, q5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
        x5 = _SIMD_ADD(x5, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif
        tmp6 = _SIMD_MUL(h1_imag, q6);
#ifdef __ELPA_USE_FMA__
        x6 = _SIMD_ADD(x6, _SIMD_FMSUBADD(h1_real, q6, _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#else
        x6 = _SIMD_ADD(x6, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q6), _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#endif

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+i)*2]);
          h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+i)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+i)*2]) )));
          h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+i)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
          h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
          h2_real = _SIMD_SET1(hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_SET1(hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_512 */


#ifndef __ELPA_USE_FMA__
          // conjugate
          h2_imag = _SIMD_XOR(h2_imag, sign);
#endif

          tmp1 = _SIMD_MUL(h2_imag, q1);
#ifdef __ELPA_USE_FMA__
          y1 = _SIMD_ADD(y1, _SIMD_FMSUBADD(h2_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
          tmp2 = _SIMD_MUL(h2_imag, q2);
#ifdef __ELPA_USE_FMA__
          y2 = _SIMD_ADD(y2, _SIMD_FMSUBADD(h2_real, q2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
          y2 = _SIMD_ADD(y2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

          tmp3 = _SIMD_MUL(h2_imag, q3);
#ifdef __ELPA_USE_FMA__
          y3 = _SIMD_ADD(y3, _SIMD_FMSUBADD(h2_real, q3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
          y3 = _SIMD_ADD(y3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
          tmp4 = _SIMD_MUL(h2_imag, q4);
#ifdef __ELPA_USE_FMA__
          y4 = _SIMD_ADD(y4, _SIMD_FMSUBADD(h2_real, q4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
          y4 = _SIMD_ADD(y4, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

          tmp5 = _SIMD_MUL(h2_imag, q5);
#ifdef __ELPA_USE_FMA__
          y5 = _SIMD_ADD(y5, _SIMD_FMSUBADD(h2_real, q5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
          y5 = _SIMD_ADD(y5, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif
          tmp6 = _SIMD_MUL(h2_imag, q6);
#ifdef __ELPA_USE_FMA__
          y6 = _SIMD_ADD(y6, _SIMD_FMSUBADD(h2_real, q6, _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#else
          y6 = _SIMD_ADD(y6, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q6), _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#endif
	
#endif /* BLOCK2 */

    }

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[(nb-1)*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[((nb-1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(nb-1)*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((nb-1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_SET1(hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
     // conjugate
     h1_imag = _SIMD_XOR(h1_imag, sign);
#endif

     q1 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+0]);
     q2 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+offset]);
     q3 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+2*offset]);
     q4 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+3*offset]);
     q5 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+4*offset]);
     q6 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+5*offset]);

     tmp1 = _SIMD_MUL(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
     x1 = _SIMD_ADD(x1, _SIMD_FMSUBADD(h1_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     x1 = _SIMD_ADD(x1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
     tmp2 = _SIMD_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
     x2 = _SIMD_ADD(x2, _SIMD_FMSUBADD(h1_real, q2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     x2 = _SIMD_ADD(x2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
     x3 = _SIMD_ADD(x3, _SIMD_FMSUBADD(h1_real, q3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     x3 = _SIMD_ADD(x3, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
     tmp4 = _SIMD_MUL(h1_imag, q4);
#ifdef __ELPA_USE_FMA__
     x4 = _SIMD_ADD(x4, _SIMD_FMSUBADD(h1_real, q4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
     x4 = _SIMD_ADD(x4, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

     tmp5 = _SIMD_MUL(h1_imag, q5);
#ifdef __ELPA_USE_FMA__
     x5 = _SIMD_ADD(x5, _SIMD_FMSUBADD(h1_real, q5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
     x5 = _SIMD_ADD(x5, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif
     tmp6 = _SIMD_MUL(h1_imag, q6);
#ifdef __ELPA_USE_FMA__
     x6 = _SIMD_ADD(x6, _SIMD_FMSUBADD(h1_real, q6, _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#else
     x6 = _SIMD_ADD(x6, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q6), _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
    h1_real = _mm_loaddup_pd(&hh_dbl[0]);
    h1_imag = _mm_loaddup_pd(&hh_dbl[1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[0]) )));
    h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[1]) )));
#endif
#endif /*  VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1_real = _SIMD_BROADCAST(&hh_dbl[0]);
    h1_imag = _SIMD_BROADCAST(&hh_dbl[1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
    h1_real = _SIMD_SET1(hh_dbl[0]);
    h1_imag = _SIMD_SET1(hh_dbl[1]);

#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_imag, (__m512i) sign);
#endif
#endif
#ifdef HAVE_AVX512_XEON
#if defined(DOUBLE_PRECISION_COMPLEX) || defined(SINGLE_PRECISION_COMPLEX)
        h1_real = _SIMD_XOR(h1_real, sign);
        h1_imag = _SIMD_XOR(h1_imag, sign);
#endif
#endif

#endif /* VEC_SET == AVX_512 */

#if VEC_SET != AVX_512
    h1_real = _SIMD_XOR(h1_real, sign);
    h1_imag = _SIMD_XOR(h1_imag, sign);
#endif /* VEC_SET != AVX_512 */

    tmp1 = _SIMD_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
    x1 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif
    tmp2 = _SIMD_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
    x2 = _SIMD_FMADDSUB(h1_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#else
    x2 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#endif
    tmp3 = _SIMD_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
    x3 = _SIMD_FMADDSUB(h1_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#else
    x3 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#endif

    tmp4 = _SIMD_MUL(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
    x4 = _SIMD_FMADDSUB(h1_real, x4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE));
#else
    x4 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE));
#endif
    tmp5 = _SIMD_MUL(h1_imag, x5);
#ifdef __ELPA_USE_FMA__
    x5 = _SIMD_FMADDSUB(h1_real, x5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE));
#else
    x5 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE));
#endif
    tmp6 = _SIMD_MUL(h1_imag, x6);
#ifdef __ELPA_USE_FMA__
    x6 = _SIMD_FMADDSUB(h1_real, x6, _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE));
#else
    x6 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x6), _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE));
#endif

#ifdef BLOCK2

#if VEC_SET == SSE_128    
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[ldh*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[(ldh*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[ldh*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh*2)+1]) )));
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[ldh*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[(ldh*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[ldh*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh*2)+1]) )));
#endif
#endif /* VEC_SET == 128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[ldh*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[(ldh*2)+1]);
     h2_real = _SIMD_BROADCAST(&hh_dbl[ldh*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[(ldh*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[ldh*2]);
     h1_imag = _SIMD_SET1(hh_dbl[(ldh*2)+1]);
     h2_real = _SIMD_SET1(hh_dbl[ldh*2]);
     h2_imag = _SIMD_SET1(hh_dbl[(ldh*2)+1]);

#ifdef HAVE_AVX512_XEON_PHI

#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h1_real, (__m512i) sign);
     h1_imag = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
     h1_imag = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h2_real, (__m512i) sign);
     h2_imag = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h2_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h2_real, (__m512i) sign);
     h2_imag = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h2_imag, (__m512i) sign);
#endif
#endif /* HAVE_AVX512_XEON_PHI */

#ifdef HAVE_AVX512_XEON
#if defined(DOUBLE_PRECISION_COMPLEX) || defined(SINGLE_PRECISION_COMPLEX)
        h1_real = _SIMD_XOR(h1_real, sign);
        h1_imag = _SIMD_XOR(h1_imag, sign);
        h2_real = _SIMD_XOR(h2_real, sign);
        h2_imag = _SIMD_XOR(h2_imag, sign);
#endif
#endif     
#endif /* VEC_SET == AVX_512 */

#if VEC_SET != AVX_512
     h1_real = _SIMD_XOR(h1_real, sign);
     h1_imag = _SIMD_XOR(h1_imag, sign);
     h2_real = _SIMD_XOR(h2_real, sign);
     h2_imag = _SIMD_XOR(h2_imag, sign);
#endif /* VEC_SET != AVX_512 */

#if VEC_SET == SSE_128
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = _mm_castpd_ps(_mm_load_pd1((double *) s_dbl));
#else
     tmp2 = _SIMD_LOADU(s_dbl);
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
     tmp2 = _mm256_set_pd(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = _mm256_set_ps(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0],
                             s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
     tmp2 = _SIMD_SET(s_dbl[1], s_dbl[0],
                        s_dbl[1], s_dbl[0],
                        s_dbl[1], s_dbl[0],
                        s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = (__SIMD_DATATYPE) _mm512_set1_pd(*(double*)(&s_dbl[0]));
#endif
#endif /* VEC_SET == AVX_512 */

     tmp1 = _SIMD_MUL(h2_imag, tmp2);
#ifdef __ELPA_USE_FMA__
     tmp2 = _SIMD_FMADDSUB(h2_real, tmp2, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
     tmp2 = _SIMD_ADDSUB( _SIMD_MUL(h2_real, tmp2), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

#if VEC_SET == AVX_512
     _SIMD_MASK_STOREU(s_dbl, 0x01 + 0x02, tmp2);

     h2_real = _SIMD_SET1(s_dbl[0]);
     h2_imag = _SIMD_SET1(s_dbl[1]);
#endif /* VEC_SET == AVX_512 */

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_movedup_pd(tmp2);
     h2_imag = _mm_set1_pd(tmp2[1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(tmp2);
     h2_imag = _mm_movehdup_ps(tmp2);
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_SET1(tmp2[0]);
     h2_imag = _SIMD_SET1(tmp2[1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

     tmp1 = _SIMD_MUL(h1_imag, y1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_FMADDSUB(h1_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
     y1 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif
     tmp2 = _SIMD_MUL(h1_imag, y2);
#ifdef __ELPA_USE_FMA__
     y2 = _SIMD_FMADDSUB(h1_real, y2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#else
     y2 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#endif

     tmp3 = _SIMD_MUL(h1_imag, y3);
#ifdef __ELPA_USE_FMA__
     y3 = _SIMD_FMADDSUB(h1_real, y3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#else
     y3 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#endif
     tmp4 = _SIMD_MUL(h1_imag, y4);
#ifdef __ELPA_USE_FMA__
     y4 = _SIMD_FMADDSUB(h1_real, y4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE));
#else
     y4 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE));
#endif

     tmp5 = _SIMD_MUL(h1_imag, y5);
#ifdef __ELPA_USE_FMA__
     y5 = _SIMD_FMADDSUB(h1_real, y5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE));
#else
     y5 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE));
#endif
     tmp6 = _SIMD_MUL(h1_imag, y6);
#ifdef __ELPA_USE_FMA__
     y6 = _SIMD_FMADDSUB(h1_real, y6, _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE));
#else
     y6 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y6), _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE));
#endif

     tmp1 = _SIMD_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_ADD(y1, _SIMD_FMADDSUB(h2_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
     tmp2 = _SIMD_MUL(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
     y2 = _SIMD_ADD(y2, _SIMD_FMADDSUB(h2_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     y2 = _SIMD_ADD(y2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h2_imag, x3);
#ifdef __ELPA_USE_FMA__
     y3 = _SIMD_ADD(y3, _SIMD_FMADDSUB(h2_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     y3 = _SIMD_ADD(y3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
     tmp4 = _SIMD_MUL(h2_imag, x4);
#ifdef __ELPA_USE_FMA__
     y4 = _SIMD_ADD(y4, _SIMD_FMADDSUB(h2_real, x4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
     y4 = _SIMD_ADD(y4, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

     tmp5 = _SIMD_MUL(h2_imag, x5);
#ifdef __ELPA_USE_FMA__
     y5 = _SIMD_ADD(y5, _SIMD_FMADDSUB(h2_real, x5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
     y5 = _SIMD_ADD(y5, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif
     tmp6 = _SIMD_MUL(h2_imag, x6);
#ifdef __ELPA_USE_FMA__
     y6 = _SIMD_ADD(y6, _SIMD_FMADDSUB(h2_real, x6, _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#else
     y6 = _SIMD_ADD(y6, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x6), _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

    q1 = _SIMD_LOAD(&q_dbl[0]);
    q2 = _SIMD_LOAD(&q_dbl[offset]);
    q3 = _SIMD_LOAD(&q_dbl[2*offset]);
    q4 = _SIMD_LOAD(&q_dbl[3*offset]);
    q5 = _SIMD_LOAD(&q_dbl[4*offset]);
    q6 = _SIMD_LOAD(&q_dbl[5*offset]);

#ifdef BLOCK1
    q1 = _SIMD_ADD(q1, x1);
    q2 = _SIMD_ADD(q2, x2);
    q3 = _SIMD_ADD(q3, x3);
    q4 = _SIMD_ADD(q4, x4);
    q5 = _SIMD_ADD(q5, x5);
    q6 = _SIMD_ADD(q6, x6);
#endif


#ifdef BLOCK2
    q1 = _SIMD_ADD(q1, y1);
    q2 = _SIMD_ADD(q2, y2);
    q3 = _SIMD_ADD(q3, y3);
    q4 = _SIMD_ADD(q4, y4);
    q5 = _SIMD_ADD(q5, y5);
    q6 = _SIMD_ADD(q6, y6);
#endif

    _SIMD_STORE(&q_dbl[0], q1);
    _SIMD_STORE(&q_dbl[offset], q2);
    _SIMD_STORE(&q_dbl[2*offset], q3);
    _SIMD_STORE(&q_dbl[3*offset], q4);
    _SIMD_STORE(&q_dbl[4*offset], q5);
    _SIMD_STORE(&q_dbl[5*offset], q6);


#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+1)*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+1)*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h2_real = _SIMD_SET1(hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_SET1(hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */
 
     q1 = _SIMD_LOAD(&q_dbl[(ldq*2)+0]);
     q2 = _SIMD_LOAD(&q_dbl[(ldq*2)+offset]);
     q3 = _SIMD_LOAD(&q_dbl[(ldq*2)+2*offset]);
     q4 = _SIMD_LOAD(&q_dbl[(ldq*2)+3*offset]);
     q5 = _SIMD_LOAD(&q_dbl[(ldq*2)+4*offset]);
     q6 = _SIMD_LOAD(&q_dbl[(ldq*2)+5*offset]);

     q1 = _SIMD_ADD(q1, x1);
     q2 = _SIMD_ADD(q2, x2);
     q3 = _SIMD_ADD(q3, x3);
     q4 = _SIMD_ADD(q4, x4);
     q5 = _SIMD_ADD(q5, x5);
     q6 = _SIMD_ADD(q6, x6);

     tmp1 = _SIMD_MUL(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h2_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
     tmp2 = _SIMD_MUL(h2_imag, y2);
#ifdef __ELPA_USE_FMA__
     q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h2_real, y2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h2_imag, y3);
#ifdef __ELPA_USE_FMA__
     q3 = _SIMD_ADD(q3, _SIMD_FMADDSUB(h2_real, y3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     q3 = _SIMD_ADD(q3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
     tmp4 = _SIMD_MUL(h2_imag, y4);
#ifdef __ELPA_USE_FMA__
     q4 = _SIMD_ADD(q4, _SIMD_FMADDSUB(h2_real, y4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
     q4 = _SIMD_ADD(q4, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

     tmp5 = _SIMD_MUL(h2_imag, y5);
#ifdef __ELPA_USE_FMA__
     q5 = _SIMD_ADD(q5, _SIMD_FMADDSUB(h2_real, y5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
     q5 = _SIMD_ADD(q5, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif
     tmp6 = _SIMD_MUL(h2_imag, y6);
#ifdef __ELPA_USE_FMA__
     q6 = _SIMD_ADD(q6, _SIMD_FMADDSUB(h2_real, y6, _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#else
     q6 = _SIMD_ADD(q6, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y6), _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#endif

     _SIMD_STORE(&q_dbl[(ldq*2)+0], q1);
     _SIMD_STORE(&q_dbl[(ldq*2)+offset], q2);
     _SIMD_STORE(&q_dbl[(ldq*2)+2*offset], q3);
     _SIMD_STORE(&q_dbl[(ldq*2)+3*offset], q4);
     _SIMD_STORE(&q_dbl[(ldq*2)+4*offset], q5);
     _SIMD_STORE(&q_dbl[(ldq*2)+5*offset], q6);

#endif /* BLOCK2 */


    for (i = BLOCK; i < nb; i++)
    {

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = _mm_loaddup_pd(&hh_dbl[(i-BLOCK+1)*2]);
        h1_imag = _mm_loaddup_pd(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i-BLOCK+1)*2]) )));
        h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((i-BLOCK+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
	h1_real = _SIMD_BROADCAST(&hh_dbl[(i-BLOCK+1)*2]);
        h1_imag = _SIMD_BROADCAST(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
        h1_real = _SIMD_SET1(hh_dbl[(i-BLOCK+1)*2]);
        h1_imag = _SIMD_SET1(hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

        q1 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+0]);
        q2 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+offset]);
        q3 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+2*offset]);
        q4 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+3*offset]);
        q5 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+4*offset]);
        q6 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+5*offset]);

        tmp1 = _SIMD_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
        q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
        q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
        tmp2 = _SIMD_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
        q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h1_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
        q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif
        tmp3 = _SIMD_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
        q3 = _SIMD_ADD(q3, _SIMD_FMADDSUB(h1_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
        q3 = _SIMD_ADD(q3, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

         tmp4 = _SIMD_MUL(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
         q4 = _SIMD_ADD(q4, _SIMD_FMADDSUB(h1_real, x4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
         q4 = _SIMD_ADD(q4, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif
         tmp5 = _SIMD_MUL(h1_imag, x5);
#ifdef __ELPA_USE_FMA__
         q5 = _SIMD_ADD(q5, _SIMD_FMADDSUB(h1_real, x5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
         q5 = _SIMD_ADD(q5, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif
         tmp6 = _SIMD_MUL(h1_imag, x6);
#ifdef __ELPA_USE_FMA__
         q6 = _SIMD_ADD(q6, _SIMD_FMADDSUB(h1_real, x6, _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#else
         q6 = _SIMD_ADD(q6, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x6), _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#endif

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+i)*2]);
          h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+i)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+i)*2]) )));
          h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+i)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
	  h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
	  h2_real = _SIMD_SET1(hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_SET1(hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_512 */

          tmp1 = _SIMD_MUL(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
          q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h2_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
          tmp2 = _SIMD_MUL(h2_imag, y2);
#ifdef __ELPA_USE_FMA__
          q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h2_real, y2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
          q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

          tmp3 = _SIMD_MUL(h2_imag, y3);
#ifdef __ELPA_USE_FMA__
          q3 = _SIMD_ADD(q3, _SIMD_FMADDSUB(h2_real, y3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
          q3 = _SIMD_ADD(q3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
          tmp4 = _SIMD_MUL(h2_imag, y4);
#ifdef __ELPA_USE_FMA__
          q4 = _SIMD_ADD(q4, _SIMD_FMADDSUB(h2_real, y4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
          q4 = _SIMD_ADD(q4, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

          tmp5 = _SIMD_MUL(h2_imag, y5);
#ifdef __ELPA_USE_FMA__
          q5 = _SIMD_ADD(q5, _SIMD_FMADDSUB(h2_real, y5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
          q5 = _SIMD_ADD(q5, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif
          tmp6 = _SIMD_MUL(h2_imag, y6);
#ifdef __ELPA_USE_FMA__
          q6 = _SIMD_ADD(q6, _SIMD_FMADDSUB(h2_real, y6, _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#else
          q6 = _SIMD_ADD(q6, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y6), _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#endif

#endif /* BLOCK2 */


         _SIMD_STORE(&q_dbl[(2*i*ldq)+0], q1);
         _SIMD_STORE(&q_dbl[(2*i*ldq)+offset], q2);
         _SIMD_STORE(&q_dbl[(2*i*ldq)+2*offset], q3);
         _SIMD_STORE(&q_dbl[(2*i*ldq)+3*offset], q4);
         _SIMD_STORE(&q_dbl[(2*i*ldq)+4*offset], q5);
         _SIMD_STORE(&q_dbl[(2*i*ldq)+5*offset], q6);
    }
#ifdef BLOCK2

#if VEC_SET == SSE_128     
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[(nb-1)*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[((nb-1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(nb-1)*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((nb-1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_SET1(hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_512 */
     
     q1 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+0]);
     q2 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+offset]);
     q3 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+2*offset]);
     q4 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+3*offset]);
     q5 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+4*offset]);
     q6 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+5*offset]);

     tmp1 = _SIMD_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
     tmp2 = _SIMD_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
     q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h1_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
     q3 = _SIMD_ADD(q3, _SIMD_FMADDSUB(h1_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     q3 = _SIMD_ADD(q3, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
     tmp4 = _SIMD_MUL(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
     q4 = _SIMD_ADD(q4, _SIMD_FMADDSUB(h1_real, x4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
     q4 = _SIMD_ADD(q4, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

     tmp5 = _SIMD_MUL(h1_imag, x5);
#ifdef __ELPA_USE_FMA__
     q5 = _SIMD_ADD(q5, _SIMD_FMADDSUB(h1_real, x5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
     q5 = _SIMD_ADD(q5, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif
     tmp6 = _SIMD_MUL(h1_imag, x6);
#ifdef __ELPA_USE_FMA__
     q6 = _SIMD_ADD(q6, _SIMD_FMADDSUB(h1_real, x6, _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#else
     q6 = _SIMD_ADD(q6, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x6), _SIMD_SHUFFLE(tmp6, tmp6, _SHUFFLE)));
#endif

     _SIMD_STORE(&q_dbl[(2*nb*ldq)+0], q1);
     _SIMD_STORE(&q_dbl[(2*nb*ldq)+offset], q2);
     _SIMD_STORE(&q_dbl[(2*nb*ldq)+2*offset], q3);
     _SIMD_STORE(&q_dbl[(2*nb*ldq)+3*offset], q4);
     _SIMD_STORE(&q_dbl[(2*nb*ldq)+4*offset], q5);
     _SIMD_STORE(&q_dbl[(2*nb*ldq)+5*offset], q6);

#endif /* BLOCK2 */

}


#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 5
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 10
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 10
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 20
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 20
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 40
#endif
#endif /* VEC_SET == AVX_512 */
static __forceinline void CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq
#ifdef BLOCK1
		)
#endif
#ifdef BLOCK2
                ,int ldh, DATA_TYPE s)
#endif
{

    DATA_TYPE_REAL_PTR q_dbl = (DATA_TYPE_REAL_PTR)q;
    DATA_TYPE_REAL_PTR hh_dbl = (DATA_TYPE_REAL_PTR)hh;
#ifdef BLOCK2
    DATA_TYPE_REAL_PTR s_dbl = (DATA_TYPE_REAL_PTR)(&s);
#endif

    __SIMD_DATATYPE x1, x2, x3, x4, x5;
    __SIMD_DATATYPE q1, q2, q3, q4, q5;
#ifdef BLOCK2
    __SIMD_DATATYPE y1, y2, y3, y4, y5;
    __SIMD_DATATYPE h2_real, h2_imag;
#endif
    __SIMD_DATATYPE h1_real, h1_imag;
    __SIMD_DATATYPE tmp1, tmp2, tmp3, tmp4, tmp5;
    int i=0;

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
    __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set_epi64x(0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set_epi64x(0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_512 */

#ifdef BLOCK2
     x1 = _SIMD_LOAD(&q_dbl[(2*ldq)+0]);
     x2 = _SIMD_LOAD(&q_dbl[(2*ldq)+offset]);
     x3 = _SIMD_LOAD(&q_dbl[(2*ldq)+2*offset]);
     x4 = _SIMD_LOAD(&q_dbl[(2*ldq)+3*offset]);
     x5 = _SIMD_LOAD(&q_dbl[(2*ldq)+4*offset]);

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+1)*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+1)*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h2_real = _SIMD_SET1(hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_SET1(hh_dbl[((ldh+1)*2)+1]);
#endif /*  VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
     // conjugate
     h2_imag = _SIMD_XOR(h2_imag, sign);
#endif

     y1 = _SIMD_LOAD(&q_dbl[0]);
     y2 = _SIMD_LOAD(&q_dbl[offset]);
     y3 = _SIMD_LOAD(&q_dbl[2*offset]);
     y4 = _SIMD_LOAD(&q_dbl[3*offset]);
     y5 = _SIMD_LOAD(&q_dbl[4*offset]);

     tmp1 = _SIMD_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_ADD(y1, _SIMD_FMSUBADD(h2_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
     tmp2 = _SIMD_MUL(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
     y2 = _SIMD_ADD(y2, _SIMD_FMSUBADD(h2_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     y2 = _SIMD_ADD(y2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h2_imag, x3);
#ifdef __ELPA_USE_FMA__
     y3 = _SIMD_ADD(y3, _SIMD_FMSUBADD(h2_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     y3 = _SIMD_ADD(y3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
     tmp4 = _SIMD_MUL(h2_imag, x4);
#ifdef __ELPA_USE_FMA__
     y4 = _SIMD_ADD(y4, _SIMD_FMSUBADD(h2_real, x4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
     y4 = _SIMD_ADD(y4, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

     tmp5 = _SIMD_MUL(h2_imag, x5);
#ifdef __ELPA_USE_FMA__
     y5 = _SIMD_ADD(y5, _SIMD_FMSUBADD(h2_real, x5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
     y5 = _SIMD_ADD(y5, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

#ifdef BLOCK1
    x1 = _SIMD_LOAD(&q_dbl[0]);
    x2 = _SIMD_LOAD(&q_dbl[offset]);
    x3 = _SIMD_LOAD(&q_dbl[2*offset]);
    x4 = _SIMD_LOAD(&q_dbl[3*offset]);
    x5 = _SIMD_LOAD(&q_dbl[4*offset]);
#endif

    for (i = BLOCK; i < nb; i++)
    {

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = _mm_loaddup_pd(&hh_dbl[(i-BLOCK+1)*2]);
        h1_imag = _mm_loaddup_pd(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i-BLOCK+1)*2]) )));
        h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((i-BLOCK+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
       h1_real = _SIMD_BROADCAST(&hh_dbl[(i-BLOCK+1)*2]);
       h1_imag = _SIMD_BROADCAST(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
       h1_real = _SIMD_SET1(hh_dbl[(i-BLOCK+1)*2]);
       h1_imag = _SIMD_SET1(hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
        // conjugate
        h1_imag = _SIMD_XOR(h1_imag, sign);
#endif

        q1 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+0]);
        q2 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+offset]);
        q3 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+2*offset]);
        q4 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+3*offset]);
        q5 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+4*offset]);

        tmp1 = _SIMD_MUL(h1_imag, q1);

#ifdef __ELPA_USE_FMA__
        x1 = _SIMD_ADD(x1, _SIMD_FMSUBADD(h1_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
        x1 = _SIMD_ADD(x1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
        tmp2 = _SIMD_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
        x2 = _SIMD_ADD(x2, _SIMD_FMSUBADD(h1_real, q2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
        x2 = _SIMD_ADD(x2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif
        tmp3 = _SIMD_MUL(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
        x3 = _SIMD_ADD(x3, _SIMD_FMSUBADD(h1_real, q3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
        x3 = _SIMD_ADD(x3, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

        tmp4 = _SIMD_MUL(h1_imag, q4);
#ifdef __ELPA_USE_FMA__
        x4 = _SIMD_ADD(x4, _SIMD_FMSUBADD(h1_real, q4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
        x4 = _SIMD_ADD(x4, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif
        tmp5 = _SIMD_MUL(h1_imag, q5);
#ifdef __ELPA_USE_FMA__
        x5 = _SIMD_ADD(x5, _SIMD_FMSUBADD(h1_real, q5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
        x5 = _SIMD_ADD(x5, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+i)*2]);
          h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+i)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+i)*2]) )));
          h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+i)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
          h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
          h2_real = _SIMD_SET1(hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_SET1(hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
          // conjugate
          h2_imag = _SIMD_XOR(h2_imag, sign);
#endif

          tmp1 = _SIMD_MUL(h2_imag, q1);
#ifdef __ELPA_USE_FMA__
          y1 = _SIMD_ADD(y1, _SIMD_FMSUBADD(h2_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
          tmp2 = _SIMD_MUL(h2_imag, q2);
#ifdef __ELPA_USE_FMA__
          y2 = _SIMD_ADD(y2, _SIMD_FMSUBADD(h2_real, q2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
          y2 = _SIMD_ADD(y2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

          tmp3 = _SIMD_MUL(h2_imag, q3);
#ifdef __ELPA_USE_FMA__
          y3 = _SIMD_ADD(y3, _SIMD_FMSUBADD(h2_real, q3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
          y3 = _SIMD_ADD(y3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
          tmp4 = _SIMD_MUL(h2_imag, q4);
#ifdef __ELPA_USE_FMA__
          y4 = _SIMD_ADD(y4, _SIMD_FMSUBADD(h2_real, q4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
          y4 = _SIMD_ADD(y4, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

          tmp5 = _SIMD_MUL(h2_imag, q5);
#ifdef __ELPA_USE_FMA__
          y5 = _SIMD_ADD(y5, _SIMD_FMSUBADD(h2_real, q5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
          y5 = _SIMD_ADD(y5, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif
	
#endif /* BLOCK2 */

    }

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[(nb-1)*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[((nb-1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(nb-1)*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((nb-1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_SET1(hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
     // conjugate
     h1_imag = _SIMD_XOR(h1_imag, sign);
#endif

     q1 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+0]);
     q2 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+offset]);
     q3 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+2*offset]);
     q4 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+3*offset]);
     q5 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+4*offset]);

     tmp1 = _SIMD_MUL(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
     x1 = _SIMD_ADD(x1, _SIMD_FMSUBADD(h1_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     x1 = _SIMD_ADD(x1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
     tmp2 = _SIMD_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
     x2 = _SIMD_ADD(x2, _SIMD_FMSUBADD(h1_real, q2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     x2 = _SIMD_ADD(x2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
     x3 = _SIMD_ADD(x3, _SIMD_FMSUBADD(h1_real, q3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     x3 = _SIMD_ADD(x3, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
     tmp4 = _SIMD_MUL(h1_imag, q4);
#ifdef __ELPA_USE_FMA__
     x4 = _SIMD_ADD(x4, _SIMD_FMSUBADD(h1_real, q4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
     x4 = _SIMD_ADD(x4, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

     tmp5 = _SIMD_MUL(h1_imag, q5);
#ifdef __ELPA_USE_FMA__
     x5 = _SIMD_ADD(x5, _SIMD_FMSUBADD(h1_real, q5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
     x5 = _SIMD_ADD(x5, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
    h1_real = _mm_loaddup_pd(&hh_dbl[0]);
    h1_imag = _mm_loaddup_pd(&hh_dbl[1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[0]) )));
    h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[1]) )));
#endif
#endif /*  VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1_real = _SIMD_BROADCAST(&hh_dbl[0]);
    h1_imag = _SIMD_BROADCAST(&hh_dbl[1]);
#endif /* AVX_256 */

#if VEC_SET == AVX_512
    h1_real = _SIMD_SET1(hh_dbl[0]);
    h1_imag = _SIMD_SET1(hh_dbl[1]);

#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_imag, (__m512i) sign);
#endif
#endif
#ifdef HAVE_AVX512_XEON
#if defined(DOUBLE_PRECISION_COMPLEX) || defined(SINGLE_PRECISION_COMPLEX)
        h1_real = _SIMD_XOR(h1_real, sign);
        h1_imag = _SIMD_XOR(h1_imag, sign);
#endif
#endif

#endif /* VEC_SET == AVX_512 */

#if VEC_SET != AVX_512
    h1_real = _SIMD_XOR(h1_real, sign);
    h1_imag = _SIMD_XOR(h1_imag, sign);
#endif /* VEC_SET != AVX_512 */

    tmp1 = _SIMD_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
    x1 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif
    tmp2 = _SIMD_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
    x2 = _SIMD_FMADDSUB(h1_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#else
    x2 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#endif
    tmp3 = _SIMD_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
    x3 = _SIMD_FMADDSUB(h1_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#else
    x3 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#endif

    tmp4 = _SIMD_MUL(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
    x4 = _SIMD_FMADDSUB(h1_real, x4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE));
#else
    x4 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE));
#endif
    tmp5 = _SIMD_MUL(h1_imag, x5);
#ifdef __ELPA_USE_FMA__
    x5 = _SIMD_FMADDSUB(h1_real, x5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE));
#else
    x5 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE));
#endif

#ifdef BLOCK2

#if VEC_SET == SSE_128       
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[ldh*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[(ldh*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[ldh*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh*2)+1]) )));
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[ldh*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[(ldh*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[ldh*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[ldh*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[(ldh*2)+1]);
     h2_real = _SIMD_BROADCAST(&hh_dbl[ldh*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[(ldh*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[ldh*2]);
     h1_imag = _SIMD_SET1(hh_dbl[(ldh*2)+1]);
     h2_real = _SIMD_SET1(hh_dbl[ldh*2]);
     h2_imag = _SIMD_SET1(hh_dbl[(ldh*2)+1]);

#ifdef HAVE_AVX512_XEON_PHI

#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h1_real, (__m512i) sign);
     h1_imag = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
     h1_imag = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h2_real, (__m512i) sign);
     h2_imag = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h2_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h2_real, (__m512i) sign);
     h2_imag = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h2_imag, (__m512i) sign);
#endif
#endif /* HAVE_AVX512_XEON_PHI */

#ifdef HAVE_AVX512_XEON
#if defined(DOUBLE_PRECISION_COMPLEX) || defined(SINGLE_PRECISION_COMPLEX)
        h1_real = _SIMD_XOR(h1_real, sign);
        h1_imag = _SIMD_XOR(h1_imag, sign);
        h2_real = _SIMD_XOR(h2_real, sign);
        h2_imag = _SIMD_XOR(h2_imag, sign);
#endif
#endif     
#endif /* VEC_SET == AVX_512 */

#if VEC_SET != AVX_512
     h1_real = _SIMD_XOR(h1_real, sign);
     h1_imag = _SIMD_XOR(h1_imag, sign);
     h2_real = _SIMD_XOR(h2_real, sign);
     h2_imag = _SIMD_XOR(h2_imag, sign);
#endif /* VEC_SET != AVX_512 */

#if VEC_SET == SSE_128
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = _mm_castpd_ps(_mm_load_pd1((double *) s_dbl));
#else
     tmp2 = _SIMD_LOADU(s_dbl);
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
     tmp2 = _mm256_set_pd(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = _mm256_set_ps(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0],
                             s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
     tmp2 = _SIMD_SET(s_dbl[1], s_dbl[0],
                        s_dbl[1], s_dbl[0],
                        s_dbl[1], s_dbl[0],
                        s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = (__SIMD_DATATYPE) _mm512_set1_pd(*(double*)(&s_dbl[0]));
#endif
#endif /* VEC_SET == AVX_512 */

     tmp1 = _SIMD_MUL(h2_imag, tmp2);
#ifdef __ELPA_USE_FMA__
     tmp2 = _SIMD_FMADDSUB(h2_real, tmp2, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
     tmp2 = _SIMD_ADDSUB( _SIMD_MUL(h2_real, tmp2), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

#if VEC_SET == AVX_512
     _SIMD_MASK_STOREU(s_dbl, 0x01 + 0x02, tmp2);

     h2_real = _SIMD_SET1(s_dbl[0]);
     h2_imag = _SIMD_SET1(s_dbl[1]);
#endif /* VEC_SET == AVX_512 */

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_movedup_pd(tmp2);
     h2_imag = _mm_set1_pd(tmp2[1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(tmp2);
     h2_imag = _mm_movehdup_ps(tmp2);
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_SET1(tmp2[0]);
     h2_imag = _SIMD_SET1(tmp2[1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */
     tmp1 = _SIMD_MUL(h1_imag, y1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_FMADDSUB(h1_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
     y1 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif
     tmp2 = _SIMD_MUL(h1_imag, y2);
#ifdef __ELPA_USE_FMA__
     y2 = _SIMD_FMADDSUB(h1_real, y2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#else
     y2 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#endif

     tmp3 = _SIMD_MUL(h1_imag, y3);
#ifdef __ELPA_USE_FMA__
     y3 = _SIMD_FMADDSUB(h1_real, y3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#else
     y3 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#endif
     tmp4 = _SIMD_MUL(h1_imag, y4);
#ifdef __ELPA_USE_FMA__
     y4 = _SIMD_FMADDSUB(h1_real, y4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE));
#else
     y4 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE));
#endif

     tmp5 = _SIMD_MUL(h1_imag, y5);
#ifdef __ELPA_USE_FMA__
     y5 = _SIMD_FMADDSUB(h1_real, y5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE));
#else
     y5 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE));
#endif

     tmp1 = _SIMD_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_ADD(y1, _SIMD_FMADDSUB(h2_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
     tmp2 = _SIMD_MUL(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
     y2 = _SIMD_ADD(y2, _SIMD_FMADDSUB(h2_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     y2 = _SIMD_ADD(y2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h2_imag, x3);
#ifdef __ELPA_USE_FMA__
     y3 = _SIMD_ADD(y3, _SIMD_FMADDSUB(h2_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     y3 = _SIMD_ADD(y3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
     tmp4 = _SIMD_MUL(h2_imag, x4);
#ifdef __ELPA_USE_FMA__
     y4 = _SIMD_ADD(y4, _SIMD_FMADDSUB(h2_real, x4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
     y4 = _SIMD_ADD(y4, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

     tmp5 = _SIMD_MUL(h2_imag, x5);
#ifdef __ELPA_USE_FMA__
     y5 = _SIMD_ADD(y5, _SIMD_FMADDSUB(h2_real, x5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
     y5 = _SIMD_ADD(y5, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

    q1 = _SIMD_LOAD(&q_dbl[0]);
    q2 = _SIMD_LOAD(&q_dbl[offset]);
    q3 = _SIMD_LOAD(&q_dbl[2*offset]);
    q4 = _SIMD_LOAD(&q_dbl[3*offset]);
    q5 = _SIMD_LOAD(&q_dbl[4*offset]);

#ifdef BLOCK1
    q1 = _SIMD_ADD(q1, x1);
    q2 = _SIMD_ADD(q2, x2);
    q3 = _SIMD_ADD(q3, x3);
    q4 = _SIMD_ADD(q4, x4);
    q5 = _SIMD_ADD(q5, x5);
#endif


#ifdef BLOCK2
    q1 = _SIMD_ADD(q1, y1);
    q2 = _SIMD_ADD(q2, y2);
    q3 = _SIMD_ADD(q3, y3);
    q4 = _SIMD_ADD(q4, y4);
    q5 = _SIMD_ADD(q5, y5);
#endif
    _SIMD_STORE(&q_dbl[0], q1);
    _SIMD_STORE(&q_dbl[offset], q2);
    _SIMD_STORE(&q_dbl[2*offset], q3);
    _SIMD_STORE(&q_dbl[3*offset], q4);
    _SIMD_STORE(&q_dbl[4*offset], q5);


#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+1)*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+1)*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h2_real = _SIMD_SET1(hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_SET1(hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

     q1 = _SIMD_LOAD(&q_dbl[(ldq*2)+0]);
     q2 = _SIMD_LOAD(&q_dbl[(ldq*2)+offset]);
     q3 = _SIMD_LOAD(&q_dbl[(ldq*2)+2*offset]);
     q4 = _SIMD_LOAD(&q_dbl[(ldq*2)+3*offset]);
     q5 = _SIMD_LOAD(&q_dbl[(ldq*2)+4*offset]);

     q1 = _SIMD_ADD(q1, x1);
     q2 = _SIMD_ADD(q2, x2);
     q3 = _SIMD_ADD(q3, x3);
     q4 = _SIMD_ADD(q4, x4);
     q5 = _SIMD_ADD(q5, x5);

     tmp1 = _SIMD_MUL(h2_imag, y1);

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h2_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
     tmp2 = _SIMD_MUL(h2_imag, y2);
#ifdef __ELPA_USE_FMA__
     q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h2_real, y2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h2_imag, y3);
#ifdef __ELPA_USE_FMA__
     q3 = _SIMD_ADD(q3, _SIMD_FMADDSUB(h2_real, y3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     q3 = _SIMD_ADD(q3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
     tmp4 = _SIMD_MUL(h2_imag, y4);
#ifdef __ELPA_USE_FMA__
     q4 = _SIMD_ADD(q4, _SIMD_FMADDSUB(h2_real, y4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
     q4 = _SIMD_ADD(q4, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

     tmp5 = _SIMD_MUL(h2_imag, y5);
#ifdef __ELPA_USE_FMA__
     q5 = _SIMD_ADD(q5, _SIMD_FMADDSUB(h2_real, y5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
     q5 = _SIMD_ADD(q5, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif

     _SIMD_STORE(&q_dbl[(ldq*2)+0], q1);
     _SIMD_STORE(&q_dbl[(ldq*2)+offset], q2);
     _SIMD_STORE(&q_dbl[(ldq*2)+2*offset], q3);
     _SIMD_STORE(&q_dbl[(ldq*2)+3*offset], q4);
     _SIMD_STORE(&q_dbl[(ldq*2)+4*offset], q5);

#endif /* BLOCK2 */


    for (i = BLOCK; i < nb; i++)
    {

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = _mm_loaddup_pd(&hh_dbl[(i-BLOCK+1)*2]);
        h1_imag = _mm_loaddup_pd(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i-BLOCK+1)*2]) )));
        h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((i-BLOCK+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
	h1_real = _SIMD_BROADCAST(&hh_dbl[(i-BLOCK+1)*2]);
        h1_imag = _SIMD_BROADCAST(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
        h1_real = _SIMD_SET1(hh_dbl[(i-BLOCK+1)*2]);
        h1_imag = _SIMD_SET1(hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

        q1 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+0]);
        q2 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+offset]);
        q3 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+2*offset]);
        q4 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+3*offset]);
        q5 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+4*offset]);

	tmp1 = _SIMD_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
        q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
        q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
        tmp2 = _SIMD_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
        q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h1_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
        q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif
        tmp3 = _SIMD_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
        q3 = _SIMD_ADD(q3, _SIMD_FMADDSUB(h1_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
        q3 = _SIMD_ADD(q3, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

         tmp4 = _SIMD_MUL(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
         q4 = _SIMD_ADD(q4, _SIMD_FMADDSUB(h1_real, x4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
         q4 = _SIMD_ADD(q4, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif
         tmp5 = _SIMD_MUL(h1_imag, x5);
#ifdef __ELPA_USE_FMA__
         q5 = _SIMD_ADD(q5, _SIMD_FMADDSUB(h1_real, x5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
         q5 = _SIMD_ADD(q5, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+i)*2]);
          h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+i)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+i)*2]) )));
          h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+i)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
	  h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
	  h2_real = _SIMD_SET1(hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_SET1(hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_512 */

          tmp1 = _SIMD_MUL(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
          q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h2_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
          tmp2 = _SIMD_MUL(h2_imag, y2);
#ifdef __ELPA_USE_FMA__
          q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h2_real, y2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
          q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

          tmp3 = _SIMD_MUL(h2_imag, y3);
#ifdef __ELPA_USE_FMA__
          q3 = _SIMD_ADD(q3, _SIMD_FMADDSUB(h2_real, y3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
          q3 = _SIMD_ADD(q3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
          tmp4 = _SIMD_MUL(h2_imag, y4);
#ifdef __ELPA_USE_FMA__
          q4 = _SIMD_ADD(q4, _SIMD_FMADDSUB(h2_real, y4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
          q4 = _SIMD_ADD(q4, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

          tmp5 = _SIMD_MUL(h2_imag, y5);
#ifdef __ELPA_USE_FMA__
          q5 = _SIMD_ADD(q5, _SIMD_FMADDSUB(h2_real, y5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
          q5 = _SIMD_ADD(q5, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

         _SIMD_STORE(&q_dbl[(2*i*ldq)+0], q1);
         _SIMD_STORE(&q_dbl[(2*i*ldq)+offset], q2);
         _SIMD_STORE(&q_dbl[(2*i*ldq)+2*offset], q3);
         _SIMD_STORE(&q_dbl[(2*i*ldq)+3*offset], q4);
         _SIMD_STORE(&q_dbl[(2*i*ldq)+4*offset], q5);
    }
#ifdef BLOCK2

#if VEC_SET == SSE_128       
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[(nb-1)*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[((nb-1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(nb-1)*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((nb-1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_SET1(hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

     q1 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+0]);
     q2 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+offset]);
     q3 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+2*offset]);
     q4 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+3*offset]);
     q5 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+4*offset]);

     tmp1 = _SIMD_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
     tmp2 = _SIMD_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
     q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h1_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
     q3 = _SIMD_ADD(q3, _SIMD_FMADDSUB(h1_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     q3 = _SIMD_ADD(q3, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
     tmp4 = _SIMD_MUL(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
     q4 = _SIMD_ADD(q4, _SIMD_FMADDSUB(h1_real, x4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
     q4 = _SIMD_ADD(q4, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

     tmp5 = _SIMD_MUL(h1_imag, x5);
#ifdef __ELPA_USE_FMA__
     q5 = _SIMD_ADD(q5, _SIMD_FMADDSUB(h1_real, x5, _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#else
     q5 = _SIMD_ADD(q5, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x5), _SIMD_SHUFFLE(tmp5, tmp5, _SHUFFLE)));
#endif

     _SIMD_STORE(&q_dbl[(2*nb*ldq)+0], q1);
     _SIMD_STORE(&q_dbl[(2*nb*ldq)+offset], q2);
     _SIMD_STORE(&q_dbl[(2*nb*ldq)+2*offset], q3);
     _SIMD_STORE(&q_dbl[(2*nb*ldq)+3*offset], q4);
     _SIMD_STORE(&q_dbl[(2*nb*ldq)+4*offset], q5);

#endif /* BLOCK2 */

}

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 8
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 16
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 16
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 32
#endif
#endif /* VEC_SET == AVX_512 */
static __forceinline void CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq
#ifdef BLOCK1
		)
#endif
#ifdef BLOCK2
                ,int ldh, DATA_TYPE s)
#endif
{
    DATA_TYPE_REAL_PTR q_dbl = (DATA_TYPE_REAL_PTR)q;
    DATA_TYPE_REAL_PTR hh_dbl = (DATA_TYPE_REAL_PTR)hh;
#ifdef BLOCK2
    DATA_TYPE_REAL_PTR s_dbl = (DATA_TYPE_REAL_PTR)(&s);
#endif

    __SIMD_DATATYPE x1, x2, x3, x4;
    __SIMD_DATATYPE q1, q2, q3, q4;
#ifdef BLOCK2
    __SIMD_DATATYPE y1, y2, y3, y4;
    __SIMD_DATATYPE h2_real, h2_imag;
#endif
    __SIMD_DATATYPE h1_real, h1_imag;
    __SIMD_DATATYPE tmp1, tmp2, tmp3, tmp4;
    int i=0;

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
    __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set_epi64x(0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set_epi64x(0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_512 */

#ifdef BLOCK2
     x1 = _SIMD_LOAD(&q_dbl[(2*ldq)+0]);
     x2 = _SIMD_LOAD(&q_dbl[(2*ldq)+offset]);
     x3 = _SIMD_LOAD(&q_dbl[(2*ldq)+2*offset]);
     x4 = _SIMD_LOAD(&q_dbl[(2*ldq)+3*offset]);

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+1)*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+1)*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h2_real = _SIMD_SET1(hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_SET1(hh_dbl[((ldh+1)*2)+1]);
#endif /*  VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
     // conjugate
     h2_imag = _SIMD_XOR(h2_imag, sign);
#endif

     y1 = _SIMD_LOAD(&q_dbl[0]);
     y2 = _SIMD_LOAD(&q_dbl[offset]);
     y3 = _SIMD_LOAD(&q_dbl[2*offset]);
     y4 = _SIMD_LOAD(&q_dbl[3*offset]);

     tmp1 = _SIMD_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_ADD(y1, _SIMD_FMSUBADD(h2_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

     tmp2 = _SIMD_MUL(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
     y2 = _SIMD_ADD(y2, _SIMD_FMSUBADD(h2_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     y2 = _SIMD_ADD(y2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h2_imag, x3);
#ifdef __ELPA_USE_FMA__
     y3 = _SIMD_ADD(y3, _SIMD_FMSUBADD(h2_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     y3 = _SIMD_ADD(y3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

     tmp4 = _SIMD_MUL(h2_imag, x4);
#ifdef __ELPA_USE_FMA__
     y4 = _SIMD_ADD(y4, _SIMD_FMSUBADD(h2_real, x4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
     y4 = _SIMD_ADD(y4, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

#ifdef BLOCK1
     x1 = _SIMD_LOAD(&q_dbl[0]);
     x2 = _SIMD_LOAD(&q_dbl[offset]);
     x3 = _SIMD_LOAD(&q_dbl[2*offset]);
     x4 = _SIMD_LOAD(&q_dbl[3*offset]);
#endif

     for (i = BLOCK; i < nb; i++)
     {
#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h1_real = _mm_loaddup_pd(&hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _mm_loaddup_pd(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i-BLOCK+1)*2]) )));
          h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((i-BLOCK+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
          h1_real = _SIMD_BROADCAST(&hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _SIMD_BROADCAST(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
          h1_real = _SIMD_SET1(hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _SIMD_SET1(hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
          // conjugate
          h1_imag = _SIMD_XOR(h1_imag, sign);
#endif

          q1 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+0]);
          q2 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+offset]);
          q3 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+2*offset]);
          q4 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+3*offset]);

          tmp1 = _SIMD_MUL(h1_imag, q1);

#ifdef __ELPA_USE_FMA__
          x1 = _SIMD_ADD(x1, _SIMD_FMSUBADD(h1_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          x1 = _SIMD_ADD(x1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

          tmp2 = _SIMD_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
          x2 = _SIMD_ADD(x2, _SIMD_FMSUBADD(h1_real, q2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
          x2 = _SIMD_ADD(x2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

          tmp3 = _SIMD_MUL(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
          x3 = _SIMD_ADD(x3, _SIMD_FMSUBADD(h1_real, q3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
          x3 = _SIMD_ADD(x3, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
          tmp4 = _SIMD_MUL(h1_imag, q4);
#ifdef __ELPA_USE_FMA__
          x4 = _SIMD_ADD(x4, _SIMD_FMSUBADD(h1_real, q4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
          x4 = _SIMD_ADD(x4, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+i)*2]);
          h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+i)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+i)*2]) )));
          h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+i)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
          h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
          h2_real = _SIMD_SET1(hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_SET1(hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
          // conjugate
          h2_imag = _SIMD_XOR(h2_imag, sign);
#endif

          tmp1 = _SIMD_MUL(h2_imag, q1);
#ifdef __ELPA_USE_FMA__
          y1 = _SIMD_ADD(y1, _SIMD_FMSUBADD(h2_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
          tmp2 = _SIMD_MUL(h2_imag, q2);
#ifdef __ELPA_USE_FMA__
          y2 = _SIMD_ADD(y2, _SIMD_FMSUBADD(h2_real, q2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
          y2 = _SIMD_ADD(y2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

          tmp3 = _SIMD_MUL(h2_imag, q3);
#ifdef __ELPA_USE_FMA__
          y3 = _SIMD_ADD(y3, _SIMD_FMSUBADD(h2_real, q3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
          y3 = _SIMD_ADD(y3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
          tmp4 = _SIMD_MUL(h2_imag, q4);
#ifdef __ELPA_USE_FMA__
          y4 = _SIMD_ADD(y4, _SIMD_FMSUBADD(h2_real, q4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
          y4 = _SIMD_ADD(y4, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif
#endif /* BLOCK2 */
     }

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[(nb-1)*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[((nb-1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(nb-1)*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((nb-1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_SET1(hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
     // conjugate
     h1_imag = _SIMD_XOR(h1_imag, sign);
#endif

     q1 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+0]);
     q2 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+offset]);
     q3 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+2*offset]);
     q4 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+3*offset]);

     tmp1 = _SIMD_MUL(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
     x1 = _SIMD_ADD(x1, _SIMD_FMSUBADD(h1_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     x1 = _SIMD_ADD(x1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

     tmp2 = _SIMD_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
     x2 = _SIMD_ADD(x2, _SIMD_FMSUBADD(h1_real, q2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     x2 = _SIMD_ADD(x2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
     x3 = _SIMD_ADD(x3, _SIMD_FMSUBADD(h1_real, q3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     x3 = _SIMD_ADD(x3, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

     tmp4 = _SIMD_MUL(h1_imag, q4);
#ifdef __ELPA_USE_FMA__
     x4 = _SIMD_ADD(x4, _SIMD_FMSUBADD(h1_real, q4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
     x4 = _SIMD_ADD(x4, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[0]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[0]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1_real = _SIMD_BROADCAST(&hh_dbl[0]);
    h1_imag = _SIMD_BROADCAST(&hh_dbl[1]);
#endif /* AVX_256 */

#if VEC_SET == AVX_512
    h1_real = _SIMD_SET1(hh_dbl[0]);
    h1_imag = _SIMD_SET1(hh_dbl[1]);

#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_imag, (__m512i) sign);
#endif
#endif
#ifdef HAVE_AVX512_XEON
#if defined(DOUBLE_PRECISION_COMPLEX) || defined(SINGLE_PRECISION_COMPLEX)
        h1_real = _SIMD_XOR(h1_real, sign);
        h1_imag = _SIMD_XOR(h1_imag, sign);
#endif
#endif

#endif /* VEC_SET == AVX_512 */

#if VEC_SET != AVX_512
     h1_real = _SIMD_XOR(h1_real, sign);
     h1_imag = _SIMD_XOR(h1_imag, sign);
#endif /* VEC_SET != AVX_512 */

     tmp1 = _SIMD_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
     x1 = _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
     x1 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

     tmp2 = _SIMD_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
     x2 = _SIMD_FMADDSUB(h1_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#else
     x2 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#endif

     tmp3 = _SIMD_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
     x3 = _SIMD_FMADDSUB(h1_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#else
     x3 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#endif

     tmp4 = _SIMD_MUL(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
     x4 = _SIMD_FMADDSUB(h1_real, x4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE));
#else
     x4 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE));
#endif

#ifdef BLOCK2

#if VEC_SET == SSE_128       
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[ldh*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[(ldh*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[ldh*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh*2)+1]) )));
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[ldh*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[(ldh*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[ldh*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh*2)+1]) )));
#endif
#endif /* VEC_SET == 128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[ldh*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[(ldh*2)+1]);
     h2_real = _SIMD_BROADCAST(&hh_dbl[ldh*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[(ldh*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[ldh*2]);
     h1_imag = _SIMD_SET1(hh_dbl[(ldh*2)+1]);
     h2_real = _SIMD_SET1(hh_dbl[ldh*2]);
     h2_imag = _SIMD_SET1(hh_dbl[(ldh*2)+1]);

#ifdef HAVE_AVX512_XEON_PHI

#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h1_real, (__m512i) sign);
     h1_imag = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
     h1_imag = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h2_real, (__m512i) sign);
     h2_imag = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h2_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h2_real, (__m512i) sign);
     h2_imag = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h2_imag, (__m512i) sign);
#endif
#endif /* HAVE_AVX512_XEON_PHI */

#ifdef HAVE_AVX512_XEON
#if defined(DOUBLE_PRECISION_COMPLEX) || defined(SINGLE_PRECISION_COMPLEX)
        h1_real = _SIMD_XOR(h1_real, sign);
        h1_imag = _SIMD_XOR(h1_imag, sign);
        h2_real = _SIMD_XOR(h2_real, sign);
        h2_imag = _SIMD_XOR(h2_imag, sign);
#endif
#endif     
#endif /* VEC_SET == AVX_512 */

#if VEC_SET != AVX_512
     h1_real = _SIMD_XOR(h1_real, sign);
     h1_imag = _SIMD_XOR(h1_imag, sign);
     h2_real = _SIMD_XOR(h2_real, sign);
     h2_imag = _SIMD_XOR(h2_imag, sign);
#endif /* VEC_SET != AVX_512 */

#if VEC_SET == SSE_128
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = _mm_castpd_ps(_mm_load_pd1((double *) s_dbl));
#else
     tmp2 = _SIMD_LOADU(s_dbl);
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
     tmp2 = _mm256_set_pd(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = _mm256_set_ps(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0],
                             s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
     tmp2 = _SIMD_SET(s_dbl[1], s_dbl[0],
                        s_dbl[1], s_dbl[0],
                        s_dbl[1], s_dbl[0],
                        s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = (__SIMD_DATATYPE) _mm512_set1_pd(*(double*)(&s_dbl[0]));
#endif
#endif /* VEC_SET == AVX_512 */

     tmp1 = _SIMD_MUL(h2_imag, tmp2);
#ifdef __ELPA_USE_FMA__
     tmp2 = _SIMD_FMADDSUB(h2_real, tmp2, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
     tmp2 = _SIMD_ADDSUB( _SIMD_MUL(h2_real, tmp2), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

#if VEC_SET == AVX_512
     _SIMD_MASK_STOREU(s_dbl, 0x01 + 0x02, tmp2);

     h2_real = _SIMD_SET1(s_dbl[0]);
     h2_imag = _SIMD_SET1(s_dbl[1]);
#endif /* VEC_SET == AVX_512 */

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_movedup_pd(tmp2);
     h2_imag = _mm_set1_pd(tmp2[1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(tmp2);
     h2_imag = _mm_movehdup_ps(tmp2);
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_SET1(tmp2[0]);
     h2_imag = _SIMD_SET1(tmp2[1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */
     tmp1 = _SIMD_MUL(h1_imag, y1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_FMADDSUB(h1_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
     y1 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

     tmp2 = _SIMD_MUL(h1_imag, y2);
#ifdef __ELPA_USE_FMA__
     y2 = _SIMD_FMADDSUB(h1_real, y2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#else
     y2 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#endif

     tmp3 = _SIMD_MUL(h1_imag, y3);
#ifdef __ELPA_USE_FMA__
     y3 = _SIMD_FMADDSUB(h1_real, y3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#else
     y3 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#endif

     tmp4 = _SIMD_MUL(h1_imag, y4);
#ifdef __ELPA_USE_FMA__
     y4 = _SIMD_FMADDSUB(h1_real, y4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE));
#else
     y4 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE));
#endif

     tmp1 = _SIMD_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_ADD(y1, _SIMD_FMADDSUB(h2_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

     tmp2 = _SIMD_MUL(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
     y2 = _SIMD_ADD(y2, _SIMD_FMADDSUB(h2_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     y2 = _SIMD_ADD(y2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h2_imag, x3);
#ifdef __ELPA_USE_FMA__
     y3 = _SIMD_ADD(y3, _SIMD_FMADDSUB(h2_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     y3 = _SIMD_ADD(y3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

     tmp4 = _SIMD_MUL(h2_imag, x4);
#ifdef __ELPA_USE_FMA__
     y4 = _SIMD_ADD(y4, _SIMD_FMADDSUB(h2_real, x4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
     y4 = _SIMD_ADD(y4, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

     q1 = _SIMD_LOAD(&q_dbl[0]);
     q2 = _SIMD_LOAD(&q_dbl[offset]);
     q3 = _SIMD_LOAD(&q_dbl[2*offset]);
     q4 = _SIMD_LOAD(&q_dbl[3*offset]);

#ifdef BLOCK1
     q1 = _SIMD_ADD(q1, x1);
     q2 = _SIMD_ADD(q2, x2);
     q3 = _SIMD_ADD(q3, x3);
     q4 = _SIMD_ADD(q4, x4);
#endif

#ifdef BLOCK2
     q1 = _SIMD_ADD(q1, y1);
     q2 = _SIMD_ADD(q2, y2);
     q3 = _SIMD_ADD(q3, y3);
     q4 = _SIMD_ADD(q4, y4);
#endif

     _SIMD_STORE(&q_dbl[0], q1);
     _SIMD_STORE(&q_dbl[offset], q2);
     _SIMD_STORE(&q_dbl[2*offset], q3);
     _SIMD_STORE(&q_dbl[3*offset], q4);

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+1)*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+1)*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h2_real = _SIMD_SET1(hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_SET1(hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

     q1 = _SIMD_LOAD(&q_dbl[(ldq*2)+0]);
     q2 = _SIMD_LOAD(&q_dbl[(ldq*2)+offset]);
     q3 = _SIMD_LOAD(&q_dbl[(ldq*2)+2*offset]);
     q4 = _SIMD_LOAD(&q_dbl[(ldq*2)+3*offset]);

     q1 = _SIMD_ADD(q1, x1);
     q2 = _SIMD_ADD(q2, x2);
     q3 = _SIMD_ADD(q3, x3);
     q4 = _SIMD_ADD(q4, x4);

     tmp1 = _SIMD_MUL(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h2_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

     tmp2 = _SIMD_MUL(h2_imag, y2);
#ifdef __ELPA_USE_FMA__
     q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h2_real, y2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h2_imag, y3);
#ifdef __ELPA_USE_FMA__
     q3 = _SIMD_ADD(q3, _SIMD_FMADDSUB(h2_real, y3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     q3 = _SIMD_ADD(q3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

     tmp4 = _SIMD_MUL(h2_imag, y4);
#ifdef __ELPA_USE_FMA__
     q4 = _SIMD_ADD(q4, _SIMD_FMADDSUB(h2_real, y4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
     q4 = _SIMD_ADD(q4, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

     _SIMD_STORE(&q_dbl[(ldq*2)+0], q1);
     _SIMD_STORE(&q_dbl[(ldq*2)+offset], q2);
     _SIMD_STORE(&q_dbl[(ldq*2)+2*offset], q3);
     _SIMD_STORE(&q_dbl[(ldq*2)+3*offset], q4);

#endif /* BLOCK2 */

     for (i = BLOCK; i < nb; i++)
     {

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h1_real = _mm_loaddup_pd(&hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _mm_loaddup_pd(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i-BLOCK+1)*2]) )));
          h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((i-BLOCK+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
	  h1_real = _SIMD_BROADCAST(&hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _SIMD_BROADCAST(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
          h1_real = _SIMD_SET1(hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _SIMD_SET1(hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

          q1 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+0]);
          q2 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+offset]);
          q3 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+2*offset]);
          q4 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+3*offset]);

          tmp1 = _SIMD_MUL(h1_imag, x1);

#ifdef __ELPA_USE_FMA__
          q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
          tmp2 = _SIMD_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
          q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h1_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
          q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

          tmp3 = _SIMD_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
          q3 = _SIMD_ADD(q3, _SIMD_FMADDSUB(h1_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
          q3 = _SIMD_ADD(q3, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
          tmp4 = _SIMD_MUL(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
          q4 = _SIMD_ADD(q4, _SIMD_FMADDSUB(h1_real, x4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
          q4 = _SIMD_ADD(q4, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+i)*2]);
          h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+i)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+i)*2]) )));
          h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+i)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
	  h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
	  h2_real = _SIMD_SET1(hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_SET1(hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_512 */

          tmp1 = _SIMD_MUL(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
          q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h2_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
          tmp2 = _SIMD_MUL(h2_imag, y2);
#ifdef __ELPA_USE_FMA__
          q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h2_real, y2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
          q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

          tmp3 = _SIMD_MUL(h2_imag, y3);
#ifdef __ELPA_USE_FMA__
          q3 = _SIMD_ADD(q3, _SIMD_FMADDSUB(h2_real, y3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
          q3 = _SIMD_ADD(q3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
          tmp4 = _SIMD_MUL(h2_imag, y4);
#ifdef __ELPA_USE_FMA__
          q4 = _SIMD_ADD(q4, _SIMD_FMADDSUB(h2_real, y4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
          q4 = _SIMD_ADD(q4, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

          _SIMD_STORE(&q_dbl[(2*i*ldq)+0], q1);
          _SIMD_STORE(&q_dbl[(2*i*ldq)+offset], q2);
          _SIMD_STORE(&q_dbl[(2*i*ldq)+2*offset], q3);
          _SIMD_STORE(&q_dbl[(2*i*ldq)+3*offset], q4);

     }
#ifdef BLOCK2

#if VEC_SET == SSE_128     
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[(nb-1)*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[((nb-1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(nb-1)*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((nb-1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_SET1(hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

     q1 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+0]);
     q2 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+offset]);
     q3 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+2*offset]);
     q4 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+3*offset]);

     tmp1 = _SIMD_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
     tmp2 = _SIMD_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
     q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h1_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
     q3 = _SIMD_ADD(q3, _SIMD_FMADDSUB(h1_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     q3 = _SIMD_ADD(q3, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
     tmp4 = _SIMD_MUL(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
     q4 = _SIMD_ADD(q4, _SIMD_FMADDSUB(h1_real, x4, _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
     q4 = _SIMD_ADD(q4, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x4), _SIMD_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

     _SIMD_STORE(&q_dbl[(2*nb*ldq)+0], q1);
     _SIMD_STORE(&q_dbl[(2*nb*ldq)+offset], q2);
     _SIMD_STORE(&q_dbl[(2*nb*ldq)+2*offset], q3);
     _SIMD_STORE(&q_dbl[(2*nb*ldq)+3*offset], q4);

#endif /* BLOCK2 */
}


#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 3
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 6
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 6
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 12
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 12
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 24
#endif
#endif /* VEC_SET == AVX_512 */

static __forceinline void CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq
#ifdef BLOCK1
		)
#endif
#ifdef BLOCK2
                ,int ldh, DATA_TYPE s)
#endif
{
    DATA_TYPE_REAL_PTR q_dbl = (DATA_TYPE_REAL_PTR)q;
    DATA_TYPE_REAL_PTR hh_dbl = (DATA_TYPE_REAL_PTR)hh;
#ifdef BLOCK2
    DATA_TYPE_REAL_PTR s_dbl = (DATA_TYPE_REAL_PTR)(&s);
#endif

    __SIMD_DATATYPE x1, x2, x3;
    __SIMD_DATATYPE q1, q2, q3;
#ifdef BLOCK2
    __SIMD_DATATYPE y1, y2, y3;
    __SIMD_DATATYPE h2_real, h2_imag;
#endif
    __SIMD_DATATYPE h1_real, h1_imag;
    __SIMD_DATATYPE tmp1, tmp2, tmp3;
    int i=0;

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
    __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set_epi64x(0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set_epi64x(0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_512 */

#ifdef BLOCK2
     x1 = _SIMD_LOAD(&q_dbl[(2*ldq)+0]);
     x2 = _SIMD_LOAD(&q_dbl[(2*ldq)+offset]);
     x3 = _SIMD_LOAD(&q_dbl[(2*ldq)+2*offset]);

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+1)*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+1)*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h2_real = _SIMD_SET1(hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_SET1(hh_dbl[((ldh+1)*2)+1]);
#endif /*  VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
     // conjugate
     h2_imag = _SIMD_XOR(h2_imag, sign);
#endif

     y1 = _SIMD_LOAD(&q_dbl[0]);
     y2 = _SIMD_LOAD(&q_dbl[offset]);
     y3 = _SIMD_LOAD(&q_dbl[2*offset]);

     tmp1 = _SIMD_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_ADD(y1, _SIMD_FMSUBADD(h2_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

     tmp2 = _SIMD_MUL(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
     y2 = _SIMD_ADD(y2, _SIMD_FMSUBADD(h2_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     y2 = _SIMD_ADD(y2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h2_imag, x3);
#ifdef __ELPA_USE_FMA__
     y3 = _SIMD_ADD(y3, _SIMD_FMSUBADD(h2_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     y3 = _SIMD_ADD(y3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

#ifdef BLOCK1
     x1 = _SIMD_LOAD(&q_dbl[0]);
     x2 = _SIMD_LOAD(&q_dbl[offset]);
     x3 = _SIMD_LOAD(&q_dbl[2*offset]);
#endif

     for (i = BLOCK; i < nb; i++)
     {
#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h1_real = _mm_loaddup_pd(&hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _mm_loaddup_pd(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i-BLOCK+1)*2]) )));
          h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((i-BLOCK+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
          h1_real = _SIMD_BROADCAST(&hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _SIMD_BROADCAST(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
          h1_real = _SIMD_SET1(hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _SIMD_SET1(hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
          // conjugate
          h1_imag = _SIMD_XOR(h1_imag, sign);
#endif

          q1 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+0]);
          q2 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+offset]);
          q3 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+2*offset]);

          tmp1 = _SIMD_MUL(h1_imag, q1);

#ifdef __ELPA_USE_FMA__
          x1 = _SIMD_ADD(x1, _SIMD_FMSUBADD(h1_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          x1 = _SIMD_ADD(x1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

          tmp2 = _SIMD_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
          x2 = _SIMD_ADD(x2, _SIMD_FMSUBADD(h1_real, q2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
          x2 = _SIMD_ADD(x2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

          tmp3 = _SIMD_MUL(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
          x3 = _SIMD_ADD(x3, _SIMD_FMSUBADD(h1_real, q3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
          x3 = _SIMD_ADD(x3, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+i)*2]);
          h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+i)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+i)*2]) )));
          h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+i)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
          h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
          h2_real = _SIMD_SET1(hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_SET1(hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
          // conjugate
          h2_imag = _SIMD_XOR(h2_imag, sign);
#endif

          tmp1 = _SIMD_MUL(h2_imag, q1);
#ifdef __ELPA_USE_FMA__
          y1 = _SIMD_ADD(y1, _SIMD_FMSUBADD(h2_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
          tmp2 = _SIMD_MUL(h2_imag, q2);
#ifdef __ELPA_USE_FMA__
          y2 = _SIMD_ADD(y2, _SIMD_FMSUBADD(h2_real, q2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
          y2 = _SIMD_ADD(y2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

          tmp3 = _SIMD_MUL(h2_imag, q3);
#ifdef __ELPA_USE_FMA__
          y3 = _SIMD_ADD(y3, _SIMD_FMSUBADD(h2_real, q3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
          y3 = _SIMD_ADD(y3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
#endif /* BLOCK2 */
     }

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[(nb-1)*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[((nb-1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(nb-1)*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((nb-1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_SET1(hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
     // conjugate
     h1_imag = _SIMD_XOR(h1_imag, sign);
#endif

     q1 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+0]);
     q2 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+offset]);
     q3 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+2*offset]);

     tmp1 = _SIMD_MUL(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
     x1 = _SIMD_ADD(x1, _SIMD_FMSUBADD(h1_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     x1 = _SIMD_ADD(x1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

     tmp2 = _SIMD_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
     x2 = _SIMD_ADD(x2, _SIMD_FMSUBADD(h1_real, q2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     x2 = _SIMD_ADD(x2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
     x3 = _SIMD_ADD(x3, _SIMD_FMSUBADD(h1_real, q3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     x3 = _SIMD_ADD(x3, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[0]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[0]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1_real = _SIMD_BROADCAST(&hh_dbl[0]);
    h1_imag = _SIMD_BROADCAST(&hh_dbl[1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
    h1_real = _SIMD_SET1(hh_dbl[0]);
    h1_imag = _SIMD_SET1(hh_dbl[1]);

#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_imag, (__m512i) sign);
#endif
#endif /* HAVE_AVX512_XEON_PHI */

#ifdef HAVE_AVX512_XEON
        h1_real = _SIMD_XOR(h1_real, sign);
        h1_imag = _SIMD_XOR(h1_imag, sign);
#endif

#endif /* VEC_SET == AVX_512 */

#if VEC_SET != AVX_512
     h1_real = _SIMD_XOR(h1_real, sign);
     h1_imag = _SIMD_XOR(h1_imag, sign);
#endif /* VEC_SET != AVX_512 */

     tmp1 = _SIMD_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
     x1 = _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
     x1 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

     tmp2 = _SIMD_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
     x2 = _SIMD_FMADDSUB(h1_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#else
     x2 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#endif

     tmp3 = _SIMD_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
     x3 = _SIMD_FMADDSUB(h1_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#else
     x3 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#endif

#ifdef BLOCK2

#if VEC_SET == SSE_128        
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[ldh*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[(ldh*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[ldh*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh*2)+1]) )));
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[ldh*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[(ldh*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[ldh*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh*2)+1]) )));
#endif
#endif /* VEC_SET == 128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[ldh*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[(ldh*2)+1]);
     h2_real = _SIMD_BROADCAST(&hh_dbl[ldh*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[(ldh*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[ldh*2]);
     h1_imag = _SIMD_SET1(hh_dbl[(ldh*2)+1]);
     h2_real = _SIMD_SET1(hh_dbl[ldh*2]);
     h2_imag = _SIMD_SET1(hh_dbl[(ldh*2)+1]);

#ifdef HAVE_AVX512_XEON_PHI

#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h1_real, (__m512i) sign);
     h1_imag = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
     h1_imag = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h2_real, (__m512i) sign);
     h2_imag = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h2_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h2_real, (__m512i) sign);
     h2_imag = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h2_imag, (__m512i) sign);
#endif
#endif /* HAVE_AVX512_XEON_PHI */

#ifdef HAVE_AVX512_XEON
        h1_real = _SIMD_XOR(h1_real, sign);
        h1_imag = _SIMD_XOR(h1_imag, sign);
        h2_real = _SIMD_XOR(h2_real, sign);
        h2_imag = _SIMD_XOR(h2_imag, sign);
#endif     
#endif /* VEC_SET == AVX_512 */

#if VEC_SET != AVX_512
     h1_real = _SIMD_XOR(h1_real, sign);
     h1_imag = _SIMD_XOR(h1_imag, sign);
     h2_real = _SIMD_XOR(h2_real, sign);
     h2_imag = _SIMD_XOR(h2_imag, sign);
#endif /* VEC_SET != AVX_512 */

#if VEC_SET == SSE_128
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = _mm_castpd_ps(_mm_load_pd1((double *) s_dbl));
#else
     tmp2 = _SIMD_LOADU(s_dbl);
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
     tmp2 = _mm256_set_pd(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = _mm256_set_ps(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0],
                             s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
     tmp2 = _SIMD_SET(s_dbl[1], s_dbl[0],
                        s_dbl[1], s_dbl[0],
                        s_dbl[1], s_dbl[0],
                        s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = (__SIMD_DATATYPE) _mm512_set1_pd(*(double*)(&s_dbl[0]));
#endif
#endif /* VEC_SET == AVX_512 */


     tmp1 = _SIMD_MUL(h2_imag, tmp2);
#ifdef __ELPA_USE_FMA__
     tmp2 = _SIMD_FMADDSUB(h2_real, tmp2, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
     tmp2 = _SIMD_ADDSUB( _SIMD_MUL(h2_real, tmp2), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

#if VEC_SET == AVX_512
     _SIMD_MASK_STOREU(s_dbl, 0x01 + 0x02, tmp2);

     h2_real = _SIMD_SET1(s_dbl[0]);
     h2_imag = _SIMD_SET1(s_dbl[1]);
#endif

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_movedup_pd(tmp2);
     h2_imag = _mm_set1_pd(tmp2[1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(tmp2);
     h2_imag = _mm_movehdup_ps(tmp2);
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_SET1(tmp2[0]);
     h2_imag = _SIMD_SET1(tmp2[1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

     tmp1 = _SIMD_MUL(h1_imag, y1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_FMADDSUB(h1_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
     y1 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

     tmp2 = _SIMD_MUL(h1_imag, y2);
#ifdef __ELPA_USE_FMA__
     y2 = _SIMD_FMADDSUB(h1_real, y2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#else
     y2 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#endif

     tmp3 = _SIMD_MUL(h1_imag, y3);
#ifdef __ELPA_USE_FMA__
     y3 = _SIMD_FMADDSUB(h1_real, y3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#else
     y3 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#endif

     tmp1 = _SIMD_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_ADD(y1, _SIMD_FMADDSUB(h2_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

     tmp2 = _SIMD_MUL(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
     y2 = _SIMD_ADD(y2, _SIMD_FMADDSUB(h2_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     y2 = _SIMD_ADD(y2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h2_imag, x3);
#ifdef __ELPA_USE_FMA__
     y3 = _SIMD_ADD(y3, _SIMD_FMADDSUB(h2_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     y3 = _SIMD_ADD(y3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

     q1 = _SIMD_LOAD(&q_dbl[0]);
     q2 = _SIMD_LOAD(&q_dbl[offset]);
     q3 = _SIMD_LOAD(&q_dbl[2*offset]);

#ifdef BLOCK1
     q1 = _SIMD_ADD(q1, x1);
     q2 = _SIMD_ADD(q2, x2);
     q3 = _SIMD_ADD(q3, x3);
#endif

#ifdef BLOCK2
     q1 = _SIMD_ADD(q1, y1);
     q2 = _SIMD_ADD(q2, y2);
     q3 = _SIMD_ADD(q3, y3);
#endif

     _SIMD_STORE(&q_dbl[0], q1);
     _SIMD_STORE(&q_dbl[offset], q2);
     _SIMD_STORE(&q_dbl[2*offset], q3);

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+1)*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+1)*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h2_real = _SIMD_SET1(hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_SET1(hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

     q1 = _SIMD_LOAD(&q_dbl[(ldq*2)+0]);
     q2 = _SIMD_LOAD(&q_dbl[(ldq*2)+offset]);
     q3 = _SIMD_LOAD(&q_dbl[(ldq*2)+2*offset]);

     q1 = _SIMD_ADD(q1, x1);
     q2 = _SIMD_ADD(q2, x2);
     q3 = _SIMD_ADD(q3, x3);

     tmp1 = _SIMD_MUL(h2_imag, y1);

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h2_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

     tmp2 = _SIMD_MUL(h2_imag, y2);
#ifdef __ELPA_USE_FMA__
     q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h2_real, y2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h2_imag, y3);
#ifdef __ELPA_USE_FMA__
     q3 = _SIMD_ADD(q3, _SIMD_FMADDSUB(h2_real, y3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     q3 = _SIMD_ADD(q3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

     _SIMD_STORE(&q_dbl[(ldq*2)+0], q1);
     _SIMD_STORE(&q_dbl[(ldq*2)+offset], q2);
     _SIMD_STORE(&q_dbl[(ldq*2)+2*offset], q3);

#endif /* BLOCK2 */

     for (i = BLOCK; i < nb; i++)
     {

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h1_real = _mm_loaddup_pd(&hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _mm_loaddup_pd(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i-BLOCK+1)*2]) )));
          h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((i-BLOCK+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
	h1_real = _SIMD_BROADCAST(&hh_dbl[(i-BLOCK+1)*2]);
        h1_imag = _SIMD_BROADCAST(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
        h1_real = _SIMD_SET1(hh_dbl[(i-BLOCK+1)*2]);
        h1_imag = _SIMD_SET1(hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

          q1 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+0]);
          q2 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+offset]);
          q3 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+2*offset]);

          tmp1 = _SIMD_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
          q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
          tmp2 = _SIMD_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
          q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h1_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
          q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

          tmp3 = _SIMD_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
          q3 = _SIMD_ADD(q3, _SIMD_FMADDSUB(h1_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
          q3 = _SIMD_ADD(q3, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+i)*2]);
          h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+i)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+i)*2]) )));
          h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+i)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
	  h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
        h2_real = _SIMD_SET1(hh_dbl[(ldh+i)*2]);
        h2_imag = _SIMD_SET1(hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_512 */

          tmp1 = _SIMD_MUL(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
          q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h2_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
          tmp2 = _SIMD_MUL(h2_imag, y2);
#ifdef __ELPA_USE_FMA__
          q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h2_real, y2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
          q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

          tmp3 = _SIMD_MUL(h2_imag, y3);
#ifdef __ELPA_USE_FMA__
          q3 = _SIMD_ADD(q3, _SIMD_FMADDSUB(h2_real, y3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
          q3 = _SIMD_ADD(q3, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

          _SIMD_STORE(&q_dbl[(2*i*ldq)+0], q1);
          _SIMD_STORE(&q_dbl[(2*i*ldq)+offset], q2);
          _SIMD_STORE(&q_dbl[(2*i*ldq)+2*offset], q3);

     }
#ifdef BLOCK2

#if VEC_SET == SSE_128     
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[(nb-1)*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[((nb-1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(nb-1)*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((nb-1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_SET1(hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

     q1 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+0]);
     q2 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+offset]);
     q3 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+2*offset]);

     tmp1 = _SIMD_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
     tmp2 = _SIMD_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
     q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h1_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     tmp3 = _SIMD_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
     q3 = _SIMD_ADD(q3, _SIMD_FMADDSUB(h1_real, x3, _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
     q3 = _SIMD_ADD(q3, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x3), _SIMD_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

     _SIMD_STORE(&q_dbl[(2*nb*ldq)+0], q1);
     _SIMD_STORE(&q_dbl[(2*nb*ldq)+offset], q2);
     _SIMD_STORE(&q_dbl[(2*nb*ldq)+2*offset], q3);

#endif /* BLOCK2 */
}


#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 2
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 4
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 8
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 16
#endif
#endif /* VEC_SET == AVX_512 */

static __forceinline void CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq
#ifdef BLOCK1
		)
#endif
#ifdef BLOCK2
                ,int ldh, DATA_TYPE s)
#endif
{

     DATA_TYPE_REAL_PTR q_dbl = (DATA_TYPE_REAL_PTR)q;
     DATA_TYPE_REAL_PTR hh_dbl = (DATA_TYPE_REAL_PTR)hh;
#ifdef BLOCK2
     DATA_TYPE_REAL_PTR s_dbl = (DATA_TYPE_REAL_PTR)(&s);
#endif

     __SIMD_DATATYPE x1, x2;
     __SIMD_DATATYPE q1, q2;
#ifdef BLOCK2
     __SIMD_DATATYPE y1, y2;
     __SIMD_DATATYPE h2_real, h2_imag;
#endif
     __SIMD_DATATYPE h1_real, h1_imag;
     __SIMD_DATATYPE tmp1, tmp2;
     int i=0;

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set_epi64x(0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set_epi64x(0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_512 */

#ifdef BLOCK2
     x1 = _SIMD_LOAD(&q_dbl[(2*ldq)+0]);
     x2 = _SIMD_LOAD(&q_dbl[(2*ldq)+offset]);

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+1)*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+1)*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h2_real = _SIMD_SET1(hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_SET1(hh_dbl[((ldh+1)*2)+1]);
#endif

#ifndef __ELPA_USE_FMA__
     // conjugate
     h2_imag = _SIMD_XOR(h2_imag, sign);
#endif

     y1 = _SIMD_LOAD(&q_dbl[0]);
     y2 = _SIMD_LOAD(&q_dbl[offset]);

     tmp1 = _SIMD_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_ADD(y1, _SIMD_FMSUBADD(h2_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
     tmp2 = _SIMD_MUL(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
     y2 = _SIMD_ADD(y2, _SIMD_FMSUBADD(h2_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     y2 = _SIMD_ADD(y2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

#ifdef BLOCK1
     x1 = _SIMD_LOAD(&q_dbl[0]);
     x2 = _SIMD_LOAD(&q_dbl[offset]);
#endif

     for (i = BLOCK; i < nb; i++)
     {
#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h1_real = _mm_loaddup_pd(&hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _mm_loaddup_pd(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i-BLOCK+1)*2]) )));
          h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((i-BLOCK+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
          h1_real = _SIMD_BROADCAST(&hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _SIMD_BROADCAST(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
          h1_real = _SIMD_SET1(hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _SIMD_SET1(hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
          // conjugate
          h1_imag = _SIMD_XOR(h1_imag, sign);
#endif

          q1 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+0]);
          q2 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+offset]);
          tmp1 = _SIMD_MUL(h1_imag, q1);

#ifdef __ELPA_USE_FMA__
          x1 = _SIMD_ADD(x1, _SIMD_FMSUBADD(h1_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          x1 = _SIMD_ADD(x1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

          tmp2 = _SIMD_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
          x2 = _SIMD_ADD(x2, _SIMD_FMSUBADD(h1_real, q2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
          x2 = _SIMD_ADD(x2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+i)*2]);
          h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+i)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+i)*2]) )));
          h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+i)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
          h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
          h2_real = _SIMD_SET1(hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_SET1(hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
          // conjugate
          h2_imag = _SIMD_XOR(h2_imag, sign);
#endif

          tmp1 = _SIMD_MUL(h2_imag, q1);
#ifdef __ELPA_USE_FMA__
          y1 = _SIMD_ADD(y1, _SIMD_FMSUBADD(h2_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
          tmp2 = _SIMD_MUL(h2_imag, q2);
#ifdef __ELPA_USE_FMA__
          y2 = _SIMD_ADD(y2, _SIMD_FMSUBADD(h2_real, q2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
          y2 = _SIMD_ADD(y2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

     }

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[(nb-1)*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[((nb-1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(nb-1)*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((nb-1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_SET1(hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
     // conjugate
     h1_imag = _SIMD_XOR(h1_imag, sign);
#endif

     q1 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+0]);
     q2 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+offset]);

     tmp1 = _SIMD_MUL(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
     x1 = _SIMD_ADD(x1, _SIMD_FMSUBADD(h1_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     x1 = _SIMD_ADD(x1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
     tmp2 = _SIMD_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
     x2 = _SIMD_ADD(x2, _SIMD_FMSUBADD(h1_real, q2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     x2 = _SIMD_ADD(x2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[0]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[0]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1_real = _SIMD_BROADCAST(&hh_dbl[0]);
    h1_imag = _SIMD_BROADCAST(&hh_dbl[1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
    h1_real = _SIMD_SET1(hh_dbl[0]);
    h1_imag = _SIMD_SET1(hh_dbl[1]);

#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_imag, (__m512i) sign);
#endif
#endif
#ifdef HAVE_AVX512_XEON
        h1_real = _SIMD_XOR(h1_real, sign);
        h1_imag = _SIMD_XOR(h1_imag, sign);
#endif

#endif /* VEC_SET == AVX_512 */

#if VEC_SET != AVX_512
     h1_real = _SIMD_XOR(h1_real, sign);
     h1_imag = _SIMD_XOR(h1_imag, sign);
#endif /* VEC_SET != AVX_512 */

     tmp1 = _SIMD_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
     x1 = _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
     x1 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

     tmp2 = _SIMD_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
     x2 = _SIMD_FMADDSUB(h1_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#else
     x2 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#endif

#ifdef BLOCK2

#if VEC_SET == SSE_128     
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[ldh*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[(ldh*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[ldh*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh*2)+1]) )));
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[ldh*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[(ldh*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[ldh*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh*2)+1]) )));
#endif
#endif /* VEC_SET == 128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[ldh*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[(ldh*2)+1]);
     h2_real = _SIMD_BROADCAST(&hh_dbl[ldh*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[(ldh*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[ldh*2]);
     h1_imag = _SIMD_SET1(hh_dbl[(ldh*2)+1]);
     h2_real = _SIMD_SET1(hh_dbl[ldh*2]);
     h2_imag = _SIMD_SET1(hh_dbl[(ldh*2)+1]);

#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h1_real, (__m512i) sign);
     h1_imag = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
     h1_imag = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h2_real, (__m512i) sign);
     h2_imag = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h2_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h2_real, (__m512i) sign);
     h2_imag = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h2_imag, (__m512i) sign);
#endif
#endif
#ifdef HAVE_AVX512_XEON
     h1_real = _SIMD_XOR(h1_real, sign);
     h1_imag = _SIMD_XOR(h1_imag, sign);
     h2_real = _SIMD_XOR(h2_real, sign);
     h2_imag = _SIMD_XOR(h2_imag, sign);
#endif

#endif /* VEC_SET == AVX_512 */

#if VEC_SET != AVX_512
     h1_real = _SIMD_XOR(h1_real, sign);
     h1_imag = _SIMD_XOR(h1_imag, sign);
     h2_real = _SIMD_XOR(h2_real, sign);
     h2_imag = _SIMD_XOR(h2_imag, sign);
#endif /* VEC_SET != AVX_512 */

#if VEC_SET == SSE_128
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = _mm_castpd_ps(_mm_load_pd1((double *) s_dbl));
#else
     tmp2 = _SIMD_LOADU(s_dbl);
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
     tmp2 = _mm256_set_pd(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = _mm256_set_ps(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0],
                             s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
     tmp2 = _SIMD_SET(s_dbl[1], s_dbl[0],
                          s_dbl[1], s_dbl[0],
                          s_dbl[1], s_dbl[0],
                          s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = (__SIMD_DATATYPE) _mm512_set1_pd(*(double*)(&s_dbl[0]));
#endif

#endif /* VEC_SET == AVX_512 */

     tmp1 = _SIMD_MUL(h2_imag, tmp2);
#ifdef __ELPA_USE_FMA__
     tmp2 = _SIMD_FMADDSUB(h2_real, tmp2, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
     tmp2 = _SIMD_ADDSUB( _SIMD_MUL(h2_real, tmp2), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

#if VEC_SET == AVX_512
     _SIMD_MASK_STOREU(s_dbl, 0x01 + 0x02, tmp2);

     h2_real = _SIMD_SET1(s_dbl[0]);
     h2_imag = _SIMD_SET1(s_dbl[1]);
#endif

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_movedup_pd(tmp2);
     h2_imag = _mm_set1_pd(tmp2[1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(tmp2);
     h2_imag = _mm_movehdup_ps(tmp2);
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_SET1(tmp2[0]);
     h2_imag = _SIMD_SET1(tmp2[1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

     tmp1 = _SIMD_MUL(h1_imag, y1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_FMADDSUB(h1_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
     y1 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif
     tmp2 = _SIMD_MUL(h1_imag, y2);
#ifdef __ELPA_USE_FMA__
     y2 = _SIMD_FMADDSUB(h1_real, y2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#else
     y2 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#endif

     tmp1 = _SIMD_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_ADD(y1, _SIMD_FMADDSUB(h2_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
     tmp2 = _SIMD_MUL(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
     y2 = _SIMD_ADD(y2, _SIMD_FMADDSUB(h2_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     y2 = _SIMD_ADD(y2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

     q1 = _SIMD_LOAD(&q_dbl[0]);
     q2 = _SIMD_LOAD(&q_dbl[offset]);

#ifdef BLOCK1
     q1 = _SIMD_ADD(q1, x1);
     q2 = _SIMD_ADD(q2, x2);
#endif

#ifdef BLOCK2
     q1 = _SIMD_ADD(q1, y1);
     q2 = _SIMD_ADD(q2, y2);
#endif
     _SIMD_STORE(&q_dbl[0], q1);
     _SIMD_STORE(&q_dbl[offset], q2);

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+1)*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+1)*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h2_real = _SIMD_SET1(hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_SET1(hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

     q1 = _SIMD_LOAD(&q_dbl[(ldq*2)+0]);
     q2 = _SIMD_LOAD(&q_dbl[(ldq*2)+offset]);

     q1 = _SIMD_ADD(q1, x1);
     q2 = _SIMD_ADD(q2, x2);

     tmp1 = _SIMD_MUL(h2_imag, y1);

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h2_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
     tmp2 = _SIMD_MUL(h2_imag, y2);
#ifdef __ELPA_USE_FMA__
     q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h2_real, y2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     _SIMD_STORE(&q_dbl[(ldq*2)+0], q1);
     _SIMD_STORE(&q_dbl[(ldq*2)+offset], q2);

#endif /* BLOCK2 */

     for (i = BLOCK; i < nb; i++)
     {
#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h1_real = _mm_loaddup_pd(&hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _mm_loaddup_pd(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i-BLOCK+1)*2]) )));
          h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((i-BLOCK+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
	  h1_real = _SIMD_BROADCAST(&hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _SIMD_BROADCAST(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
          h1_real = _SIMD_SET1(hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _SIMD_SET1(hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

          q1 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+0]);
          q2 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+offset]);
          tmp1 = _SIMD_MUL(h1_imag, x1);

#ifdef __ELPA_USE_FMA__
          q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

          tmp2 = _SIMD_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
          q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h1_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
          q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif


#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+i)*2]);
          h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+i)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+i)*2]) )));
          h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+i)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
	  h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
         h2_real = _SIMD_SET1(hh_dbl[(ldh+i)*2]);
         h2_imag = _SIMD_SET1(hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_512 */

          tmp1 = _SIMD_MUL(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
          q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h2_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
          tmp2 = _SIMD_MUL(h2_imag, y2);
#ifdef __ELPA_USE_FMA__
          q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h2_real, y2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
          q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

          _SIMD_STORE(&q_dbl[(2*i*ldq)+0], q1);
          _SIMD_STORE(&q_dbl[(2*i*ldq)+offset], q2);
    }
#ifdef BLOCK2

#if VEC_SET == SSE_128     
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[(nb-1)*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[((nb-1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(nb-1)*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((nb-1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_SET1(hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

     q1 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+0]);
     q2 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+offset]);

     tmp1 = _SIMD_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
     tmp2 = _SIMD_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
     q2 = _SIMD_ADD(q2, _SIMD_FMADDSUB(h1_real, x2, _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
     q2 = _SIMD_ADD(q2, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x2), _SIMD_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

     _SIMD_STORE(&q_dbl[(2*nb*ldq)+0], q1);
     _SIMD_STORE(&q_dbl[(2*nb*ldq)+offset], q2);

#endif /* BLOCK2 */

}

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 1
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 2
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 2
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 4
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define ROW_LENGTH 8
#endif
#endif /* VEC_SET == AVX_512 */


static __forceinline void CONCAT_8ARGS(hh_trafo_complex_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq
#ifdef BLOCK1
		)
#endif
#ifdef BLOCK2
                ,int ldh, DATA_TYPE s)
#endif
{

     DATA_TYPE_REAL_PTR q_dbl = (DATA_TYPE_REAL_PTR)q;
     DATA_TYPE_REAL_PTR hh_dbl = (DATA_TYPE_REAL_PTR)hh;
#ifdef BLOCK2
     DATA_TYPE_REAL_PTR s_dbl = (DATA_TYPE_REAL_PTR)(&s);
#endif

     __SIMD_DATATYPE x1;
     __SIMD_DATATYPE q1;
#ifdef BLOCK2
     __SIMD_DATATYPE y1;
     __SIMD_DATATYPE h2_real, h2_imag;
#endif
     __SIMD_DATATYPE h1_real, h1_imag;
     __SIMD_DATATYPE tmp1, tmp2;
     int i=0;

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set_epi64x(0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set_epi64x(0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_512 */

#ifdef BLOCK2
     x1 = _SIMD_LOAD(&q_dbl[(2*ldq)+0]);

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+1)*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+1)*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h2_real = _SIMD_SET1(hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_SET1(hh_dbl[((ldh+1)*2)+1]);
#endif

#ifndef __ELPA_USE_FMA__
     // conjugate
     h2_imag = _SIMD_XOR(h2_imag, sign);
#endif

     y1 = _SIMD_LOAD(&q_dbl[0]);

     tmp1 = _SIMD_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_ADD(y1, _SIMD_FMSUBADD(h2_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

#ifdef BLOCK1
     x1 = _SIMD_LOAD(&q_dbl[0]);
#endif

     for (i = BLOCK; i < nb; i++)
     {
#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h1_real = _mm_loaddup_pd(&hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _mm_loaddup_pd(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i-BLOCK+1)*2]) )));
          h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((i-BLOCK+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
          h1_real = _SIMD_BROADCAST(&hh_dbl[(i-BLOCK+1)*2]);
          h1_imag = _SIMD_BROADCAST(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
         h1_real = _SIMD_SET1(hh_dbl[(i-BLOCK+1)*2]);
         h1_imag = _SIMD_SET1(hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
          // conjugate
          h1_imag = _SIMD_XOR(h1_imag, sign);
#endif

          q1 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+0]);

          tmp1 = _SIMD_MUL(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
          x1 = _SIMD_ADD(x1, _SIMD_FMSUBADD(h1_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          x1 = _SIMD_ADD(x1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
          h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+i)*2]);
          h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+i)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
          h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+i)*2]) )));
          h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+i)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
          h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
          h2_real = _SIMD_SET1(hh_dbl[(ldh+i)*2]);
          h2_imag = _SIMD_SET1(hh_dbl[((ldh+i)*2)+1]);
#endif /* AVX_512 */

#ifndef __ELPA_USE_FMA__
          // conjugate
          h2_imag = _SIMD_XOR(h2_imag, sign);
#endif

          tmp1 = _SIMD_MUL(h2_imag, q1);
#ifdef __ELPA_USE_FMA__
          y1 = _SIMD_ADD(y1, _SIMD_FMSUBADD(h2_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
          y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

     }

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[(nb-1)*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[((nb-1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(nb-1)*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((nb-1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_SET1(hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

#ifndef __ELPA_USE_FMA__
     // conjugate
     h1_imag = _SIMD_XOR(h1_imag, sign);
#endif

     q1 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+0]);

     tmp1 = _SIMD_MUL(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
     x1 = _SIMD_ADD(x1, _SIMD_FMSUBADD(h1_real, q1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     x1 = _SIMD_ADD(x1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, q1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[0]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[0]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1_real = _SIMD_BROADCAST(&hh_dbl[0]);
    h1_imag = _SIMD_BROADCAST(&hh_dbl[1]);
#endif /*  VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
    h1_real = _SIMD_SET1(hh_dbl[0]);
    h1_imag = _SIMD_SET1(hh_dbl[1]);

#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__SIMD_DATATYPE) _SIMD_XOR_EPI((__m512i) h1_imag, (__m512i) sign);
#endif
#endif
#ifdef HAVE_AVX512_XEON
        h1_real = _SIMD_XOR(h1_real, sign);
        h1_imag = _SIMD_XOR(h1_imag, sign);
#endif

#endif /* VEC_SET == AVX_512 */

#if VEC_SET != AVX_512
     h1_real = _SIMD_XOR(h1_real, sign);
     h1_imag = _SIMD_XOR(h1_imag, sign);
#endif /* VEC_SET != AVX_512 */

     tmp1 = _SIMD_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
     x1 = _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
     x1 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

#ifdef BLOCK2

#if VEC_SET == SSE_128       
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[ldh*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[(ldh*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[ldh*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh*2)+1]) )));
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[ldh*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[(ldh*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[ldh*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh*2)+1]) )));
#endif
#endif /* VEC_SET == 128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[ldh*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[(ldh*2)+1]);
     h2_real = _SIMD_BROADCAST(&hh_dbl[ldh*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[(ldh*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[ldh*2]);
     h1_imag = _SIMD_SET1(hh_dbl[(ldh*2)+1]);
     h2_real = _SIMD_SET1(hh_dbl[ldh*2]);
     h2_imag = _SIMD_SET1(hh_dbl[(ldh*2)+1]);

#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h1_real, (__m512i) sign);
     h1_imag = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h1_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
     h1_imag = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h2_real, (__m512i) sign);
     h2_imag = (__SIMD_DATATYPE) _mm512_xor_epi64((__m512i) h2_imag, (__m512i) sign);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h2_real, (__m512i) sign);
     h2_imag = (__SIMD_DATATYPE) _mm512_xor_epi32((__m512i) h2_imag, (__m512i) sign);
#endif
#endif
#ifdef HAVE_AVX512_XEON
     h1_real = _SIMD_XOR(h1_real, sign);
     h1_imag = _SIMD_XOR(h1_imag, sign);
     h2_real = _SIMD_XOR(h2_real, sign);
     h2_imag = _SIMD_XOR(h2_imag, sign);
#endif

#endif /* VEC_SET == AVX_512 */

#if VEC_SET != AVX_512
     h1_real = _SIMD_XOR(h1_real, sign);
     h1_imag = _SIMD_XOR(h1_imag, sign);
     h2_real = _SIMD_XOR(h2_real, sign);
     h2_imag = _SIMD_XOR(h2_imag, sign);
#endif

#if VEC_SET == SSE_128
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = _mm_castpd_ps(_mm_load_pd1((double *) s_dbl));
#else
     tmp2 = _SIMD_LOADU(s_dbl);
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_COMPLEX
     tmp2 = _mm256_set_pd(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = _mm256_set_ps(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0],
                          s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_COMPLEX
     tmp2 = _SIMD_SET(s_dbl[1], s_dbl[0],
                          s_dbl[1], s_dbl[0],
                          s_dbl[1], s_dbl[0],
                          s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     tmp2 = (__SIMD_DATATYPE) _mm512_set1_pd(*(double*)(&s_dbl[0]));
#endif

#endif /* VEC_SET == AVX_512 */

     tmp1 = _SIMD_MUL(h2_imag, tmp2);
#ifdef __ELPA_USE_FMA__
     tmp2 = _SIMD_FMADDSUB(h2_real, tmp2, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
     tmp2 = _SIMD_ADDSUB( _SIMD_MUL(h2_real, tmp2), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

#if VEC_SET == AVX_512
     _SIMD_MASK_STOREU(s_dbl, 0x01 + 0x02, tmp2);

     h2_real = _SIMD_SET1(s_dbl[0]);
     h2_imag = _SIMD_SET1(s_dbl[1]);
#endif

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_movedup_pd(tmp2);
     h2_imag = _mm_set1_pd(tmp2[1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(tmp2);
     h2_imag = _mm_movehdup_ps(tmp2);
#endif
#endif /*  VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_SET1(tmp2[0]);
     h2_imag = _SIMD_SET1(tmp2[1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

     tmp1 = _SIMD_MUL(h1_imag, y1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_FMADDSUB(h1_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
     y1 = _SIMD_ADDSUB( _SIMD_MUL(h1_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

     tmp1 = _SIMD_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
     y1 = _SIMD_ADD(y1, _SIMD_FMADDSUB(h2_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     y1 = _SIMD_ADD(y1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

#endif /* BLOCK2 */

     q1 = _SIMD_LOAD(&q_dbl[0]);

#ifdef BLOCK1
     q1 = _SIMD_ADD(q1, x1);
#endif

#ifdef BLOCK2
     q1 = _SIMD_ADD(q1, y1);
#endif
     _SIMD_STORE(&q_dbl[0], q1);

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
     h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+1)*2]);
     h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+1)*2]) )));
     h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h2_real = _SIMD_SET1(hh_dbl[(ldh+1)*2]);
     h2_imag = _SIMD_SET1(hh_dbl[((ldh+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

     q1 = _SIMD_LOAD(&q_dbl[(ldq*2)+0]);

     q1 = _SIMD_ADD(q1, x1);

     tmp1 = _SIMD_MUL(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h2_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

     _SIMD_STORE(&q_dbl[(ldq*2)+0], q1);

#endif /* BLOCK2 */

     for (i = BLOCK; i < nb; i++)
     {
#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
        h1_real = _mm_loaddup_pd(&hh_dbl[(i-BLOCK+1)*2]);
        h1_imag = _mm_loaddup_pd(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i-BLOCK+1)*2]) )));
        h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((i-BLOCK+1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
	h1_real = _SIMD_BROADCAST(&hh_dbl[(i-BLOCK+1)*2]);
        h1_imag = _SIMD_BROADCAST(&hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
        h1_real = _SIMD_SET1(hh_dbl[(i-BLOCK+1)*2]);
        h1_imag = _SIMD_SET1(hh_dbl[((i-BLOCK+1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

        q1 = _SIMD_LOAD(&q_dbl[(2*i*ldq)+0]);

        tmp1 = _SIMD_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
        q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
        q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

#ifdef BLOCK2

#if VEC_SET == SSE_128
#ifdef DOUBLE_PRECISION_COMPLEX
        h2_real = _mm_loaddup_pd(&hh_dbl[(ldh+i)*2]);
        h2_imag = _mm_loaddup_pd(&hh_dbl[((ldh+i)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
        h2_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(ldh+i)*2]) )));
        h2_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((ldh+i)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
	h2_real = _SIMD_BROADCAST(&hh_dbl[(ldh+i)*2]);
        h2_imag = _SIMD_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
        h2_real = _SIMD_SET1(hh_dbl[(ldh+i)*2]);
        h2_imag = _SIMD_SET1(hh_dbl[((ldh+i)*2)+1]);
#endif /* VEC_SET == AVX_512 */

        tmp1 = _SIMD_MUL(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
        q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h2_real, y1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
        q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h2_real, y1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
#endif /* BLOCK2 */

        _SIMD_STORE(&q_dbl[(2*i*ldq)+0], q1);
    }
#ifdef BLOCK2

#if VEC_SET == SSE_128     
#ifdef DOUBLE_PRECISION_COMPLEX
     h1_real = _mm_loaddup_pd(&hh_dbl[(nb-1)*2]);
     h1_imag = _mm_loaddup_pd(&hh_dbl[((nb-1)*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
     h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(nb-1)*2]) )));
     h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[((nb-1)*2)+1]) )));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h1_real = _SIMD_BROADCAST(&hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_BROADCAST(&hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if VEC_SET == AVX_512
     h1_real = _SIMD_SET1(hh_dbl[(nb-1)*2]);
     h1_imag = _SIMD_SET1(hh_dbl[((nb-1)*2)+1]);
#endif /* VEC_SET == AVX_512 */

     q1 = _SIMD_LOAD(&q_dbl[(2*nb*ldq)+0]);

     tmp1 = _SIMD_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_ADD(q1, _SIMD_FMADDSUB(h1_real, x1, _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
     q1 = _SIMD_ADD(q1, _SIMD_ADDSUB( _SIMD_MUL(h1_real, x1), _SIMD_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

     _SIMD_STORE(&q_dbl[(2*nb*ldq)+0], q1);

#endif /* BLOCK2 */

}
