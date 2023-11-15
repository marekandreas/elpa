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
//
#include "config-f90.h"

#include <stdlib.h>

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
#define SPARC64_SSE 1281
#define VSX_SSE 1282
#define NEON_ARCH64_128 1285
#define SVE_128 1286
#define AVX_256 256
#define AVX2_256 2562
#define SVE_256 2563
#define AVX_512 512
#define SVE_512 5121


#if VEC_SET == SSE_128 || VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == AVX_512
#include <x86intrin.h>
#endif

#if VEC_SET == SPARC64_SSE
#include <fjmfunc.h>
#include <emmintrin.h>
#endif

#if VEC_SET == VSX_SSE
#include <altivec.h>
#endif

#if VEC_SET == NEON_ARCH64_128
#include <arm_neon.h>
#endif

#if VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
#include <arm_sve.h>
#endif

#include <stdio.h>
#include <stdlib.h>

#ifdef BLOCK6
#define PREFIX hexa
#define BLOCK 6
#endif

#ifdef BLOCK4
#define PREFIX quad
#define BLOCK 4
#endif

#ifdef BLOCK2
#define PREFIX double
#define BLOCK 2
#endif

#if VEC_SET == SSE_128
#define SIMD_SET SSE
#endif

#if VEC_SET == SPARC64_SSE
#define SIMD_SET SPARC64
#endif

#if VEC_SET == VSX_SSE
#define SIMD_SET VSX
#endif

#if VEC_SET == NEON_ARCH64_128
#define SIMD_SET NEON_ARCH64
#endif

#if VEC_SET == SVE_128
#define SIMD_SET SVE128
#endif

#if VEC_SET == AVX_256
#define SIMD_SET AVX
#endif

#if VEC_SET == AVX2_256
#define SIMD_SET AVX2
#endif

#if VEC_SET == SVE_256
#define SIMD_SET SVE256
#endif

#if VEC_SET == AVX_512
#define SIMD_SET AVX512
#endif

#if VEC_SET == SVE_512
#define SIMD_SET SVE512
#endif

#ifdef DOUBLE_PRECISION_REAL
#define ONE 1.0
#define MONE -1.0
#endif

#ifdef SINGLE_PRECISION_REAL
#define ONE 1.0f
#define MONE -1.0f
#endif

#define __forceinline __attribute__((always_inline)) static

#if VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE
#define ADDITIONAL_ARGUMENT
#ifdef DOUBLE_PRECISION_REAL
#define offset 2
#define __SIMD_DATATYPE __m128d
#define _SIMD_LOAD _mm_load_pd
#define _SIMD_STORE _mm_store_pd
#define _SIMD_ADD _mm_add_pd
#define _SIMD_MUL _mm_mul_pd
#define _SIMD_SUB _mm_sub_pd
#define _SIMD_XOR _mm_xor_pd
#if VEC_SET == SSE_128
#define _SIMD_SET _mm_set_pd
#define _SIMD_SET1 _mm_set1_pd
#define _SIMD_NEG 1
#endif
#if VEC_SET == SPARC64_SSE
#define _SIMD_NEG _fjsp_neg_v2r8
#endif
#endif /* DOUBLE_PRECISION_REAL */
#ifdef SINGLE_PRECISION_REAL
#define offset 4
#define __SIMD_DATATYPE __m128
#define _SIMD_LOAD _mm_load_ps
#define _SIMD_STORE _mm_store_ps
#define _SIMD_ADD _mm_add_ps
#define _SIMD_MUL _mm_mul_ps
#define _SIMD_SUB _mm_sub_ps
#define _SIMD_XOR _mm_xor_ps
#if VEC_SET == SSE_128
#define _SIMD_SET _mm_set_ps
#define _SIMD_SET1 _mm_set1_ps
#define _SIMD_NEG 1
#endif 
#if VEC_SET == SPARC64_SSE
#define _SIMD_NEG 1
#endif
#endif /* SINGLE_PRECISION_REAL */
#endif /* VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE */

#if VEC_SET == VSX_SSE
#define ADDITIONAL_ARGUMENT

#ifdef DOUBLE_PRECISION_REAL
#define offset 2
#define __SIMD_DATATYPE __vector double
#define _SIMD_LOAD (__vector double) vec_ld
#endif

#ifdef SINGLE_PRECISION_REAL
#define offset 4
#define __SIMD_DATATYPE __vector float
#define _SIMD_LOAD  (__vector float) vec_ld
#endif

#define _SIMD_NEG 1
#define _SIMD_STORE vec_st
#define _SIMD_ADD vec_add
#define _SIMD_MUL vec_mul
#define _SIMD_SUB vec_sub
#define _SIMD_SET1 vec_splats

#endif /*  VEC_SET == VSX_SSE */

#if VEC_SET == NEON_ARCH64_128
#define ADDITIONAL_ARGUMENT
#define __ELPA_USE_FMA__
//#undef  __ELPA_USE_FMA__

#ifdef DOUBLE_PRECISION_REAL
#define offset 2
#define __SIMD_DATATYPE __Float64x2_t
#define _SIMD_LOAD vld1q_f64
#define _SIMD_STORE vst1q_f64
#define _SIMD_ADD vaddq_f64
#define _SIMD_MUL vmulq_f64
#define _SIMD_SUB vsubq_f64
#define _SIMD_NEG vnegq_f64
#define _SIMD_FMA(a, b, c) vfmaq_f64(c ,b, a)
#define _SIMD_NFMA(a, b, c) vfmsq_f64(c, b, a)
#define _SIMD_FMSUB(a, b, c) vnegq_f64(vfmsq_f64(c, b, a))
//#define _SIMD_XOR _mm_xor_pd
#define _SIMD_SET1 vdupq_n_f64
#endif /* DOUBLE_PRECISION_REAL */
#ifdef SINGLE_PRECISION_REAL
#define offset 4
#define __SIMD_DATATYPE __Float32x4_t
#define _SIMD_LOAD vld1q_f32
#define _SIMD_STORE vst1q_f32
#define _SIMD_ADD vaddq_f32
#define _SIMD_MUL vmulq_f32
#define _SIMD_SUB vsubq_f32
#define _SIMD_NEG vnegq_f32
#define _SIMD_FMA(a, b, c) vfmaq_f32(c ,b, a)
#define _SIMD_NFMA(a, b, c) vfmsq_f32(a, b, c)
#define _SIMD_FMSUB(a, b, c) vnegq_f32(vfmsq_f32(c, b, a))
//#define _SIMD_XOR _mm_xor_ps
#define _SIMD_SET1 vdupq_n_f32
#endif /* SINGLE_PRECISION_REAL */
#endif /* NEON_ARCH64_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
#define ADDITIONAL_ARGUMENT
#ifdef DOUBLE_PRECISION_REAL
#define offset 4
#define __SIMD_DATATYPE __m256d
#define _SIMD_LOAD _mm256_load_pd
#define _SIMD_STORE _mm256_store_pd
#define _SIMD_ADD _mm256_add_pd
#define _SIMD_MUL _mm256_mul_pd
#define _SIMD_SUB _mm256_sub_pd
#define _SIMD_SET1 _mm256_set1_pd
#define _SIMD_XOR _mm256_xor_pd
#define _SIMD_BROADCAST _mm256_broadcast_sd
#define _SIMD_NEG 1

#ifdef HAVE_AVX2
#if VEC_SET == AVX2_256
#ifdef __FMA4__
#define __ELPA_USE_FMA__
#define _mm256_FMA_pd(a,b,c) _mm256_macc_pd(a,b,c)
#define _mm256_NFMA_pd(a,b,c) _mm256_nmacc_pd(a,b,c)
#error "This should be prop _mm256_msub_pd instead of _mm256_msub"
#define _mm256_FMSUB_pd(a,b,c) _mm256_msub(a,b,c)
#endif /* __FMA4__ */

#ifdef __AVX2__
#define __ELPA_USE_FMA__
#define _mm256_FMA_pd(a,b,c) _mm256_fmadd_pd(a,b,c)
#define _mm256_NFMA_pd(a,b,c) _mm256_fnmadd_pd(a,b,c)
#define _mm256_FMSUB_pd(a,b,c) _mm256_fmsub_pd(a,b,c)
#endif /* __AVX2__ */
#ifdef __ELPA_USE_FMA__
#define _SIMD_FMA _mm256_FMA_pd
#define _SIMD_NFMA _mm256_NFMA_pd
#define _SIMD_FMSUB _mm256_FMSUB_pd
#endif
#endif /* VEC_SET == AVX2_256 */
#endif /* HAVE_AVX2 */
#endif /* DOUBLE_PRECISION_REAL */

#ifdef SINGLE_PRECISION_REAL
#define offset 8
#define __SIMD_DATATYPE __m256
#define _SIMD_LOAD _mm256_load_ps
#define _SIMD_STORE _mm256_store_ps
#define _SIMD_ADD _mm256_add_ps
#define _SIMD_MUL _mm256_mul_ps
#define _SIMD_SUB _mm256_sub_ps
#define _SIMD_SET1 _mm256_set1_ps
#define _SIMD_XOR _mm256_xor_ps
#define _SIMD_BROADCAST _mm256_broadcast_ss
#define _SIMD_NEG 1

#ifdef HAVE_AVX2
#if VEC_SET == AVX2_256
#ifdef __FMA4__
#define __ELPA_USE_FMA__
#define _mm256_FMA_ps(a,b,c) _mm256_macc_ps(a,b,c)
#define _mm256_NFMA_ps(a,b,c) _mm256_nmacc_ps(a,b,c)
#error "This should be prop _mm256_msub_ps instead of _mm256_msub"
#define _mm256_FMSUB_ps(a,b,c) _mm256_msub(a,b,c)
#endif /* __FMA4__ */
#ifdef __AVX2__
#define __ELPA_USE_FMA__
#define _mm256_FMA_ps(a,b,c) _mm256_fmadd_ps(a,b,c)
#define _mm256_NFMA_ps(a,b,c) _mm256_fnmadd_ps(a,b,c)
#define _mm256_FMSUB_ps(a,b,c) _mm256_fmsub_ps(a,b,c)
#endif /* __AVX2__ */
#ifdef __ELPA_USE_FMA__
#define _SIMD_FMA _mm256_FMA_ps
#define _SIMD_NFMA _mm256_NFMA_ps
#define _SIMD_FMSUB _mm256_FMSUB_ps
#endif
#endif /* VEC_SET == AVX2_256 */
#endif /* HAVE_AVX2 */
#endif /* SINGLE_PRECISION_REAL */
#endif /* VEC_SET == AVX_256 */

#if VEC_SET == AVX_512
#define ADDITIONAL_ARGUMENT
#ifdef DOUBLE_PRECISION_REAL
#define offset 8
#define __SIMD_DATATYPE __m512d
#define __SIMD_INTEGER  __m512i
#define _SIMD_LOAD _mm512_load_pd
#define _SIMD_STORE _mm512_store_pd
#define _SIMD_ADD _mm512_add_pd
#define _SIMD_MUL _mm512_mul_pd
#define _SIMD_SUB _mm512_sub_pd
#define _SIMD_SET1 _mm512_set1_pd
#define _SIMD_NEG 1
#ifdef HAVE_AVX512_XEON
#define _SIMD_XOR _mm512_xor_pd
#endif
#ifdef HAVE_AVX512
#define __ELPA_USE_FMA__
#define _mm512_FMA_pd(a,b,c) _mm512_fmadd_pd(a,b,c)
#define _mm512_NFMA_pd(a,b,c) _mm512_fnmadd_pd(a,b,c)
#define _mm512_FMSUB_pd(a,b,c) _mm512_fmsub_pd(a,b,c)
#ifdef __ELPA_USE_FMA__
#define _SIMD_FMA _mm512_FMA_pd
#define _SIMD_NFMA _mm512_NFMA_pd
#define _SIMD_FMSUB _mm512_FMSUB_pd
#endif
#endif /* HAVE_AVX512 */
#endif /* DOUBLE_PRECISION_REAL */

#ifdef SINGLE_PRECISION_REAL
#define offset 16
#define __SIMD_DATATYPE __m512
#define __SIMD_INTEGER  __m512i
#define _SIMD_LOAD _mm512_load_ps
#define _SIMD_STORE _mm512_store_ps
#define _SIMD_ADD _mm512_add_ps
#define _SIMD_MUL _mm512_mul_ps
#define _SIMD_SUB _mm512_sub_ps
#define _SIMD_SET1 _mm512_set1_ps
#define _SIMD_NEG 1
#ifdef HAVE_AVX512_XEON
#define _SIMD_XOR _mm512_xor_ps
#endif
#ifdef HAVE_AVX512
#define __ELPA_USE_FMA__
#define _mm512_FMA_ps(a,b,c) _mm512_fmadd_ps(a,b,c)
#define _mm512_NFMA_ps(a,b,c) _mm512_fnmadd_ps(a,b,c)
#define _mm512_FMSUB_ps(a,b,c) _mm512_fmsub_ps(a,b,c)
#ifdef __ELPA_USE_FMA__
#define _SIMD_FMA _mm512_FMA_ps
#define _SIMD_NFMA _mm512_NFMA_ps
#define _SIMD_FMSUB _mm512_FMSUB_ps
#endif
#endif /* HAVE_AVX512 */
#endif /* SINGLE_PRECISION_REAL */
#endif /* VEC_SET == AVX_512 */



#if VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
#define __ELPA_USE_FMA__
#ifdef DOUBLE_PRECISION_REAL
#define ADDITIONAL_ARGUMENT svptrue_b64(),
#if VEC_SET == SVE_512
#define offset 8
#endif
#if VEC_SET == SVE_256
#define offset 4
#endif
#if VEC_SET == SVE_128
#define offset 2
#endif
#define __SIMD_DATATYPE svfloat64_t
#define _SIMD_LOAD svld1_f64
#define _SIMD_STORE svst1_f64
#define _SIMD_ADD svadd_f64_z
#define _SIMD_MUL svmul_f64_z
#define _SIMD_SUB svsub_f64_z
#define _SIMD_NEG svneg_f64_z
#define _SIMD_FMA(a, b, c) svmad_f64_z(svptrue_b64(), a, b, c)
//#define _SIMD_NFMA(a, b, c) svneg_f64_z(svptrue_b64(), svmad_f64_z(svptrue_b64(), a, b, c))
#define _SIMD_NFMA(a, b, c) svmsb_f64_z(svptrue_b64(), a, b, c)
#define _SIMD_FMSUB(a, b, c) svneg_f64_z(svptrue_b64(), svmsb_f64_z(svptrue_b64(), a, b, c))
//#define _SIMD_XOR _mm_xor_pd
#define _SIMD_SET1 svdup_f64
#endif /* DOUBLE_PRECISION_REAL */
#ifdef SINGLE_PRECISION_REAL
#define ADDITIONAL_ARGUMENT svptrue_b32(),
#if VEC_SET == SVE_512
#define offset 16
#endif
#if VEC_SET == SVE_256
#define offset 8
#endif
#if VEC_SET == SVE_128
#define offset 4
#endif
#define __SIMD_DATATYPE svfloat32_t
#define _SIMD_LOAD svld1_f32
#define _SIMD_STORE svst1_f32
#define _SIMD_ADD svadd_f32_z
#define _SIMD_MUL svmul_f32_z
#define _SIMD_SUB svsub_f32_z
#define _SIMD_NEG svneg_f32_z
#define _SIMD_FMA(a, b, c) svmad_f32_z(svptrue_b32(), a, b, c)
//#define _SIMD_NFMA(a, b, c) svneg_f32_z(svptrue_b32(), svmad_f32_z(svptrue_b32(), a, b, c))
#define _SIMD_NFMA(a, b, c) svmsb_f32_z(svptrue_b32(), a, b, c)
#define _SIMD_FMSUB(a, b, c) svneg_f32_z(svptrue_b32(), svmsb_f32_z(svptrue_b32(), a, b, c))
//#define _SIMD_XOR _mm_xor_ps
#define _SIMD_SET1 svdup_f32
#endif /* SINGLE_PRECISION_REAL */
#endif /* SVE_512 */



#ifdef DOUBLE_PRECISION_REAL
#define WORD_LENGTH double
#define DATA_TYPE double
#define DATA_TYPE_PTR double*
#endif
#ifdef SINGLE_PRECISION_REAL
#define WORD_LENGTH single
#define DATA_TYPE float
#define DATA_TYPE_PTR float*
#endif

#if VEC_SET == SSE_128
#undef __AVX__
#endif

#if VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == AVX_512
#undef _LOAD
#undef _STORE
#undef _XOR
#define _LOAD(x) _SIMD_LOAD(x)
#define _STORE(a, b) _SIMD_STORE(a, b)
#define _XOR(a, b) _SIMD_XOR(a, b)
#endif

#if VEC_SET == VSX_SSE
#undef _LOAD
#undef _STORE
#undef _XOR
#define _LOAD(x) _SIMD_LOAD(0, (unsigned long int *) x)
#define _STORE(a, b) _SIMD_STORE((__vector unsigned int) b, 0, (unsigned int *) a)
#define _XOR(a, b) vec_mul(b, a)
#endif

#if VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
#undef _LOAD
#undef _STORE
#undef _XOR
#ifdef DOUBLE_PRECISION_REAL
#define _LOAD(x) _SIMD_LOAD(svptrue_b64(), x)
#define _STORE(a, b) _SIMD_STORE(svptrue_b64(), a, b)
#endif
#ifdef SINGLE_PRECISION_REAL
#define _LOAD(x) _SIMD_LOAD(svptrue_b32(), x)
#define _STORE(a, b) _SIMD_STORE(svptrue_b32(), a, b)
#endif
//#define _XOR(a, b) _SIMD_XOR(a, b)
#endif

#if VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE ||  VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128
//Forward declaration
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 2
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 4
#endif
#endif /* VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 8
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 16
#endif
#endif /* VEC_SET == AVX_512 || SVE_512 */
__forceinline void CONCAT_8ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq, int ldh, 
#ifdef BLOCK2
	DATA_TYPE s);
#endif
#ifdef BLOCK4
	DATA_TYPE s_1_2, DATA_TYPE s_1_3, DATA_TYPE s_2_3, DATA_TYPE s_1_4, DATA_TYPE s_2_4, DATA_TYPE s_3_4);
#endif
#ifdef BLOCK6
	DATA_TYPE_PTR scalarprods);
#endif

#if VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 8
#endif
#endif /* VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 16
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 16
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 32
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */
__forceinline void CONCAT_8ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq, int ldh, 
#ifdef BLOCK2
	DATA_TYPE s);
#endif
#ifdef BLOCK4
	DATA_TYPE s_1_2, DATA_TYPE s_1_3, DATA_TYPE s_2_3, DATA_TYPE s_1_4, DATA_TYPE s_2_4, DATA_TYPE s_3_4);
#endif
#ifdef BLOCK6
	DATA_TYPE_PTR scalarprods);
#endif

#if VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 6
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 12
#endif
#endif /* VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_256 */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 12
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 24
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 24
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 48
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */

__forceinline void CONCAT_8ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq, int ldh,
#ifdef BLOCK2
	DATA_TYPE s);
#endif
#ifdef BLOCK4
	DATA_TYPE s_1_2, DATA_TYPE s_1_3, DATA_TYPE s_2_3, DATA_TYPE s_1_4, DATA_TYPE s_2_4, DATA_TYPE s_3_4);
#endif
#ifdef BLOCK6
	DATA_TYPE_PTR scalarprods);
#endif

#if VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 16
#endif
#endif /* VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_256  */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 16
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 32
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 32
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 64
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */

__forceinline void CONCAT_8ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq, int ldh, 
#ifdef BLOCK2
	DATA_TYPE s);
#endif
#ifdef BLOCK4
	DATA_TYPE s_1_2, DATA_TYPE s_1_3, DATA_TYPE s_2_3, DATA_TYPE s_1_4, DATA_TYPE s_2_4, DATA_TYPE s_3_4);
#endif
#ifdef BLOCK6
	DATA_TYPE_PTR scalarprods);
#endif

#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 10
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 20
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 20
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 40
#endif
#endif /*  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 40
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 80
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */

__forceinline void CONCAT_8ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq, int ldh, 
#ifdef BLOCK2
	DATA_TYPE s);
#endif
#ifdef BLOCK4
	DATA_TYPE s_1_2, DATA_TYPE s_1_3, DATA_TYPE s_2_3, DATA_TYPE s_1_4, DATA_TYPE s_2_4, DATA_TYPE s_3_4);
#endif
#ifdef BLOCK6
	DATA_TYPE_PTR scalarprods);
#endif

#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 12
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 24
#endif
#endif /* VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 24
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 48
#endif
#endif /*  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 48
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 96
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */

__forceinline void CONCAT_8ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq, int ldh,
#ifdef BLOCK2
	DATA_TYPE s);
#endif
#ifdef BLOCK4
	DATA_TYPE s_1_2, DATA_TYPE s_1_3, DATA_TYPE s_2_3, DATA_TYPE s_1_4, DATA_TYPE s_2_4, DATA_TYPE s_3_4);
#endif
#ifdef BLOCK6
	DATA_TYPE_PTR scalarprods);
#endif

void CONCAT_7ARGS(PREFIX,_hh_trafo_real_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int* pnb, int* pnq, int* pldq, int* pldh);

/*
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine double_hh_trafo_real_SSE_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="double_hh_trafo_real_SSE_2hv_double")
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
!f>   subroutine double_hh_trafo_real_SSE_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>              bind(C, name="double_hh_trafo_real_SSE_2hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int) :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value  :: q
!f>     real(kind=c_float)  :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_SPARC64_SSE
!f> interface
!f>   subroutine double_hh_trafo_real_SPARC64_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>              bind(C, name="double_hh_trafo_real_SPARC64_2hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int) :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value  :: q
!f>     real(kind=c_double) :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_SPARC64_SSE
!f> interface
!f>   subroutine double_hh_trafo_real_SPARC64_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>              bind(C, name="double_hh_trafo_real_SPARC64_2hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int) :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value  :: q
!f>     real(kind=c_float)  :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_NEON_ARCH64_SSE
!f> interface
!f>   subroutine double_hh_trafo_real_NEON_ARCH64_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>              bind(C, name="double_hh_trafo_real_NEON_ARCH64_2hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int) :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value  :: q
!f>     real(kind=c_double) :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_NEON_ARCH64_SSE
!f> interface
!f>   subroutine double_hh_trafo_real_NEON_ARCH64_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>              bind(C, name="double_hh_trafo_real_NEON_ARCH64_2hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int) :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value  :: q
!f>     real(kind=c_float)  :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/


/*
!f>#ifdef HAVE_VSX_SSE
!f> interface
!f>   subroutine double_hh_trafo_real_VSX_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="double_hh_trafo_real_VSX_2hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int) :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value  :: q
!f>        real(kind=c_double) :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_VSX_SSE
!f> interface
!f>   subroutine double_hh_trafo_real_VSX_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="double_hh_trafo_real_VSX_2hv_single")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_float)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/


/*
!f>#ifdef HAVE_SVE128
!f> interface
!f>   subroutine double_hh_trafo_real_SVE128_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="double_hh_trafo_real_SVE128_2hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int) :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value  :: q
!f>        real(kind=c_double) :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_SVE128
!f> interface
!f>   subroutine double_hh_trafo_real_SVE128_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="double_hh_trafo_real_SVE128_2hv_single")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_float)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/


/*
!f>#if defined(HAVE_AVX) 
!f> interface
!f>   subroutine double_hh_trafo_real_AVX_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="double_hh_trafo_real_AVX_2hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_double)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_AVX)
!f> interface
!f>   subroutine double_hh_trafo_real_AVX_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="double_hh_trafo_real_AVX_2hv_single")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)       :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_float)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_AVX2) 
!f> interface
!f>   subroutine double_hh_trafo_real_AVX2_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="double_hh_trafo_real_AVX2_2hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_double)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_AVX2)
!f> interface
!f>   subroutine double_hh_trafo_real_AVX2_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="double_hh_trafo_real_AVX2_2hv_single")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)       :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_float)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/


/*
!f>#if defined(HAVE_SVE256) 
!f> interface
!f>   subroutine double_hh_trafo_real_SVE256_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="double_hh_trafo_real_SVE256_2hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_double)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_SVE256)
!f> interface
!f>   subroutine double_hh_trafo_real_SVE256_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="double_hh_trafo_real_SVE256_2hv_single")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)       :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_float)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine double_hh_trafo_real_AVX512_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="double_hh_trafo_real_AVX512_2hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_double)     :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#if defined(HAVE_SVE512)
!f> interface
!f>   subroutine double_hh_trafo_real_SVE512_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="double_hh_trafo_real_SVE512_2hv_double")
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
!f>   subroutine double_hh_trafo_real_AVX512_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="double_hh_trafo_real_AVX512_2hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)      :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#if defined(HAVE_SVE512)
!f> interface
!f>   subroutine double_hh_trafo_real_SVE512_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="double_hh_trafo_real_SVE512_2hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)      :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine quad_hh_trafo_real_SSE_4hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="quad_hh_trafo_real_SSE_4hv_double")
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
!f>   subroutine quad_hh_trafo_real_SSE_4hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>              bind(C, name="quad_hh_trafo_real_SSE_4hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int) :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value  :: q
!f>     real(kind=c_float)  :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_SPARC64_SSE
!f> interface
!f>   subroutine quad_hh_trafo_real_SPARC64_4hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>              bind(C, name="quad_hh_trafo_real_SPARC64_4hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int) :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value  :: q
!f>     real(kind=c_double) :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_SPARC64_SSE
!f> interface
!f>   subroutine quad_hh_trafo_real_SPARC64_4hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>              bind(C, name="quad_hh_trafo_real_SPARC64_4hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int) :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value  :: q
!f>     real(kind=c_float)  :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_NEON_ARCH64_SSE
!f> interface
!f>   subroutine quad_hh_trafo_real_NEON_ARCH64_4hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>              bind(C, name="quad_hh_trafo_real_NEON_ARCH64_4hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int) :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value  :: q
!f>     real(kind=c_double) :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_NEON_ARCH64_SSE
!f> interface
!f>   subroutine quad_hh_trafo_real_NEON_ARCH64_4hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>              bind(C, name="quad_hh_trafo_real_NEON_ARCH64_4hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int) :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value  :: q
!f>     real(kind=c_float)  :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_VSX_SSE
!f> interface
!f>   subroutine quad_hh_trafo_real_VSX_4hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="quad_hh_trafo_real_VSX_4hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_double)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_VSX_SSE
!f> interface
!f>   subroutine quad_hh_trafo_real_VSX_4hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="quad_hh_trafo_real_VSX_4hv_single")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_float)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_SVE128
!f> interface
!f>   subroutine quad_hh_trafo_real_SVE128_4hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="quad_hh_trafo_real_SVE128_4hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_double)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_SVE128
!f> interface
!f>   subroutine quad_hh_trafo_real_SVE128_4hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="quad_hh_trafo_real_SVE128_4hv_single")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_float)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_AVX)
!f> interface
!f>   subroutine quad_hh_trafo_real_AVX_4hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="quad_hh_trafo_real_AVX_4hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_double)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#if defined(HAVE_AVX)
!f> interface
!f>   subroutine quad_hh_trafo_real_AVX_4hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>              bind(C, name="quad_hh_trafo_real_AVX_4hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int) :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value  :: q
!f>     real(kind=c_float)  :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/


/*
!f>#if defined(HAVE_AVX2)
!f> interface
!f>   subroutine quad_hh_trafo_real_AVX2_4hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="quad_hh_trafo_real_AVX2_4hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_double)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#if defined(HAVE_AVX2)
!f> interface
!f>   subroutine quad_hh_trafo_real_AVX2_4hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>              bind(C, name="quad_hh_trafo_real_AVX2_4hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int) :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value  :: q
!f>     real(kind=c_float)  :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_SVE256)
!f> interface
!f>   subroutine quad_hh_trafo_real_SVE256_4hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="quad_hh_trafo_real_SVE256_4hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_double)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#if defined(HAVE_SVE256)
!f> interface
!f>   subroutine quad_hh_trafo_real_SVE256_4hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>              bind(C, name="quad_hh_trafo_real_SVE256_4hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int) :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value  :: q
!f>     real(kind=c_float)  :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine quad_hh_trafo_real_AVX512_4hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="quad_hh_trafo_real_AVX512_4hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_double)     :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#if defined(HAVE_SVE512)
!f> interface
!f>   subroutine quad_hh_trafo_real_SVE512_4hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="quad_hh_trafo_real_SVE512_4hv_double")
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
!f>   subroutine quad_hh_trafo_real_AVX512_4hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="quad_hh_trafo_real_AVX512_4hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)      :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#if defined(HAVE_SVE512)
!f> interface
!f>   subroutine quad_hh_trafo_real_SVE512_4hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="quad_hh_trafo_real_SVE512_4hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)      :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine hexa_hh_trafo_real_SSE_6hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="hexa_hh_trafo_real_SSE_6hv_double")
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
!f>   subroutine hexa_hh_trafo_real_SPARC64_6hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="hexa_hh_trafo_real_SPARC64_6hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_double)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_NEON_ARCH64_SSE
!f> interface
!f>   subroutine hexa_hh_trafo_real_NEON_ARCH64_6hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="hexa_hh_trafo_real_NEON_ARCH64_6hv_double")
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
!f>   subroutine hexa_hh_trafo_real_SSE_6hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="hexa_hh_trafo_real_SSE_6hv_single")
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
!f>   subroutine hexa_hh_trafo_real_SPARC64_6hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="hexa_hh_trafo_real_SPARC64_6hv_single")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_float)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_NEON_ARCH64_SSE
!f> interface
!f>   subroutine hexa_hh_trafo_real_NEON_ARCH64_6hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="hexa_hh_trafo_real_NEON_ARCH64_6hv_single")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_float)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#ifdef HAVE_VSX_SSE
!f> interface
!f>   subroutine hexa_hh_trafo_real_VSX_6hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="hexa_hh_trafo_real_VSX_6hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_double)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#ifdef HAVE_VSX_SSE
!f> interface
!f>   subroutine hexa_hh_trafo_real_VSX_6hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="hexa_hh_trafo_real_VSX_6hv_single")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_float)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#ifdef HAVE_SVE128
!f> interface
!f>   subroutine hexa_hh_trafo_real_SVE128_6hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="hexa_hh_trafo_real_SVE128_6hv_double")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_double)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#ifdef HAVE_SVE128
!f> interface
!f>   subroutine hexa_hh_trafo_real_SVE128_6hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="hexa_hh_trafo_real_SVE128_6hv_single")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_float)        :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/


/*
!f>#if defined(HAVE_AVX)
!f> interface
!f>   subroutine hexa_hh_trafo_real_AVX_6hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="hexa_hh_trafo_real_AVX_6hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_double)     :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#if defined(HAVE_AVX)
!f> interface
!f>   subroutine hexa_hh_trafo_real_AVX_6hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="hexa_hh_trafo_real_AVX_6hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)      :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_AVX2)
!f> interface
!f>   subroutine hexa_hh_trafo_real_AVX2_6hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="hexa_hh_trafo_real_AVX2_6hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_double)     :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#if defined(HAVE_AVX2)
!f> interface
!f>   subroutine hexa_hh_trafo_real_AVX2_6hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="hexa_hh_trafo_real_AVX2_6hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)      :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

/*
!f>#if defined(HAVE_SVE256)
!f> interface
!f>   subroutine hexa_hh_trafo_real_SVE256_6hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="hexa_hh_trafo_real_SVE256_6hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_double)     :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#if defined(HAVE_SVE256)
!f> interface
!f>   subroutine hexa_hh_trafo_real_SVE256_6hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="hexa_hh_trafo_real_SVE256_6hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)      :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine hexa_hh_trafo_real_AVX512_6hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="hexa_hh_trafo_real_AVX512_6hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_double)     :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#if defined(HAVE_SVE512)
!f> interface
!f>   subroutine hexa_hh_trafo_real_SVE512_6hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="hexa_hh_trafo_real_SVE512_6hv_double")
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
!f>   subroutine hexa_hh_trafo_real_AVX512_6hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="hexa_hh_trafo_real_AVX512_6hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)      :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
/*
!f>#if defined(HAVE_SVE512)
!f> interface
!f>   subroutine hexa_hh_trafo_real_SVE512_6hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="hexa_hh_trafo_real_SVE512_6hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)      :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

void CONCAT_7ARGS(PREFIX,_hh_trafo_real_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int* pnb, int* pnq, int* pldq, int* pldh)
{
  int i;
  int nb = *pnb;
  int nq = *pldq;
  int ldq = *pldq;
  int ldh = *pldh;
  int worked_on;

  worked_on = 0;

#if VEC_SET == SVE_512
  double br;
  uint64_t length;

  svfloat64_t a_sve;
  br=10.;
  a_sve = svdup_f64(br);
  length = svlen_f64(a_sve);
  if (length != 8) {
    printf("Vector length is %d instead of 8\n",length);
    abort();
  }
#endif

#if VEC_SET == SVE_256
  double br;
  uint64_t length;

  svfloat64_t a_sve;
  br=10.;
  a_sve = svdup_f64(br);
  length = svlen_f64(a_sve);
  if (length != 4) {
    printf("Vector length is %d instead of 4\n",length);
    abort();
  }
#endif

#if VEC_SET == SVE_128
  double br;
  uint64_t length;

  svfloat64_t a_sve;
  br=10.;
  a_sve = svdup_f64(br);
  length = svlen_f64(a_sve);
  if (length != 2) {
    printf("Vector length is %d instead of 2\n",length);
    abort();
  }
#endif

#ifdef BLOCK2
  // calculating scalar product to compute
  // 2 householder vectors simultaneously
  DATA_TYPE s = hh[(ldh)+1]*ONE;
#endif

#ifdef BLOCK4
  // calculating scalar products to compute
  // 4 householder vectors simultaneously
  DATA_TYPE s_1_2 = hh[(ldh)+1];  
  DATA_TYPE s_1_3 = hh[(ldh*2)+2];
  DATA_TYPE s_2_3 = hh[(ldh*2)+1];
  DATA_TYPE s_1_4 = hh[(ldh*3)+3];
  DATA_TYPE s_2_4 = hh[(ldh*3)+2];
  DATA_TYPE s_3_4 = hh[(ldh*3)+1];

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
#endif /* BLOCK4 */

#ifdef BLOCK6
  // calculating scalar products to compute
  // 6 householder vectors simultaneously
  DATA_TYPE scalarprods[15];

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


#endif /* BLOCK6 */

#if VEC_SET == SSE_128 || VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == AVX_512
  #pragma ivdep
#endif
  for (i = BLOCK; i < nb; i++)
    {
#ifdef BLOCK2
      s += hh[i-1] * hh[(i+ldh)];
#endif

#ifdef BLOCK4
      s_1_2 += hh[i-1] * hh[(i+ldh)];           
      s_2_3 += hh[(ldh)+i-1] * hh[i+(ldh*2)];   
      s_3_4 += hh[(ldh*2)+i-1] * hh[i+(ldh*3)]; 

      s_1_3 += hh[i-2] * hh[i+(ldh*2)];         
      s_2_4 += hh[(ldh*1)+i-2] * hh[i+(ldh*3)]; 

      s_1_4 += hh[i-3] * hh[i+(ldh*3)];         
#endif /* BLOCK4 */

#ifdef BLOCK6
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
#endif /* BLOCK6 */

    }

  // Production level kernel calls with padding
#ifdef BLOCK2

#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE  || VEC_SET == NEON_ARCH64_128
#ifdef DOUBLE_PRECISION_REAL
#define STEP_SIZE 12
#define ROW_LENGTH 12
#define UPPER_BOUND 10
#endif
#ifdef SINGLE_PRECISION_REAL
#define STEP_SIZE 24
#define ROW_LENGTH 24
#define UPPER_BOUND 20
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#define STEP_SIZE 24
#define ROW_LENGTH 24
#define UPPER_BOUND 20
#endif
#ifdef SINGLE_PRECISION_REAL
#define STEP_SIZE 48
#define ROW_LENGTH 48
#define UPPER_BOUND 40
#endif
#endif /*  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define STEP_SIZE 32
#define ROW_LENGTH 32
#define UPPER_BOUND 24
#endif
#ifdef SINGLE_PRECISION_REAL
#define STEP_SIZE 64
#define ROW_LENGTH 64
#define UPPER_BOUND 48
#endif
#endif /*  VEC_SET == AVX_512 || VEC_SET == SVE_512 */


  for (i = 0; i < nq - UPPER_BOUND; i+= STEP_SIZE )
    {
      CONCAT_6ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
      worked_on += ROW_LENGTH;
    }

  if (nq == i)
    {
      return;
    }

#undef ROW_LENGTH
#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 10
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 20
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE  || VEC_SET == NEON_ARCH64_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 20
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 40
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 24
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 48
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */

  if (nq-i == ROW_LENGTH)
    {
      CONCAT_6ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
      worked_on += ROW_LENGTH;
    }

#undef ROW_LENGTH
#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 16
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE  || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 16
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 32
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 16
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 32
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */


  if (nq-i == ROW_LENGTH)
    {
      CONCAT_6ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
      worked_on += ROW_LENGTH;
    }

#undef ROW_LENGTH
#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 6
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 12
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 12
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 24
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 16
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */
  if (nq-i == ROW_LENGTH)
    {
      CONCAT_6ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
      worked_on += ROW_LENGTH;
    }

#if VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256

#undef ROW_LENGTH
#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 8
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 16
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */


  if (nq-i == ROW_LENGTH)
    {
      CONCAT_6ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
      worked_on += ROW_LENGTH;
    }

#undef ROW_LENGTH
#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 2
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 4
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE  || VEC_SET == NEON_ARCH64_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 8
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

  if (nq-i == ROW_LENGTH)
    {
      CONCAT_6ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
      worked_on += ROW_LENGTH;
    }

#endif /* VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE  || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#endif /* BLOCK2 */

#ifdef BLOCK4


#undef ROW_LENGTH
#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 6
#define STEP_SIZE 6
#define UPPER_BOUND 4
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 12
#define STEP_SIZE 12
#define UPPER_BOUND 8
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 12
#define STEP_SIZE 12
#define UPPER_BOUND 8
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 24
#define STEP_SIZE 24
#define UPPER_BOUND 16
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 32
#define STEP_SIZE 32
#define UPPER_BOUND 24
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 64
#define STEP_SIZE 64
#define UPPER_BOUND 48
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */
  for (i = 0; i < nq - UPPER_BOUND; i+= STEP_SIZE )
    {
      CONCAT_6ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_4hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
      worked_on += ROW_LENGTH;
    }

  if (nq == i)
    {
      return;
    }


#undef ROW_LENGTH
#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 8
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 16
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 24
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 48
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */


  if (nq-i == ROW_LENGTH )
    {
      CONCAT_6ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_4hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
      worked_on += ROW_LENGTH;
    }

#undef ROW_LENGTH
#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 2
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 4
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 8
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 16
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 32
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */

   if (nq-i == ROW_LENGTH )
     {
       CONCAT_6ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_4hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
       worked_on += ROW_LENGTH;
     }

#if VEC_SET == AVX_512 || VEC_SET == SVE_512

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 16
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */

   if (nq-i == ROW_LENGTH )
     {
       CONCAT_6ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_4hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
       worked_on += ROW_LENGTH;
     }

#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */

#endif /* BLOCK4 */

#ifdef BLOCK6

#undef ROW_LENGTH
#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 4
#define STEP_SIZE 4
#define UPPER_BOUND 2
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 8
#define STEP_SIZE 8
#define UPPER_BOUND 4
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE  || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 8
#define STEP_SIZE 8
#define UPPER_BOUND 4
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 16
#define STEP_SIZE 16
#define UPPER_BOUND 8
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 32
#define STEP_SIZE 32
#define UPPER_BOUND 24
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 64
#define STEP_SIZE 64
#define UPPER_BOUND 48
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */

  for (i = 0; i < nq - UPPER_BOUND; i+= STEP_SIZE)
    { 
      CONCAT_6ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_6hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, scalarprods);
      worked_on += ROW_LENGTH;
    }
    if (nq == i)
      {
        return;
      }

#undef ROW_LENGTH
#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 2
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 4
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE  || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 8
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 24
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 48
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */


    if (nq -i == ROW_LENGTH )
      {
        CONCAT_6ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_6hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, scalarprods);
        worked_on += ROW_LENGTH;
      }
#if VEC_SET == AVX_512 || VEC_SET == SVE_512

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 16
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 32
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */


    if (nq -i == ROW_LENGTH )
      {
        CONCAT_6ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_6hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, scalarprods);
        worked_on += ROW_LENGTH;
      }

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 16
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */


    if (nq -i == ROW_LENGTH )
      {
        CONCAT_6ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_6hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, scalarprods);
        worked_on += ROW_LENGTH;
      }
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */

#endif /* BLOCK6 */

#ifdef WITH_DEBUG
  if (worked_on != nq)
    {
      printf("Error in real SIMD_SET BLOCK BLOCK kernel %d %d\n", worked_on, nq);
      abort();
    }
#endif
}

#undef ROW_LENGTH
#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 12
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 24
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 24
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 48
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 48
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 96
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */

/*
 * Unrolled kernel that computes
 * ROW_LENGTH rows of Q simultaneously, a
 * matrix Vector product with two householder
 */
#ifdef BLOCK2
/*
 * vectors + a rank 2 update is performed
 */
#endif
#ifdef BLOCK4
/*
 * vectors + a rank 1 update is performed
 */
#endif
__forceinline void CONCAT_8ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq, int ldh,
#ifdef BLOCK2
               DATA_TYPE s)
#endif
#ifdef BLOCK4
               DATA_TYPE s_1_2, DATA_TYPE s_1_3, DATA_TYPE s_2_3, DATA_TYPE s_1_4, DATA_TYPE s_2_4, DATA_TYPE s_3_4)
#endif 
#ifdef BLOCK6
               DATA_TYPE_PTR scalarprods)
#endif
  {
#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Matrix Vector Multiplication, Q [10 x nb+1] * hh
    // hh contains two householder vectors, with offset 1
    /////////////////////////////////////////////////////
#endif
#if defined(BLOCK4) || defined(BLOCK6)
    /////////////////////////////////////////////////////
    // Matrix Vector Multiplication, Q [10 x nb+3] * hh
    // hh contains four householder vectors
    /////////////////////////////////////////////////////
#endif

    int i;

#ifdef BLOCK2
#if VEC_SET == SSE_128
    // Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == VSX_SSE
    __SIMD_DATATYPE sign = vec_splats(MONE);
#endif

#if VEC_SET == NEON_ARCH64_128
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = vdupq_n_f64(MONE);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = vdupq_n_f32(MONE);
#endif
#endif
    
#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set1_epi64x(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if  VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_512 */

#if VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = svdup_f64(MONE);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = svdup_f32(MONE);
#endif
#endif

    __SIMD_DATATYPE x1 = _LOAD(&q[ldq]);
    __SIMD_DATATYPE x2 = _LOAD(&q[ldq+offset]);
    __SIMD_DATATYPE x3 = _LOAD(&q[ldq+2*offset]);
    __SIMD_DATATYPE x4 = _LOAD(&q[ldq+3*offset]);
    __SIMD_DATATYPE x5 = _LOAD(&q[ldq+4*offset]);
    __SIMD_DATATYPE x6 = _LOAD(&q[ldq+5*offset]);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE  || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h1 = _SIMD_SET1(hh[ldh+1]);
#endif
#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h1 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif
 
    __SIMD_DATATYPE h2;
#ifdef __ELPA_USE_FMA__
    __SIMD_DATATYPE q1 = _LOAD(q);
    __SIMD_DATATYPE y1 = _SIMD_FMA(x1, h1, q1);
    __SIMD_DATATYPE q2 = _LOAD(&q[offset]);
    __SIMD_DATATYPE y2 = _SIMD_FMA(x2, h1, q2);
    __SIMD_DATATYPE q3 = _LOAD(&q[2*offset]);
    __SIMD_DATATYPE y3 = _SIMD_FMA(x3, h1, q3);
    __SIMD_DATATYPE q4 = _LOAD(&q[3*offset]);
    __SIMD_DATATYPE y4 = _SIMD_FMA(x4, h1, q4);
    __SIMD_DATATYPE q5 = _LOAD(&q[4*offset]);
    __SIMD_DATATYPE y5 = _SIMD_FMA(x5, h1, q5);
    __SIMD_DATATYPE q6 = _LOAD(&q[5*offset]);
    __SIMD_DATATYPE y6 = _SIMD_FMA(x6, h1, q6);
#else
    __SIMD_DATATYPE q1 = _LOAD(q);
    __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
    __SIMD_DATATYPE q2 = _LOAD(&q[offset]);
    __SIMD_DATATYPE y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
    __SIMD_DATATYPE q3 = _LOAD(&q[2*offset]);
    __SIMD_DATATYPE y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
    __SIMD_DATATYPE q4 = _LOAD(&q[3*offset]);
    __SIMD_DATATYPE y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
    __SIMD_DATATYPE q5 = _LOAD(&q[4*offset]);
    __SIMD_DATATYPE y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT x5, h1));
    __SIMD_DATATYPE q6 = _LOAD(&q[5*offset]);
    __SIMD_DATATYPE y6 = _SIMD_ADD( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT x6, h1));
#endif
#endif /* BLOCK2 */

#ifdef BLOCK4
    __SIMD_DATATYPE a1_1 = _LOAD(&q[ldq*3]);
    __SIMD_DATATYPE a2_1 = _LOAD(&q[ldq*2]);
    __SIMD_DATATYPE a3_1 = _LOAD(&q[ldq]);  
    __SIMD_DATATYPE a4_1 = _LOAD(&q[0]);    

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_2_1 = _SIMD_SET1(hh[ldh+1]);    
    __SIMD_DATATYPE h_3_2 = _SIMD_SET1(hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET1(hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_SET1(hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET1(hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_2_1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_2_1 = _SIMD_BROADCAST(&hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w1 = _SIMD_FMA(a3_1, h_4_3, a4_1);
    w1 = _SIMD_FMA(a2_1, h_4_2, w1);
    w1 = _SIMD_FMA(a1_1, h_4_1, w1);
    register __SIMD_DATATYPE z1 = _SIMD_FMA(a2_1, h_3_2, a3_1);
    z1 = _SIMD_FMA(a1_1, h_3_1, z1);
    register __SIMD_DATATYPE y1 = _SIMD_FMA(a1_1, h_2_1, a2_1);
    register __SIMD_DATATYPE x1 = a1_1;
#else
    register __SIMD_DATATYPE w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_4_3));
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_4_2));                          
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_4_1));                          
    register __SIMD_DATATYPE z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_3_2));
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_3_1));                          
    register __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_2_1));
    register __SIMD_DATATYPE x1 = a1_1;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE a1_2 = _LOAD(&q[(ldq*3)+offset]);                  
    __SIMD_DATATYPE a2_2 = _LOAD(&q[(ldq*2)+offset]);
    __SIMD_DATATYPE a3_2 = _LOAD(&q[ldq+offset]);
    __SIMD_DATATYPE a4_2 = _LOAD(&q[0+offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w2 = _SIMD_FMA(a3_2, h_4_3, a4_2);
    w2 = _SIMD_FMA(a2_2, h_4_2, w2);
    w2 = _SIMD_FMA(a1_2, h_4_1, w2);
    register __SIMD_DATATYPE z2 = _SIMD_FMA(a2_2, h_3_2, a3_2);
    z2 = _SIMD_FMA(a1_2, h_3_1, z2);
    register __SIMD_DATATYPE y2 = _SIMD_FMA(a1_2, h_2_1, a2_2);
    register __SIMD_DATATYPE x2 = a1_2;
#else
    register __SIMD_DATATYPE w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_4_3));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_4_2));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_4_1));
    register __SIMD_DATATYPE z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_3_2));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_3_1));
    register __SIMD_DATATYPE y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_2_1));
    register __SIMD_DATATYPE x2 = a1_2;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE a1_3 = _LOAD(&q[(ldq*3)+2*offset]);
    __SIMD_DATATYPE a2_3 = _LOAD(&q[(ldq*2)+2*offset]);
    __SIMD_DATATYPE a3_3 = _LOAD(&q[ldq+2*offset]);
    __SIMD_DATATYPE a4_3 = _LOAD(&q[0+2*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w3 = _SIMD_FMA(a3_3, h_4_3, a4_3);
    w3 = _SIMD_FMA(a2_3, h_4_2, w3);
    w3 = _SIMD_FMA(a1_3, h_4_1, w3);
    register __SIMD_DATATYPE z3 = _SIMD_FMA(a2_3, h_3_2, a3_3);
    z3 = _SIMD_FMA(a1_3, h_3_1, z3);
    register __SIMD_DATATYPE y3 = _SIMD_FMA(a1_3, h_2_1, a2_3);
    register __SIMD_DATATYPE x3 = a1_3;
#else
    register __SIMD_DATATYPE w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_3, h_4_3));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_4_2));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_4_1));
    register __SIMD_DATATYPE z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_3_2));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_3_1));
    register __SIMD_DATATYPE y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_2_1));
    register __SIMD_DATATYPE x3 = a1_3;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE a1_4 = _LOAD(&q[(ldq*3)+3*offset]);
    __SIMD_DATATYPE a2_4 = _LOAD(&q[(ldq*2)+3*offset]);
    __SIMD_DATATYPE a3_4 = _LOAD(&q[ldq+3*offset]);
    __SIMD_DATATYPE a4_4 = _LOAD(&q[0+3*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w4 = _SIMD_FMA(a3_4, h_4_3, a4_4);
    w4 = _SIMD_FMA(a2_4, h_4_2, w4);
    w4 = _SIMD_FMA(a1_4, h_4_1, w4);
    register __SIMD_DATATYPE z4 = _SIMD_FMA(a2_4, h_3_2, a3_4);
    z4 = _SIMD_FMA(a1_4, h_3_1, z4);
    register __SIMD_DATATYPE y4 = _SIMD_FMA(a1_4, h_2_1, a2_4);
    register __SIMD_DATATYPE x4 = a1_4;
#else
    register __SIMD_DATATYPE w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_4, h_4_3));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_4_2));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_4_1));
    register __SIMD_DATATYPE z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_3_2));
    z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_3_1));
    register __SIMD_DATATYPE y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_2_1));
    register __SIMD_DATATYPE x4 = a1_4;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE a1_5 = _LOAD(&q[(ldq*3)+4*offset]);
    __SIMD_DATATYPE a2_5 = _LOAD(&q[(ldq*2)+4*offset]);
    __SIMD_DATATYPE a3_5 = _LOAD(&q[ldq+4*offset]);
    __SIMD_DATATYPE a4_5 = _LOAD(&q[0+4*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w5 = _SIMD_FMA(a3_5, h_4_3, a4_5);
    w5 = _SIMD_FMA(a2_5, h_4_2, w5);
    w5 = _SIMD_FMA(a1_5, h_4_1, w5);
    register __SIMD_DATATYPE z5 = _SIMD_FMA(a2_5, h_3_2, a3_5);
    z5 = _SIMD_FMA(a1_5, h_3_1, z5);
    register __SIMD_DATATYPE y5 = _SIMD_FMA(a1_5, h_2_1, a2_5);
    register __SIMD_DATATYPE x5 = a1_5;
#else
    register __SIMD_DATATYPE w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_5, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_5, h_4_3));
    w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT w5, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_5, h_4_2));
    w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT w5, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_5, h_4_1));
    register __SIMD_DATATYPE z5 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_5, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_5, h_3_2));
    z5 = _SIMD_ADD( ADDITIONAL_ARGUMENT z5, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_5, h_3_1));
    register __SIMD_DATATYPE y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_5, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_5, h_2_1));
    register __SIMD_DATATYPE x5 = a1_5;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE a1_6 = _LOAD(&q[(ldq*3)+5*offset]);
    __SIMD_DATATYPE a2_6 = _LOAD(&q[(ldq*2)+5*offset]);
    __SIMD_DATATYPE a3_6 = _LOAD(&q[ldq+5*offset]);
    __SIMD_DATATYPE a4_6 = _LOAD(&q[0+5*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w6 = _SIMD_FMA(a3_6, h_4_3, a4_6);
    w6 = _SIMD_FMA(a2_6, h_4_2, w6);
    w6 = _SIMD_FMA(a1_6, h_4_1, w6);
    register __SIMD_DATATYPE z6 = _SIMD_FMA(a2_6, h_3_2, a3_6);
    z6 = _SIMD_FMA(a1_6, h_3_1, z6);
    register __SIMD_DATATYPE y6 = _SIMD_FMA(a1_6, h_2_1, a2_6);
    register __SIMD_DATATYPE x6 = a1_6;
#else
    register __SIMD_DATATYPE w6 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_6, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_6, h_4_3));
    w6 = _SIMD_ADD( ADDITIONAL_ARGUMENT w6, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_6, h_4_2));
    w6 = _SIMD_ADD( ADDITIONAL_ARGUMENT w6, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_6, h_4_1));
    register __SIMD_DATATYPE z6 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_6, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_6, h_3_2));
    z6 = _SIMD_ADD( ADDITIONAL_ARGUMENT z6, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_6, h_3_1));
    register __SIMD_DATATYPE y6 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_6, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_6, h_2_1));
    register __SIMD_DATATYPE x6 = a1_6;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE q1;
    __SIMD_DATATYPE q2;
    __SIMD_DATATYPE q3;
    __SIMD_DATATYPE q4;
    __SIMD_DATATYPE q5;
    __SIMD_DATATYPE q6;

    __SIMD_DATATYPE h1;
    __SIMD_DATATYPE h2;
    __SIMD_DATATYPE h3;
    __SIMD_DATATYPE h4;
#endif /* BLOCK4 */

#ifdef BLOCK6
    
    __SIMD_DATATYPE a1_1 = _LOAD(&q[ldq*5]);
    __SIMD_DATATYPE a2_1 = _LOAD(&q[ldq*4]);
    __SIMD_DATATYPE a3_1 = _LOAD(&q[ldq*3]);
    __SIMD_DATATYPE a4_1 = _LOAD(&q[ldq*2]);
    __SIMD_DATATYPE a5_1 = _LOAD(&q[ldq]);  
    __SIMD_DATATYPE a6_1 = _LOAD(&q[0]);    

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_6_5 = _SIMD_SET1(hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_SET1(hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_SET1(hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_SET1(hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_SET1(hh[(ldh*5)+5]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_6_5 = _SIMD_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_6_5 = _SIMD_BROADCAST(&hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_BROADCAST(&hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_BROADCAST(&hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_BROADCAST(&hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_BROADCAST(&hh[(ldh*5)+5]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t1 = _SIMD_FMA(a5_1, h_6_5, a6_1);
    t1 = _SIMD_FMA(a4_1, h_6_4, t1);
    t1 = _SIMD_FMA(a3_1, h_6_3, t1);
    t1 = _SIMD_FMA(a2_1, h_6_2, t1);
    t1 = _SIMD_FMA(a1_1, h_6_1, t1);
#else
    register __SIMD_DATATYPE t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_1, h_6_5)); 
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_1, h_6_4));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_6_3));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_6_2));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_6_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_5_4 = _SIMD_SET1(hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_SET1(hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_SET1(hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_SET1(hh[(ldh*4)+4]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_5_4 = _SIMD_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_5_4 = _SIMD_BROADCAST(&hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_BROADCAST(&hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_BROADCAST(&hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_BROADCAST(&hh[(ldh*4)+4]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE v1 = _SIMD_FMA(a4_1, h_5_4, a5_1);
    v1 = _SIMD_FMA(a3_1, h_5_3, v1);
    v1 = _SIMD_FMA(a2_1, h_5_2, v1);
    v1 = _SIMD_FMA(a1_1, h_5_1, v1);
#else
    register __SIMD_DATATYPE v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_1, h_5_4)); 
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_5_3));
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_5_2));
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_5_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_4_3 = _SIMD_SET1(hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET1(hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_4_3 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_4_3 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w1 = _SIMD_FMA(a3_1, h_4_3, a4_1);
    w1 = _SIMD_FMA(a2_1, h_4_2, w1);
    w1 = _SIMD_FMA(a1_1, h_4_1, w1);
#else
    register __SIMD_DATATYPE w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_4_3)); 
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_4_2));
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_4_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_2_1 = _SIMD_SET1(hh[ldh+1]);    
    __SIMD_DATATYPE h_3_2 = _SIMD_SET1(hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_2_1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_2_1 = _SIMD_BROADCAST(&hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE z1 = _SIMD_FMA(a2_1, h_3_2, a3_1);
    z1 = _SIMD_FMA(a1_1, h_3_1, z1);
    register __SIMD_DATATYPE y1 = _SIMD_FMA(a1_1, h_2_1, a2_1);
#else
    register __SIMD_DATATYPE z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_3_2));
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_3_1));
    register __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_2_1)); 
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x1 = a1_1;

    __SIMD_DATATYPE a1_2 = _LOAD(&q[(ldq*5)+offset]);
    __SIMD_DATATYPE a2_2 = _LOAD(&q[(ldq*4)+offset]);
    __SIMD_DATATYPE a3_2 = _LOAD(&q[(ldq*3)+offset]);
    __SIMD_DATATYPE a4_2 = _LOAD(&q[(ldq*2)+offset]);
    __SIMD_DATATYPE a5_2 = _LOAD(&q[(ldq)+offset]);
    __SIMD_DATATYPE a6_2 = _LOAD(&q[offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t2 = _SIMD_FMA(a5_2, h_6_5, a6_2);
    t2 = _SIMD_FMA(a4_2, h_6_4, t2);
    t2 = _SIMD_FMA(a3_2, h_6_3, t2);
    t2 = _SIMD_FMA(a2_2, h_6_2, t2);
    t2 = _SIMD_FMA(a1_2, h_6_1, t2);
    register __SIMD_DATATYPE v2 = _SIMD_FMA(a4_2, h_5_4, a5_2);
    v2 = _SIMD_FMA(a3_2, h_5_3, v2);
    v2 = _SIMD_FMA(a2_2, h_5_2, v2);
    v2 = _SIMD_FMA(a1_2, h_5_1, v2);
    register __SIMD_DATATYPE w2 = _SIMD_FMA(a3_2, h_4_3, a4_2);
    w2 = _SIMD_FMA(a2_2, h_4_2, w2);
    w2 = _SIMD_FMA(a1_2, h_4_1, w2);
    register __SIMD_DATATYPE z2 = _SIMD_FMA(a2_2, h_3_2, a3_2);
    z2 = _SIMD_FMA(a1_2, h_3_1, z2);
    register __SIMD_DATATYPE y2 = _SIMD_FMA(a1_2, h_2_1, a2_2);
#else
    register __SIMD_DATATYPE t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_2, h_6_5));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_2, h_6_4));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_6_3));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_6_2));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_6_1));
    register __SIMD_DATATYPE v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_2, h_5_4));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_5_3));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_5_2));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_5_1));
    register __SIMD_DATATYPE w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_4_3));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_4_2));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_4_1));
    register __SIMD_DATATYPE z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_3_2));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_3_1));
    register __SIMD_DATATYPE y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_2_1));
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x2 = a1_2;

    __SIMD_DATATYPE a1_3 = _LOAD(&q[(ldq*5)+2*offset]);
    __SIMD_DATATYPE a2_3 = _LOAD(&q[(ldq*4)+2*offset]);
    __SIMD_DATATYPE a3_3 = _LOAD(&q[(ldq*3)+2*offset]);
    __SIMD_DATATYPE a4_3 = _LOAD(&q[(ldq*2)+2*offset]);
    __SIMD_DATATYPE a5_3 = _LOAD(&q[(ldq)+2*offset]);
    __SIMD_DATATYPE a6_3 = _LOAD(&q[2*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t3 = _SIMD_FMA(a5_3, h_6_5, a6_3);
    t3 = _SIMD_FMA(a4_3, h_6_4, t3);
    t3 = _SIMD_FMA(a3_3, h_6_3, t3);
    t3 = _SIMD_FMA(a2_3, h_6_2, t3);
    t3 = _SIMD_FMA(a1_3, h_6_1, t3);
    register __SIMD_DATATYPE v3 = _SIMD_FMA(a4_3, h_5_4, a5_3);
    v3 = _SIMD_FMA(a3_3, h_5_3, v3);
    v3 = _SIMD_FMA(a2_3, h_5_2, v3);
    v3 = _SIMD_FMA(a1_3, h_5_1, v3);
    register __SIMD_DATATYPE w3 = _SIMD_FMA(a3_3, h_4_3, a4_3);
    w3 = _SIMD_FMA(a2_3, h_4_2, w3);
    w3 = _SIMD_FMA(a1_3, h_4_1, w3);
    register __SIMD_DATATYPE z3 = _SIMD_FMA(a2_3, h_3_2, a3_3);
    z3 = _SIMD_FMA(a1_3, h_3_1, z3);
    register __SIMD_DATATYPE y3 = _SIMD_FMA(a1_3, h_2_1, a2_3);
#else
    register __SIMD_DATATYPE t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_3, h_6_5));
    t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_3, h_6_4));
    t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_3, h_6_3));
    t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_6_2));
    t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_6_1));
    register __SIMD_DATATYPE v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_3, h_5_4));
    v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_3, h_5_3));
    v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_5_2));
    v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_5_1));
    register __SIMD_DATATYPE w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_3, h_4_3));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_4_2));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_4_1));
    register __SIMD_DATATYPE z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_3_2));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_3_1));
    register __SIMD_DATATYPE y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_2_1));
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x3 = a1_3;

    __SIMD_DATATYPE a1_4 = _LOAD(&q[(ldq*5)+3*offset]);
    __SIMD_DATATYPE a2_4 = _LOAD(&q[(ldq*4)+3*offset]);
    __SIMD_DATATYPE a3_4 = _LOAD(&q[(ldq*3)+3*offset]);
    __SIMD_DATATYPE a4_4 = _LOAD(&q[(ldq*2)+3*offset]);
    __SIMD_DATATYPE a5_4 = _LOAD(&q[(ldq)+3*offset]);
    __SIMD_DATATYPE a6_4 = _LOAD(&q[3*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t4 = _SIMD_FMA(a5_4, h_6_5, a6_4);
    t4 = _SIMD_FMA(a4_4, h_6_4, t4);
    t4 = _SIMD_FMA(a3_4, h_6_3, t4);
    t4 = _SIMD_FMA(a2_4, h_6_2, t4);
    t4 = _SIMD_FMA(a1_4, h_6_1, t4);
    register __SIMD_DATATYPE v4 = _SIMD_FMA(a4_4, h_5_4, a5_4);
    v4 = _SIMD_FMA(a3_4, h_5_3, v4);
    v4 = _SIMD_FMA(a2_4, h_5_2, v4);
    v4 = _SIMD_FMA(a1_4, h_5_1, v4);
    register __SIMD_DATATYPE w4 = _SIMD_FMA(a3_4, h_4_3, a4_4);
    w4 = _SIMD_FMA(a2_4, h_4_2, w4);
    w4 = _SIMD_FMA(a1_4, h_4_1, w4);
    register __SIMD_DATATYPE z4 = _SIMD_FMA(a2_4, h_3_2, a3_4);
    z4 = _SIMD_FMA(a1_4, h_3_1, z4);
    register __SIMD_DATATYPE y4 = _SIMD_FMA(a1_4, h_2_1, a2_4);
#else
    register __SIMD_DATATYPE t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_4, h_6_5));
    t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT t4, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_4, h_6_4));
    t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT t4, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_4, h_6_3));
    t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT t4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_6_2));
    t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT t4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_6_1));
    register __SIMD_DATATYPE v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_4, h_5_4));
    v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT v4, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_4, h_5_3));
    v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT v4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_5_2));
    v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT v4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_5_1));
    register __SIMD_DATATYPE w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_4, h_4_3));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_4_2));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_4_1));
    register __SIMD_DATATYPE z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_3_2));
    z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_3_1));
    register __SIMD_DATATYPE y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_2_1));
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x4 = a1_4;

    __SIMD_DATATYPE a1_5 = _LOAD(&q[(ldq*5)+4*offset]);
    __SIMD_DATATYPE a2_5 = _LOAD(&q[(ldq*4)+4*offset]);
    __SIMD_DATATYPE a3_5 = _LOAD(&q[(ldq*3)+4*offset]);
    __SIMD_DATATYPE a4_5 = _LOAD(&q[(ldq*2)+4*offset]);
    __SIMD_DATATYPE a5_5 = _LOAD(&q[(ldq)+4*offset]);
    __SIMD_DATATYPE a6_5 = _LOAD(&q[4*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t5 = _SIMD_FMA(a5_5, h_6_5, a6_5);
    t5 = _SIMD_FMA(a4_5, h_6_4, t5);
    t5 = _SIMD_FMA(a3_5, h_6_3, t5);
    t5 = _SIMD_FMA(a2_5, h_6_2, t5);
    t5 = _SIMD_FMA(a1_5, h_6_1, t5);
    register __SIMD_DATATYPE v5 = _SIMD_FMA(a4_5, h_5_4, a5_5);
    v5 = _SIMD_FMA(a3_5, h_5_3, v5);
    v5 = _SIMD_FMA(a2_5, h_5_2, v5);
    v5 = _SIMD_FMA(a1_5, h_5_1, v5);
    register __SIMD_DATATYPE w5 = _SIMD_FMA(a3_5, h_4_3, a4_5);
    w5 = _SIMD_FMA(a2_5, h_4_2, w5);
    w5 = _SIMD_FMA(a1_5, h_4_1, w5);
    register __SIMD_DATATYPE z5 = _SIMD_FMA(a2_5, h_3_2, a3_5);
    z5 = _SIMD_FMA(a1_5, h_3_1, z5);
    register __SIMD_DATATYPE y5 = _SIMD_FMA(a1_5, h_2_1, a2_5);
#else
    register __SIMD_DATATYPE t5 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_5, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_5, h_6_5));
    t5 = _SIMD_ADD( ADDITIONAL_ARGUMENT t5, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_5, h_6_4));
    t5 = _SIMD_ADD( ADDITIONAL_ARGUMENT t5, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_5, h_6_3));
    t5 = _SIMD_ADD( ADDITIONAL_ARGUMENT t5, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_5, h_6_2));
    t5 = _SIMD_ADD( ADDITIONAL_ARGUMENT t5, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_5, h_6_1));
    register __SIMD_DATATYPE v5 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_5, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_5, h_5_4));
    v5 = _SIMD_ADD( ADDITIONAL_ARGUMENT v5, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_5, h_5_3));
    v5 = _SIMD_ADD( ADDITIONAL_ARGUMENT v5, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_5, h_5_2));
    v5 = _SIMD_ADD( ADDITIONAL_ARGUMENT v5, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_5, h_5_1));
    register __SIMD_DATATYPE w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_5, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_5, h_4_3));
    w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT w5, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_5, h_4_2));
    w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT w5, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_5, h_4_1));
    register __SIMD_DATATYPE z5 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_5, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_5, h_3_2));
    z5 = _SIMD_ADD( ADDITIONAL_ARGUMENT z5, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_5, h_3_1));
    register __SIMD_DATATYPE y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_5, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_5, h_2_1));
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x5 = a1_5;

    __SIMD_DATATYPE a1_6 = _LOAD(&q[(ldq*5)+5*offset]);
    __SIMD_DATATYPE a2_6 = _LOAD(&q[(ldq*4)+5*offset]);
    __SIMD_DATATYPE a3_6 = _LOAD(&q[(ldq*3)+5*offset]);
    __SIMD_DATATYPE a4_6 = _LOAD(&q[(ldq*2)+5*offset]);
    __SIMD_DATATYPE a5_6 = _LOAD(&q[(ldq)+5*offset]);
    __SIMD_DATATYPE a6_6 = _LOAD(&q[5*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t6 = _SIMD_FMA(a5_6, h_6_5, a6_6);
    t6 = _SIMD_FMA(a4_6, h_6_4, t6);
    t6 = _SIMD_FMA(a3_6, h_6_3, t6);
    t6 = _SIMD_FMA(a2_6, h_6_2, t6);
    t6 = _SIMD_FMA(a1_6, h_6_1, t6);
    register __SIMD_DATATYPE v6 = _SIMD_FMA(a4_6, h_5_4, a5_6);
    v6 = _SIMD_FMA(a3_6, h_5_3, v6);
    v6 = _SIMD_FMA(a2_6, h_5_2, v6);
    v6 = _SIMD_FMA(a1_6, h_5_1, v6);
    register __SIMD_DATATYPE w6 = _SIMD_FMA(a3_6, h_4_3, a4_6);
    w6 = _SIMD_FMA(a2_6, h_4_2, w6);
    w6 = _SIMD_FMA(a1_6, h_4_1, w6);
    register __SIMD_DATATYPE z6 = _SIMD_FMA(a2_6, h_3_2, a3_6);
    z6 = _SIMD_FMA(a1_6, h_3_1, z6);
    register __SIMD_DATATYPE y6 = _SIMD_FMA(a1_6, h_2_1, a2_6);
#else
    register __SIMD_DATATYPE t6 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_6, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_6, h_6_5));
    t6 = _SIMD_ADD( ADDITIONAL_ARGUMENT t6, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_6, h_6_4));
    t6 = _SIMD_ADD( ADDITIONAL_ARGUMENT t6, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_6, h_6_3));
    t6 = _SIMD_ADD( ADDITIONAL_ARGUMENT t6, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_6, h_6_2));
    t6 = _SIMD_ADD( ADDITIONAL_ARGUMENT t6, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_6, h_6_1));
    register __SIMD_DATATYPE v6 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_6, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_6, h_5_4));
    v6 = _SIMD_ADD( ADDITIONAL_ARGUMENT v6, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_6, h_5_3));
    v6 = _SIMD_ADD( ADDITIONAL_ARGUMENT v6, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_6, h_5_2));
    v6 = _SIMD_ADD( ADDITIONAL_ARGUMENT v6, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_6, h_5_1));
    register __SIMD_DATATYPE w6 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_6, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_6, h_4_3));
    w6 = _SIMD_ADD( ADDITIONAL_ARGUMENT w6, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_6, h_4_2));
    w6 = _SIMD_ADD( ADDITIONAL_ARGUMENT w6, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_6, h_4_1));
    register __SIMD_DATATYPE z6 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_6, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_6, h_3_2));
    z6 = _SIMD_ADD( ADDITIONAL_ARGUMENT z6, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_6, h_3_1));
    register __SIMD_DATATYPE y6 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_6, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_6, h_2_1));
#endif /* __ELPA_USE_FMA__ */
 
    register __SIMD_DATATYPE x6 = a1_6;

    __SIMD_DATATYPE q1;
    __SIMD_DATATYPE q2;
    __SIMD_DATATYPE q3;
    __SIMD_DATATYPE q4;
    __SIMD_DATATYPE q5;
    __SIMD_DATATYPE q6;

    __SIMD_DATATYPE h1;
    __SIMD_DATATYPE h2;
    __SIMD_DATATYPE h3;
    __SIMD_DATATYPE h4;
    __SIMD_DATATYPE h5;
    __SIMD_DATATYPE h6;

#endif /* BLOCK6 */


    for(i = BLOCK; i < nb; i++)
      {

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h1 = _SIMD_SET1(hh[i-(BLOCK-1)]);
        h2 = _SIMD_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == SPARC64_SSE
        h1 = _SIMD_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
        h2 = _SIMD_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif
#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h1 = _SIMD_BROADCAST(&hh[i-(BLOCK-1)]);
        h2 = _SIMD_BROADCAST(&hh[ldh+i-(BLOCK-2)]);
#endif /*   VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

        q1 = _LOAD(&q[i*ldq]);
        q2 = _LOAD(&q[(i*ldq)+offset]);
        q3 = _LOAD(&q[(i*ldq)+2*offset]);
        q4 = _LOAD(&q[(i*ldq)+3*offset]);
        q5 = _LOAD(&q[(i*ldq)+4*offset]);
        q6 = _LOAD(&q[(i*ldq)+5*offset]);

#ifdef __ELPA_USE_FMA__
        x1 = _SIMD_FMA(q1, h1, x1);
        y1 = _SIMD_FMA(q1, h2, y1);
        x2 = _SIMD_FMA(q2, h1, x2);
        y2 = _SIMD_FMA(q2, h2, y2);
        x3 = _SIMD_FMA(q3, h1, x3);
        y3 = _SIMD_FMA(q3, h2, y3);
        x4 = _SIMD_FMA(q4, h1, x4);
        y4 = _SIMD_FMA(q4, h2, y4);
        x5 = _SIMD_FMA(q5, h1, x5);
        y5 = _SIMD_FMA(q5, h2, y5);
        x6 = _SIMD_FMA(q6, h1, x6);
        y6 = _SIMD_FMA(q6, h2, y6);
#else
        x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
        y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
        x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
        y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
        x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
        y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
        x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
        y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
        x5 = _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h1));
        y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT y5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h2));
        x6 = _SIMD_ADD( ADDITIONAL_ARGUMENT x6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h1));
        y6 = _SIMD_ADD( ADDITIONAL_ARGUMENT y6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h2));
#endif /* __ELPA_USE_FMA__ */

#if defined(BLOCK4) || defined(BLOCK6)
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h3 = _SIMD_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
        h3 = _SIMD_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h3 = _SIMD_BROADCAST(&hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
        z1 = _SIMD_FMA(q1, h3, z1);
        z2 = _SIMD_FMA(q2, h3, z2);
        z3 = _SIMD_FMA(q3, h3, z3);
        z4 = _SIMD_FMA(q4, h3, z4);
        z5 = _SIMD_FMA(q5, h3, z5);
        z6 = _SIMD_FMA(q6, h3, z6);
#else
        z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
        z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
        z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h3));
        z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h3));
        z5 = _SIMD_ADD( ADDITIONAL_ARGUMENT z5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h3));
        z6 = _SIMD_ADD( ADDITIONAL_ARGUMENT z6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h4 = _SIMD_SET1(hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == SPARC64_SSE
        h4 = _SIMD_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h4 = _SIMD_BROADCAST(&hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
        w1 = _SIMD_FMA(q1, h4, w1);
        w2 = _SIMD_FMA(q2, h4, w2);
        w3 = _SIMD_FMA(q3, h4, w3);
        w4 = _SIMD_FMA(q4, h4, w4);
        w5 = _SIMD_FMA(q5, h4, w5);
        w6 = _SIMD_FMA(q6, h4, w6);
#else
        w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4));
        w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h4));
        w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h4));
        w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h4));
        w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT w5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h4));
        w6 = _SIMD_ADD( ADDITIONAL_ARGUMENT w6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h4));
#endif /* __ELPA_USE_FMA__ */
				
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h5 = _SIMD_SET1(hh[(ldh*4)+i-1]);
#endif
#if VEC_SET == SPARC64_SSE
        h5 = _SIMD_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h5 = _SIMD_BROADCAST(&hh[(ldh*4)+i-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
        v1 = _SIMD_FMA(q1, h5, v1);
        v2 = _SIMD_FMA(q2, h5, v2);
        v3 = _SIMD_FMA(q3, h5, v3);
        v4 = _SIMD_FMA(q4, h5, v4);
        v5 = _SIMD_FMA(q5, h5, v5);
        v6 = _SIMD_FMA(q6, h5, v6);
#else
        v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h5));
        v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h5));
        v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h5));
        v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT v4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h5));
        v5 = _SIMD_ADD( ADDITIONAL_ARGUMENT v5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h5));
        v6 = _SIMD_ADD( ADDITIONAL_ARGUMENT v6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h6 = _SIMD_SET1(hh[(ldh*5)+i]);
#endif

#if VEC_SET == SPARC64_SSE
        h6 = _SIMD_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif


#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h6 = _SIMD_BROADCAST(&hh[(ldh*5)+i-(BLOCK-6)]);
#endif

#ifdef __ELPA_USE_FMA__
        t1 = _SIMD_FMA(q1, h6, t1);
        t2 = _SIMD_FMA(q2, h6, t2);
        t3 = _SIMD_FMA(q3, h6, t3);
        t4 = _SIMD_FMA(q4, h6, t4);
        t5 = _SIMD_FMA(q5, h6, t5);
        t6 = _SIMD_FMA(q6, h6, t6);
#else
        t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h6));
        t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h6));
        t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h6));
        t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT t4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h6));
        t5 = _SIMD_ADD( ADDITIONAL_ARGUMENT t5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h6));
        t6 = _SIMD_ADD( ADDITIONAL_ARGUMENT t6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h6));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */
      }
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-1)]);
#endif

    q1 = _LOAD(&q[nb*ldq]);
    q2 = _LOAD(&q[(nb*ldq)+offset]);
    q3 = _LOAD(&q[(nb*ldq)+2*offset]);
    q4 = _LOAD(&q[(nb*ldq)+3*offset]);
    q5 = _LOAD(&q[(nb*ldq)+4*offset]);
    q6 = _LOAD(&q[(nb*ldq)+5*offset]);
#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
    x5 = _SIMD_FMA(q5, h1, x5);
    x6 = _SIMD_FMA(q6, h1, x6);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
    x5 = _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h1));
    x6 = _SIMD_ADD( ADDITIONAL_ARGUMENT x6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h1));
#endif /* __ELPA_USE_FMA__ */

#if defined(BLOCK4) || defined(BLOCK6)
    
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
    y4 = _SIMD_FMA(q4, h2, y4);
    y5 = _SIMD_FMA(q5, h2, y5);
    y6 = _SIMD_FMA(q6, h2, y6);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
    y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
    y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT y5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h2));
    y6 = _SIMD_ADD( ADDITIONAL_ARGUMENT y6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
    z2 = _SIMD_FMA(q2, h3, z2);
    z3 = _SIMD_FMA(q3, h3, z3);
    z4 = _SIMD_FMA(q4, h3, z4);
    z5 = _SIMD_FMA(q5, h3, z5);
    z6 = _SIMD_FMA(q6, h3, z6);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h3));
    z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h3));
    z5 = _SIMD_ADD( ADDITIONAL_ARGUMENT z5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h3));
    z6 = _SIMD_ADD( ADDITIONAL_ARGUMENT z6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h3));
#endif

#ifdef BLOCK4
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-2]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-2], hh[nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-2]);
#endif

    q1 = _LOAD(&q[(nb+1)*ldq]);
    q2 = _LOAD(&q[((nb+1)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+1)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+1)*ldq)+3*offset]);
    q5 = _LOAD(&q[((nb+1)*ldq)+4*offset]);
    q6 = _LOAD(&q[((nb+1)*ldq)+5*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
    x5 = _SIMD_FMA(q5, h1, x5);
    x6 = _SIMD_FMA(q6, h1, x6);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
    x5 = _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h1));
    x6 = _SIMD_ADD( ADDITIONAL_ARGUMENT x6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[(ldh*1)+nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[(ldh*1)+nb-1], hh[(ldh*1)+nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[(ldh*1)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
    y4 = _SIMD_FMA(q4, h2, y4);
    y5 = _SIMD_FMA(q5, h2, y5);
    y6 = _SIMD_FMA(q6, h2, y6);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
    y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
    y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT y5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h2));
    y6 = _SIMD_ADD( ADDITIONAL_ARGUMENT y6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-1], hh[nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-1]);
#endif

    q1 = _LOAD(&q[(nb+2)*ldq]);
    q2 = _LOAD(&q[((nb+2)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+2)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+2)*ldq)+3*offset]);
    q5 = _LOAD(&q[((nb+2)*ldq)+4*offset]);
    q6 = _LOAD(&q[((nb+2)*ldq)+5*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
    x5 = _SIMD_FMA(q5, h1, x5);
    x6 = _SIMD_FMA(q6, h1, x6);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
    x5 = _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h1));
    x6 = _SIMD_ADD( ADDITIONAL_ARGUMENT x6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h1));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 */
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h4 = _SIMD_SET1(hh[(ldh*3)+nb-2]);
#endif

#if VEC_SET == SPARC64_SSE
    h4 = _SIMD_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    w1 = _SIMD_FMA(q1, h4, w1);
    w2 = _SIMD_FMA(q2, h4, w2);
    w3 = _SIMD_FMA(q3, h4, w3);
    w4 = _SIMD_FMA(q4, h4, w4);
    w5 = _SIMD_FMA(q5, h4, w5);
    w6 = _SIMD_FMA(q6, h4, w6);
#else
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4)); 
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h4));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h4));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h4));
    w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT w5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h4));
    w6 = _SIMD_ADD( ADDITIONAL_ARGUMENT w6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h5 = _SIMD_SET1(hh[(ldh*4)+nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h5 = _SIMD_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h5 = _SIMD_BROADCAST(&hh[(ldh*4)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    v1 = _SIMD_FMA(q1, h5, v1);
    v2 = _SIMD_FMA(q2, h5, v2);
    v3 = _SIMD_FMA(q3, h5, v3);
    v4 = _SIMD_FMA(q4, h5, v4);
    v5 = _SIMD_FMA(q5, h5, v5);
    v6 = _SIMD_FMA(q6, h5, v6);
#else
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h5));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h5));
    v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h5));
    v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT v4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h5));
    v5 = _SIMD_ADD( ADDITIONAL_ARGUMENT v5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h5));
    v6 = _SIMD_ADD( ADDITIONAL_ARGUMENT v6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-4]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-4], hh[nb-4]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-2)]);
#endif

    q1 = _LOAD(&q[(nb+1)*ldq]);
    q2 = _LOAD(&q[((nb+1)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+1)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+1)*ldq)+3*offset]);
    q5 = _LOAD(&q[((nb+1)*ldq)+4*offset]);
    q6 = _LOAD(&q[((nb+1)*ldq)+5*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
    x5 = _SIMD_FMA(q5, h1, x5);
    x6 = _SIMD_FMA(q6, h1, x6);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
    x5 = _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h1));
    x6 = _SIMD_ADD( ADDITIONAL_ARGUMENT x6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-3]);
#endif
#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-3], hh[ldh+nb-3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
    y4 = _SIMD_FMA(q4, h2, y4);
    y5 = _SIMD_FMA(q5, h2, y5);
    y6 = _SIMD_FMA(q6, h2, y6);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
    y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
    y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT y5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h2));
    y6 = _SIMD_ADD( ADDITIONAL_ARGUMENT y6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
    z2 = _SIMD_FMA(q2, h3, z2);
    z3 = _SIMD_FMA(q3, h3, z3);
    z4 = _SIMD_FMA(q4, h3, z4);
    z5 = _SIMD_FMA(q5, h3, z5);
    z6 = _SIMD_FMA(q6, h3, z6);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h3));
    z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h3));
    z5 = _SIMD_ADD( ADDITIONAL_ARGUMENT z5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h3));
    z6 = _SIMD_ADD( ADDITIONAL_ARGUMENT z6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h4 = _SIMD_SET1(hh[(ldh*3)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
    h4 = _SIMD_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    w1 = _SIMD_FMA(q1, h4, w1);
    w2 = _SIMD_FMA(q2, h4, w2);
    w3 = _SIMD_FMA(q3, h4, w3);
    w4 = _SIMD_FMA(q4, h4, w4);
    w5 = _SIMD_FMA(q5, h4, w5);
    w6 = _SIMD_FMA(q6, h4, w6);
#else
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h4));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h4));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h4));
    w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT w5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h4));
    w6 = _SIMD_ADD( ADDITIONAL_ARGUMENT w6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-3]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-3], hh[nb-3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-3]);
#endif

    q1 = _LOAD(&q[(nb+2)*ldq]);
    q2 = _LOAD(&q[((nb+2)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+2)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+2)*ldq)+3*offset]);
    q5 = _LOAD(&q[((nb+2)*ldq)+4*offset]);
    q6 = _LOAD(&q[((nb+2)*ldq)+5*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
    x5 = _SIMD_FMA(q5, h1, x5);
    x6 = _SIMD_FMA(q6, h1, x6);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
    x5 = _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h1));
    x6 = _SIMD_ADD( ADDITIONAL_ARGUMENT x6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-2]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
    y4 = _SIMD_FMA(q4, h2, y4);
    y5 = _SIMD_FMA(q5, h2, y5);
    y6 = _SIMD_FMA(q6, h2, y6);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
    y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
    y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT y5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h2));
    y6 = _SIMD_ADD( ADDITIONAL_ARGUMENT y6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h2));
#endif /* __ELPA_USE_FMA__ */
 
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
    z2 = _SIMD_FMA(q2, h3, z2);
    z3 = _SIMD_FMA(q3, h3, z3);
    z4 = _SIMD_FMA(q4, h3, z4);
    z5 = _SIMD_FMA(q5, h3, z5);
    z6 = _SIMD_FMA(q6, h3, z6);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h3));
    z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h3));
    z5 = _SIMD_ADD( ADDITIONAL_ARGUMENT z5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h3));
    z6 = _SIMD_ADD( ADDITIONAL_ARGUMENT z6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-2]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-2], hh[nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-4)]);
#endif
    q1 = _LOAD(&q[(nb+3)*ldq]);
    q2 = _LOAD(&q[((nb+3)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+3)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+3)*ldq)+3*offset]);
    q5 = _LOAD(&q[((nb+3)*ldq)+4*offset]);
    q6 = _LOAD(&q[((nb+3)*ldq)+5*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
    x5 = _SIMD_FMA(q5, h1, x5);
    x6 = _SIMD_FMA(q6, h1, x6);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
    x5 = _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h1));
    x6 = _SIMD_ADD( ADDITIONAL_ARGUMENT x6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
    y4 = _SIMD_FMA(q4, h2, y4);
    y5 = _SIMD_FMA(q5, h2, y5);
    y6 = _SIMD_FMA(q6, h2, y6);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
    y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
    y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT y5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h2));
    y6 = _SIMD_ADD( ADDITIONAL_ARGUMENT y6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-1], hh[nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-5)]);
#endif

    q1 = _LOAD(&q[(nb+4)*ldq]);
    q2 = _LOAD(&q[((nb+4)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+4)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+4)*ldq)+3*offset]);
    q5 = _LOAD(&q[((nb+4)*ldq)+4*offset]);
    q6 = _LOAD(&q[((nb+4)*ldq)+5*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
    x5 = _SIMD_FMA(q5, h1, x5);
    x6 = _SIMD_FMA(q6, h1, x6);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
    x5 = _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h1));
    x6 = _SIMD_ADD( ADDITIONAL_ARGUMENT x6, _SIMD_MUL( ADDITIONAL_ARGUMENT q6,h1));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Rank-2 update of Q [ROW_LENGTH x nb+1]
    /////////////////////////////////////////////////////
#endif
#ifdef BLOCK4
    /////////////////////////////////////////////////////
    // Rank-1 update of Q [6 x nb+3]
    /////////////////////////////////////////////////////
#endif
#ifdef BLOCK6
    /////////////////////////////////////////////////////
    // Apply tau, correct wrong calculation using pre-calculated scalar products
    /////////////////////////////////////////////////////
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE tau1 = _SIMD_SET1(hh[0]);

    __SIMD_DATATYPE tau2 = _SIMD_SET1(hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_SET1(hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_SET1(hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_SET1(hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_SET1(hh[ldh*5]);       
#endif

#ifdef BLOCK2    
    __SIMD_DATATYPE vs = _SIMD_SET1(s);
#endif
#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET1(s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET1(s_1_3);  
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET1(s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET1(s_1_4);  
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET1(s_2_4);  
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET1(s_3_4);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET1(scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET1(scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET1(scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET1(scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET1(scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET1(scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_SET1(scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_SET1(scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_SET1(scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_SET1(scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_SET1(scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_SET1(scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_SET1(scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_SET1(scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_SET1(scalarprods[14]);
#endif
#endif /* VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128  */

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE tau1 = _SIMD_SET(hh[0], hh[0]);

    __SIMD_DATATYPE tau2 = _SIMD_SET(hh[ldh], hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_SET(hh[ldh*2], hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_SET(hh[ldh*3], hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_SET(hh[ldh*4], hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_SET(hh[ldh*5], hh[ldh*5]);
#endif

#ifdef BLOCK2
    __SIMD_DATATYPE vs = _SIMD_SET(s, s);
#endif
#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET(s_1_2, s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET(s_1_3, s_1_3);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET(s_2_3, s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET(s_1_4, s_1_4);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET(s_2_4, s_2_4);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET(s_3_4, s_3_4);

#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET(scalarprods[0], scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET(scalarprods[1], scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET(scalarprods[2], scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET(scalarprods[3], scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET(scalarprods[4], scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET(scalarprods[5], scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_SET(scalarprods[6], scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_SET(scalarprods[7], scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_SET(scalarprods[8], scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_SET(scalarprods[9], scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_SET(scalarprods[10], scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_SET(scalarprods[11], scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_SET(scalarprods[12], scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_SET(scalarprods[13], scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_SET(scalarprods[14], scalarprods[14]);
#endif
#endif /* VEC_SET == SPARC64_SSE */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   __SIMD_DATATYPE tau1 = _SIMD_BROADCAST(hh);
   __SIMD_DATATYPE tau2 = _SIMD_BROADCAST(&hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_BROADCAST(&hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_BROADCAST(&hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_BROADCAST(&hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_BROADCAST(&hh[ldh*5]);
#endif

#ifdef BLOCK2  
   __SIMD_DATATYPE vs = _SIMD_BROADCAST(&s);
#endif

#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_BROADCAST(&s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_BROADCAST(&s_1_3);
   __SIMD_DATATYPE vs_2_3 = _SIMD_BROADCAST(&s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_BROADCAST(&s_1_4);
   __SIMD_DATATYPE vs_2_4 = _SIMD_BROADCAST(&s_2_4);
   __SIMD_DATATYPE vs_3_4 = _SIMD_BROADCAST(&s_3_4);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_BROADCAST(&scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_BROADCAST(&scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_BROADCAST(&scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_BROADCAST(&scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_BROADCAST(&scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_BROADCAST(&scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_BROADCAST(&scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_BROADCAST(&scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_BROADCAST(&scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_BROADCAST(&scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_BROADCAST(&scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_BROADCAST(&scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_BROADCAST(&scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_BROADCAST(&scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_BROADCAST(&scalarprods[14]);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == VSX_SSE || VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _XOR(tau1, sign);
#endif

#if VEC_SET == SPARC64_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
    h1 = _SIMD_NEG( ADDITIONAL_ARGUMENT tau1);
#endif

#if VEC_SET == AVX_512
#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi64((__SIMD_INTEGER) tau1, (__SIMD_INTEGER) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi32((__SIMD_INTEGER) tau1, (__SIMD_INTEGER) sign);
#endif
#endif /* HAVE_AVX512_XEON_PHI */

#ifdef HAVE_AVX512_XEON
    h1 = _XOR(tau1, sign);
#endif
#endif /* VEC_SET == AVX_512 */
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau1;
#endif

   x1 = _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1);
   x2 = _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1);
   x3 = _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1);
   x4 = _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1);
   x5 = _SIMD_MUL( ADDITIONAL_ARGUMENT x5, h1);
   x6 = _SIMD_MUL( ADDITIONAL_ARGUMENT x6, h1);

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == VSX_SSE || VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _XOR(tau2, sign);
#endif

#if VEC_SET == SPARC64_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
   h1 = _SIMD_NEG( ADDITIONAL_ARGUMENT tau2);
#endif

#if VEC_SET == AVX_512
#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi64((__SIMD_INTEGER) tau2, (__SIMD_INTEGER) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi32((__SIMD_INTEGER) tau2, (__SIMD_INTEGER) sign);
#endif
#endif /* HAVE_AVX512_XEON_PHI */

#ifdef HAVE_AVX512_XEON
    h1 = _XOR(tau2, sign);
#endif
#endif /* VEC_SET == AVX_512 */
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs);
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau2;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_2);
#endif /* BLOCK4 || BLOCK6  */

#ifdef BLOCK2

#ifdef __ELPA_USE_FMA__
   y1 = _SIMD_FMA(y1, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_FMA(y2, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
   y3 = _SIMD_FMA(y3, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2));
   y4 = _SIMD_FMA(y4, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2));
   y5 = _SIMD_FMA(y5, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2));
   y6 = _SIMD_FMA(y6, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x6,h2));
#else
   y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
   y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2));
   y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2));
   y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y5,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2));
   y6 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y6,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x6,h2));
#endif
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)

#ifdef __ELPA_USE_FMA__
   y1 = _SIMD_FMSUB(y1, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_FMSUB(y2, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
   y3 = _SIMD_FMSUB(y3, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2));
   y4 = _SIMD_FMSUB(y4, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2));
   y5 = _SIMD_FMSUB(y5, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2));
   y6 = _SIMD_FMSUB(y6, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x6,h2));
#else   
   y1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
   y3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2));
   y4 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2));
   y5 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y5,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2));
   y6 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y6,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x6,h2));
#endif /* __ELPA_USE_FMA__ */

   h1 = tau3;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_3);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_2_3);

#ifdef __ELPA_USE_FMA__
   z1 = _SIMD_FMSUB(z1, h1, _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)));
   z2 = _SIMD_FMSUB(z2, h1, _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)));
   z3 = _SIMD_FMSUB(z3, h1, _SIMD_FMA(y3, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2)));
   z4 = _SIMD_FMSUB(z4, h1, _SIMD_FMA(y4, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2)));
   z5 = _SIMD_FMSUB(z5, h1, _SIMD_FMA(y5, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2)));
   z6 = _SIMD_FMSUB(z6, h1, _SIMD_FMA(y6, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x6,h2)));
#else
   z1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)));
   z2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)));
   z3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2)));
   z4 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2)));
   z5 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z5,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y5,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2)));
   z6 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z6,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y6,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x6,h2)));
#endif /* __ELPA_USE_FMA__ */

   h1 = tau4;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_4);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_2_4);
   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_3_4);

#ifdef __ELPA_USE_FMA__
   w1 = _SIMD_FMSUB(w1, h1, _SIMD_FMA(z1, h4, _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   w2 = _SIMD_FMSUB(w2, h1, _SIMD_FMA(z2, h4, _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
   w3 = _SIMD_FMSUB(w3, h1, _SIMD_FMA(z3, h4, _SIMD_FMA(y3, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2))));
   w4 = _SIMD_FMSUB(w4, h1, _SIMD_FMA(z4, h4, _SIMD_FMA(y4, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2))));
   w5 = _SIMD_FMSUB(w5, h1, _SIMD_FMA(z5, h4, _SIMD_FMA(y5, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2))));
   w6 = _SIMD_FMSUB(w6, h1, _SIMD_FMA(z6, h4, _SIMD_FMA(y6, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x6,h2))));
#else
   w1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   w2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
   w3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w3,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2))));
   w4 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w4,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2))));
   w5 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w5,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z5,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y5,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2))));
   w6 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w6,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z6,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y6,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x6,h2))));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_1_5); 
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_2_5);
   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_3_5);
   h5 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_4_5);

#ifdef __ELPA_USE_FMA__
   v1 = _SIMD_FMSUB(v1, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w1, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   v2 = _SIMD_FMSUB(v2, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w2, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
   v3 = _SIMD_FMSUB(v3, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w3, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4)), _SIMD_FMA(y3, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2))));
   v4 = _SIMD_FMSUB(v4, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w4, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h4)), _SIMD_FMA(y4, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2))));
   v5 = _SIMD_FMSUB(v5, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w5, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5,h4)), _SIMD_FMA(y5, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2))));
   v6 = _SIMD_FMSUB(v6, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w6, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z6,h4)), _SIMD_FMA(y6, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x6,h2))));
#else
   v1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v1,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   v2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v2,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
   v3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v3,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w3,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2))));
   v4 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v4,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w4,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2))));
   v5 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v5,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w5,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z5,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y5,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2))));
   v6 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v6,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w6,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z6,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y6,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x6,h2))));
#endif /* __ELPA_USE_FMA__ */

   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_1_6);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_2_6);
   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_3_6);
   h5 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_4_6);
   h6 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_5_6);

#ifdef __ELPA_USE_FMA__
   t1 = _SIMD_FMSUB(t1, tau6, _SIMD_FMA(v1, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w1, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)))));
   t2 = _SIMD_FMSUB(t2, tau6, _SIMD_FMA(v2, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w2, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)))));
   t3 = _SIMD_FMSUB(t3, tau6, _SIMD_FMA(v3, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w3, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4)), _SIMD_FMA(y3, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2)))));
   t4 = _SIMD_FMSUB(t4, tau6, _SIMD_FMA(v4, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w4, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h4)), _SIMD_FMA(y4, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2)))));
   t5 = _SIMD_FMSUB(t5, tau6, _SIMD_FMA(v5, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w5, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5,h4)), _SIMD_FMA(y5, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2)))));
   t6 = _SIMD_FMSUB(t6, tau6, _SIMD_FMA(v6, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w6, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z6,h4)), _SIMD_FMA(y6, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x6,h2)))));
#else
   t1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t1,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v1,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)))));
   t2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t2,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v2,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)))));
   t3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t3,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v3,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w3,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2)))));
   t4 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t4,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v4,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w4,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2)))));
   t5 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t5,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v5,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w5,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z5,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y5,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2)))));
   t6 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t6,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v6,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w6,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z6,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y6,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x6,h2)))));
#endif /* __ELPA_USE_FMA__ */

   /////////////////////////////////////////////////////
   // Rank-1 update of Q [ROW_LENGTH x nb+3]
   /////////////////////////////////////////////////////
#endif /* BLOCK6 */

   q1 = _LOAD(&q[0]);
#ifdef BLOCK2
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, y1);
#endif
#ifdef BLOCK4
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, w1);
#endif
#ifdef BLOCK6
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, t1); 
#endif
   _STORE(&q[0],q1);
   q2 = _LOAD(&q[offset]);
#ifdef BLOCK2
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, y2);
#endif
#ifdef BLOCK4
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, w2);
#endif
#ifdef BLOCK6
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, t2);
#endif
   _STORE(&q[offset],q2);
   q3 = _LOAD(&q[2*offset]);
#ifdef BLOCK2
   q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, y3);
#endif
#ifdef BLOCK4
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, w3);
#endif
#ifdef BLOCK6
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, t3);
#endif
   _STORE(&q[2*offset],q3);
   q4 = _LOAD(&q[3*offset]);
#ifdef BLOCK2
   q4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, y4);
#endif
#ifdef BLOCK4
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, w4);
#endif
#ifdef BLOCK6
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, t4);
#endif
   _STORE(&q[3*offset],q4);
   q5 = _LOAD(&q[4*offset]);
#ifdef BLOCK2
   q5 = _SIMD_ADD( ADDITIONAL_ARGUMENT q5, y5);
#endif
#ifdef BLOCK4
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, w5);
#endif
#ifdef BLOCK6
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, t5);
#endif
   _STORE(&q[4*offset],q5);
   q6 = _LOAD(&q[5*offset]);
#ifdef BLOCK2
   q6 = _SIMD_ADD( ADDITIONAL_ARGUMENT q6, y6);
#endif
#ifdef BLOCK4
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, w6);
#endif
#ifdef BLOCK6
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, t6);
#endif
   _STORE(&q[5*offset],q6);

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif

   q1 = _LOAD(&q[ldq]);
   q2 = _LOAD(&q[ldq+offset]);
   q3 = _LOAD(&q[ldq+2*offset]);
   q4 = _LOAD(&q[ldq+3*offset]);
   q5 = _LOAD(&q[ldq+4*offset]);
   q6 = _LOAD(&q[ldq+5*offset]);
#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_FMA(y1, h2, x1));
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_FMA(y2, h2, x2));
   q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_FMA(y3, h2, x3));
   q4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, _SIMD_FMA(y4, h2, x4));
   q5 = _SIMD_ADD( ADDITIONAL_ARGUMENT q5, _SIMD_FMA(y5, h2, x5));
   q6 = _SIMD_ADD( ADDITIONAL_ARGUMENT q6, _SIMD_FMA(y6, h2, x6));
#else
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2)));
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2)));
   q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2)));
   q4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2)));
   q5 = _SIMD_ADD( ADDITIONAL_ARGUMENT q5, _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT y5, h2)));
   q6 = _SIMD_ADD( ADDITIONAL_ARGUMENT q6, _SIMD_ADD( ADDITIONAL_ARGUMENT x6, _SIMD_MUL( ADDITIONAL_ARGUMENT y6, h2)));
#endif /* __ELPA_USE_FMA__ */
   _STORE(&q[ldq],q1);
   _STORE(&q[ldq+offset],q2);
   _STORE(&q[ldq+2*offset],q3);
   _STORE(&q[ldq+3*offset],q4);
   _STORE(&q[ldq+4*offset],q5);
   _STORE(&q[ldq+5*offset],q6);
#endif /* BLOCK2 */

#ifdef BLOCK4
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
#endif

   q1 = _LOAD(&q[ldq]);
   q2 = _LOAD(&q[ldq+offset]);
   q3 = _LOAD(&q[ldq+2*offset]);
   q4 = _LOAD(&q[ldq+3*offset]);
   q5 = _LOAD(&q[ldq+4*offset]);
   q6 = _LOAD(&q[ldq+5*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_FMA(w1, h4, z1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_FMA(w2, h4, z2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_FMA(w3, h4, z3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_FMA(w4, h4, z4));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_FMA(w5, h4, z5));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_FMA(w6, h4, z6));
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4)));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4)));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4)));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4)));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_ADD( ADDITIONAL_ARGUMENT z5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5, h4)));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_ADD( ADDITIONAL_ARGUMENT z6, _SIMD_MUL( ADDITIONAL_ARGUMENT w6, h4)));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq],q1);
   _STORE(&q[ldq+offset],q2);
   _STORE(&q[ldq+2*offset],q3);
   _STORE(&q[ldq+3*offset],q4);
   _STORE(&q[ldq+4*offset],q5);
   _STORE(&q[ldq+5*offset],q6);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+2]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
#endif
   q1 = _LOAD(&q[ldq*2]);
   q2 = _LOAD(&q[(ldq*2)+offset]);
   q3 = _LOAD(&q[(ldq*2)+2*offset]);
   q4 = _LOAD(&q[(ldq*2)+3*offset]);
   q5 = _LOAD(&q[(ldq*2)+4*offset]);
   q6 = _LOAD(&q[(ldq*2)+5*offset]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, y1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, y2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, y3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, y4);
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, y5);
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, y6);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
   q5 = _SIMD_NFMA(w5, h4, q5);
   q6 = _SIMD_NFMA(w6, h4, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5, h4));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT w6, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
#endif
 
#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
   q5 = _SIMD_NFMA(z5, h3, q5);
   q6 = _SIMD_NFMA(z6, h3, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5, h3));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT z6, h3));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*2],q1);
   _STORE(&q[(ldq*2)+offset],q2);
   _STORE(&q[(ldq*2)+2*offset],q3);
   _STORE(&q[(ldq*2)+3*offset],q4);
   _STORE(&q[(ldq*2)+4*offset],q5);
   _STORE(&q[(ldq*2)+5*offset],q6);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

   q1 = _LOAD(&q[ldq*3]);
   q2 = _LOAD(&q[(ldq*3)+offset]);
   q3 = _LOAD(&q[(ldq*3)+2*offset]);
   q4 = _LOAD(&q[(ldq*3)+3*offset]);
   q5 = _LOAD(&q[(ldq*3)+4*offset]);
   q6 = _LOAD(&q[(ldq*3)+5*offset]);

   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, x1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, x2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, x3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, x4);
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, x5);
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, x6);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
   q5 = _SIMD_NFMA(w5, h4, q5);
   q6 = _SIMD_NFMA(w6, h4, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5, h4));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT w6, h4));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
   q5 = _SIMD_NFMA(y5, h2, q5);
   q6 = _SIMD_NFMA(y6, h2, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT y5, h2));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT y6, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
   q5 = _SIMD_NFMA(z5, h3, q5);
   q6 = _SIMD_NFMA(z6, h3, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5, h3));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT z6, h3));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*3], q1);
   _STORE(&q[(ldq*3)+offset], q2);
   _STORE(&q[(ldq*3)+2*offset], q3);
   _STORE(&q[(ldq*3)+3*offset], q4);
   _STORE(&q[(ldq*3)+4*offset], q5);
   _STORE(&q[(ldq*3)+5*offset], q6);

#endif /* BLOCK4 */

#ifdef BLOCK6
   
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+1]);
#endif

   q1 = _LOAD(&q[ldq]);
   q2 = _LOAD(&q[(ldq+offset)]);
   q3 = _LOAD(&q[(ldq+2*offset)]);
   q4 = _LOAD(&q[(ldq+3*offset)]);
   q5 = _LOAD(&q[(ldq+4*offset)]);
   q6 = _LOAD(&q[(ldq+5*offset)]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, v1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, v2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, v3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, v4);
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, v5);
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, v6);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
   q4 = _SIMD_NFMA(t4, h6, q4);
   q5 = _SIMD_NFMA(t5, h6, q5);
   q6 = _SIMD_NFMA(t6, h6, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT t5, h6));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT t6, h6));
#endif

   _STORE(&q[ldq],q1);
   _STORE(&q[(ldq+offset)],q2);
   _STORE(&q[(ldq+2*offset)],q3);
   _STORE(&q[(ldq+3*offset)],q4);
   _STORE(&q[(ldq+4*offset)],q5);
   _STORE(&q[(ldq+5*offset)],q6);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+1]);
#endif

   q1 = _LOAD(&q[ldq*2]);
   q2 = _LOAD(&q[(ldq*2)+offset]);
   q3 = _LOAD(&q[(ldq*2)+2*offset]);
   q4 = _LOAD(&q[(ldq*2)+3*offset]);
   q5 = _LOAD(&q[(ldq*2)+4*offset]);
   q6 = _LOAD(&q[(ldq*2)+5*offset]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, w1); 
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, w2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, w3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, w4);
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, w5);
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, w6);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
   q4 = _SIMD_NFMA(v4, h5, q4);
   q5 = _SIMD_NFMA(v5, h5, q5);
   q6 = _SIMD_NFMA(v6, h5, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5)); 
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));  
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));  
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));  
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT v5, h5));  
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT v6, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
   q4 = _SIMD_NFMA(t4, h6, q4);
   q5 = _SIMD_NFMA(t5, h6, q5);
   q6 = _SIMD_NFMA(t6, h6, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT t5, h6));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT t6, h6));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*2],q1);
   _STORE(&q[(ldq*2)+offset],q2);
   _STORE(&q[(ldq*2)+2*offset],q3);
   _STORE(&q[(ldq*2)+3*offset],q4);
   _STORE(&q[(ldq*2)+4*offset],q5);
   _STORE(&q[(ldq*2)+5*offset],q6);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
#endif

   q1 = _LOAD(&q[ldq*3]);
   q2 = _LOAD(&q[(ldq*3)+offset]);
   q3 = _LOAD(&q[(ldq*3)+2*offset]);
   q4 = _LOAD(&q[(ldq*3)+3*offset]);
   q5 = _LOAD(&q[(ldq*3)+4*offset]);
   q6 = _LOAD(&q[(ldq*3)+5*offset]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, z1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, z2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, z3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, z4);
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, z5);
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, z6);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
   q5 = _SIMD_NFMA(w5, h4, q5);
   q6 = _SIMD_NFMA(w6, h4, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5, h4));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT w6, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
   q4 = _SIMD_NFMA(v4, h5, q4);
   q5 = _SIMD_NFMA(v5, h5, q5);
   q6 = _SIMD_NFMA(v6, h5, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT v5, h5));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT v6, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
   q4 = _SIMD_NFMA(t4, h6, q4);
   q5 = _SIMD_NFMA(t5, h6, q5);
   q6 = _SIMD_NFMA(t6, h6, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT t5, h6));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT t6, h6));
#endif

   _STORE(&q[ldq*3],q1);
   _STORE(&q[(ldq*3)+offset],q2);
   _STORE(&q[(ldq*3)+2*offset],q3);
   _STORE(&q[(ldq*3)+3*offset],q4);
   _STORE(&q[(ldq*3)+4*offset],q5);
   _STORE(&q[(ldq*3)+5*offset],q6);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
#endif
   q1 = _LOAD(&q[ldq*4]);
   q2 = _LOAD(&q[(ldq*4)+offset]);
   q3 = _LOAD(&q[(ldq*4)+2*offset]);
   q4 = _LOAD(&q[(ldq*4)+3*offset]);
   q5 = _LOAD(&q[(ldq*4)+4*offset]);
   q6 = _LOAD(&q[(ldq*4)+5*offset]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, y1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, y2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, y3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, y4);
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, y5);
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, y6);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
   q5 = _SIMD_NFMA(z5, h3, q5);
   q6 = _SIMD_NFMA(z6, h3, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5, h3));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT z6, h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
   q5 = _SIMD_NFMA(w5, h4, q5);
   q6 = _SIMD_NFMA(w6, h4, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5, h4));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT w6, h4));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
   q4 = _SIMD_NFMA(v4, h5, q4);
   q5 = _SIMD_NFMA(v5, h5, q5);
   q6 = _SIMD_NFMA(v6, h5, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT v5, h5));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT v6, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+4]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+4]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
   q4 = _SIMD_NFMA(t4, h6, q4);
   q5 = _SIMD_NFMA(t5, h6, q5);
   q6 = _SIMD_NFMA(t6, h6, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT t5, h6));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT t6, h6));
#endif

   _STORE(&q[ldq*4],q1);
   _STORE(&q[(ldq*4)+offset],q2);
   _STORE(&q[(ldq*4)+2*offset],q3);
   _STORE(&q[(ldq*4)+3*offset],q4);
   _STORE(&q[(ldq*4)+4*offset],q5);
   _STORE(&q[(ldq*4)+5*offset],q6);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[(ldh)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[(ldh)+1], hh[(ldh)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[(ldh)+1]);
#endif
   q1 = _LOAD(&q[ldq*5]);
   q2 = _LOAD(&q[(ldq*5)+offset]);
   q3 = _LOAD(&q[(ldq*5)+2*offset]);
   q4 = _LOAD(&q[(ldq*5)+3*offset]);
   q5 = _LOAD(&q[(ldq*5)+4*offset]);
   q6 = _LOAD(&q[(ldq*5)+5*offset]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, x1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, x2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, x3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, x4);
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, x5);
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, x6);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
   q5 = _SIMD_NFMA(y5, h2, q5);
   q6 = _SIMD_NFMA(y6, h2, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT y5, h2));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT y6, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
   q5 = _SIMD_NFMA(z5, h3, q5);
   q6 = _SIMD_NFMA(z6, h3, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5, h3));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT z6, h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
   q5 = _SIMD_NFMA(w5, h4, q5);
   q6 = _SIMD_NFMA(w6, h4, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5, h4));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT w6, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+4]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+4]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
   q4 = _SIMD_NFMA(v4, h5, q4);
   q5 = _SIMD_NFMA(v5, h5, q5);
   q6 = _SIMD_NFMA(v6, h5, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT v5, h5));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT v6, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+5]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+5]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
   q4 = _SIMD_NFMA(t4, h6, q4);
   q5 = _SIMD_NFMA(t5, h6, q5);
   q6 = _SIMD_NFMA(t6, h6, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT t5, h6));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT t6, h6));
#endif

   _STORE(&q[ldq*5],q1);
   _STORE(&q[(ldq*5)+offset],q2);
   _STORE(&q[(ldq*5)+2*offset],q3);
   _STORE(&q[(ldq*5)+3*offset],q4);
   _STORE(&q[(ldq*5)+4*offset],q5);
   _STORE(&q[(ldq*5)+5*offset],q6);

#endif /* BLOCK6 */

   for (i = BLOCK; i < nb; i++)
   {
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h1 = _SIMD_SET1(hh[i-(BLOCK-1)]);
     h2 = _SIMD_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == SPARC64_SSE
     h1 = _SIMD_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
     h2 = _SIMD_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[i-(BLOCK-1)]);
    h2 = _SIMD_BROADCAST(&hh[ldh+i-(BLOCK-2)]);
#endif

     q1 = _LOAD(&q[i*ldq]);
     q2 = _LOAD(&q[(i*ldq)+offset]);
     q3 = _LOAD(&q[(i*ldq)+2*offset]);
     q4 = _LOAD(&q[(i*ldq)+3*offset]);
     q5 = _LOAD(&q[(i*ldq)+4*offset]);
     q6 = _LOAD(&q[(i*ldq)+5*offset]);

#ifdef BLOCK2
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_FMA(x1, h1, q1);
     q1 = _SIMD_FMA(y1, h2, q1);
     q2 = _SIMD_FMA(x2, h1, q2);
     q2 = _SIMD_FMA(y2, h2, q2);
     q3 = _SIMD_FMA(x3, h1, q3);
     q3 = _SIMD_FMA(y3, h2, q3);
     q4 = _SIMD_FMA(x4, h1, q4);
     q4 = _SIMD_FMA(y4, h2, q4);
     q5 = _SIMD_FMA(x5, h1, q5);
     q5 = _SIMD_FMA(y5, h2, q5);
     q6 = _SIMD_FMA(x6, h1, q6);
     q6 = _SIMD_FMA(y6, h2, q6);
#else
     q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2)));
     q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2)));
     q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2)));
     q4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2)));
     q5 = _SIMD_ADD( ADDITIONAL_ARGUMENT q5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y5, h2)));
     q6 = _SIMD_ADD( ADDITIONAL_ARGUMENT q6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x6,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y6, h2)));
#endif /* __ELPA_USE_FMA__ */
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
               
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(x1, h1, q1);
     q2 = _SIMD_NFMA(x2, h1, q2);
     q3 = _SIMD_NFMA(x3, h1, q3);
     q4 = _SIMD_NFMA(x4, h1, q4);
     q5 = _SIMD_NFMA(x5, h1, q5);
     q6 = _SIMD_NFMA(x6, h1, q6);
#else  
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h1));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h1));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h1));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h1));
     q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h1));
     q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT x6,h1));
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(y1, h2, q1);
     q2 = _SIMD_NFMA(y2, h2, q2);
     q3 = _SIMD_NFMA(y3, h2, q3);
     q4 = _SIMD_NFMA(y4, h2, q4);
     q5 = _SIMD_NFMA(y5, h2, q5);
     q6 = _SIMD_NFMA(y6, h2, q6);
#else   
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h2));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h2));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h2));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h2));
     q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT y5,h2));
     q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT y6,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h3 = _SIMD_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
     h3 = _SIMD_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h3 = _SIMD_BROADCAST(&hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(z1, h3, q1);
     q2 = _SIMD_NFMA(z2, h3, q2);
     q3 = _SIMD_NFMA(z3, h3, q3);
     q4 = _SIMD_NFMA(z4, h3, q4);
     q5 = _SIMD_NFMA(z5, h3, q5);
     q6 = _SIMD_NFMA(z6, h3, q6);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h3));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h3));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h3));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h3));
     q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5,h3));
     q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT z6,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h4 = _SIMD_SET1(hh[(ldh*3)+i-(BLOCK-4)]); 
#endif

#if VEC_SET == SPARC64_SSE
     h4 = _SIMD_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h4 = _SIMD_BROADCAST(&hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(w1, h4, q1);
     q2 = _SIMD_NFMA(w2, h4, q2);
     q3 = _SIMD_NFMA(w3, h4, q3);
     q4 = _SIMD_NFMA(w4, h4, q4);
     q5 = _SIMD_NFMA(w5, h4, q5);
     q6 = _SIMD_NFMA(w6, h4, q6);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h4));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h4));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3,h4));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4,h4));
     q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5,h4));
     q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT w6,h4));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 || BLOCK6  */

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h5 = _SIMD_SET1(hh[(ldh*4)+i-1]);
#endif
#if VEC_SET == SPARC64_SSE
     h5 = _SIMD_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h5 = _SIMD_BROADCAST(&hh[(ldh*4)+i-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(v1, h5, q1);
     q2 = _SIMD_NFMA(v2, h5, q2);
     q3 = _SIMD_NFMA(v3, h5, q3);
     q4 = _SIMD_NFMA(v4, h5, q4);
     q5 = _SIMD_NFMA(v5, h5, q5);
     q6 = _SIMD_NFMA(v6, h5, q6);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));
     q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT v5, h5));
     q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT v6, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h6 = _SIMD_SET1(hh[(ldh*5)+i]);
#endif
#if VEC_SET == SPARC64_SSE
     h6 = _SIMD_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h6 = _SIMD_BROADCAST(&hh[(ldh*5)+i-(BLOCK-6)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(t1, h6, q1);
     q2 = _SIMD_NFMA(t2, h6, q2);
     q3 = _SIMD_NFMA(t3, h6, q3);
     q4 = _SIMD_NFMA(t4, h6, q4);
     q5 = _SIMD_NFMA(t5, h6, q5);
     q6 = _SIMD_NFMA(t6, h6, q6);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
     q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT t5, h6));
     q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT t6, h6));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

     _STORE(&q[i*ldq],q1);
     _STORE(&q[(i*ldq)+offset],q2);
     _STORE(&q[(i*ldq)+2*offset],q3);
     _STORE(&q[(i*ldq)+3*offset],q4);
     _STORE(&q[(i*ldq)+4*offset],q5);
     _STORE(&q[(i*ldq)+5*offset],q6);

   }
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-1)]);
#endif

   q1 = _LOAD(&q[nb*ldq]);
   q2 = _LOAD(&q[(nb*ldq)+offset]);
   q3 = _LOAD(&q[(nb*ldq)+2*offset]);
   q4 = _LOAD(&q[(nb*ldq)+3*offset]);
   q5 = _LOAD(&q[(nb*ldq)+4*offset]);
   q6 = _LOAD(&q[(nb*ldq)+5*offset]);

#ifdef BLOCK2

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_FMA(x1, h1, q1);
   q2 = _SIMD_FMA(x2, h1, q2);
   q3 = _SIMD_FMA(x3, h1, q3);
   q4 = _SIMD_FMA(x4, h1, q4);
   q5 = _SIMD_FMA(x5, h1, q5);
   q6 = _SIMD_FMA(x6, h1, q6);
#else
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
   q5 = _SIMD_ADD( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT x5, h1));
   q6 = _SIMD_ADD( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT x6, h1));
#endif
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
   q4 = _SIMD_NFMA(x4, h1, q4);
   q5 = _SIMD_NFMA(x5, h1, q5);
   q6 = _SIMD_NFMA(x6, h1, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT x5, h1));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT x6, h1));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
   q5 = _SIMD_NFMA(y5, h2, q5);
   q6 = _SIMD_NFMA(y6, h2, q6);
#else   
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT y5, h2));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
   q5 = _SIMD_NFMA(z5, h3, q5);
   q6 = _SIMD_NFMA(z6, h3, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5, h3));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT z6, h3));
#endif

#endif /* BLOCK4 || BLOCK6  */

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
   q5 = _SIMD_NFMA(w5, h4, q5);
   q6 = _SIMD_NFMA(w6, h4, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5, h4));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT w6, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
   q4 = _SIMD_NFMA(v4, h5, q4);
   q5 = _SIMD_NFMA(v5, h5, q5);
   q6 = _SIMD_NFMA(v6, h5, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT v5, h5));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT v6, h5));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

   _STORE(&q[nb*ldq],q1);
   _STORE(&q[(nb*ldq)+offset],q2);
   _STORE(&q[(nb*ldq)+2*offset],q3);
   _STORE(&q[(nb*ldq)+3*offset],q4);
   _STORE(&q[(nb*ldq)+4*offset],q5);
   _STORE(&q[(nb*ldq)+5*offset],q6);

#if defined(BLOCK4) || defined(BLOCK6)
   
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-2)], hh[nb-(BLOCK-2)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-2)]);
#endif

   q1 = _LOAD(&q[(nb+1)*ldq]);
   q2 = _LOAD(&q[((nb+1)*ldq)+offset]);
   q3 = _LOAD(&q[((nb+1)*ldq)+2*offset]);
   q4 = _LOAD(&q[((nb+1)*ldq)+3*offset]);
   q5 = _LOAD(&q[((nb+1)*ldq)+4*offset]);
   q6 = _LOAD(&q[((nb+1)*ldq)+5*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
   q4 = _SIMD_NFMA(x4, h1, q4);
   q5 = _SIMD_NFMA(x5, h1, q5);
   q6 = _SIMD_NFMA(x6, h1, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT x5, h1));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT x6, h1));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-3)], hh[ldh+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
   q5 = _SIMD_NFMA(y5, h2, q5);
   q6 = _SIMD_NFMA(y6, h2, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT y5, h2));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT y6, h2));
#endif

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
   q5 = _SIMD_NFMA(z5, h3, q5);
   q6 = _SIMD_NFMA(z6, h3, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5, h3));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT z6, h3));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
   q5 = _SIMD_NFMA(w5, h4, q5);
   q6 = _SIMD_NFMA(w6, h4, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5, h4));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT w6, h4));
#endif /* __ELPA_USE_FMA__ */
#endif /* BLOCK6 */

   _STORE(&q[(nb+1)*ldq],q1);
   _STORE(&q[((nb+1)*ldq)+offset],q2);
   _STORE(&q[((nb+1)*ldq)+2*offset],q3);
   _STORE(&q[((nb+1)*ldq)+3*offset],q4);
   _STORE(&q[((nb+1)*ldq)+4*offset],q5);
   _STORE(&q[((nb+1)*ldq)+5*offset],q6);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-3)], hh[nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-3)]);
#endif

   q1 = _LOAD(&q[(nb+2)*ldq]);
   q2 = _LOAD(&q[((nb+2)*ldq)+offset]);
   q3 = _LOAD(&q[((nb+2)*ldq)+2*offset]);
   q4 = _LOAD(&q[((nb+2)*ldq)+3*offset]);
   q5 = _LOAD(&q[((nb+2)*ldq)+4*offset]);
   q6 = _LOAD(&q[((nb+2)*ldq)+5*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
   q4 = _SIMD_NFMA(x4, h1, q4);
   q5 = _SIMD_NFMA(x5, h1, q5);
   q6 = _SIMD_NFMA(x6, h1, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT x5, h1));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT x6, h1));
#endif

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
   q5 = _SIMD_NFMA(y5, h2, q5);
   q6 = _SIMD_NFMA(y6, h2, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT y5, h2));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT y6, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
   q5 = _SIMD_NFMA(z5, h3, q5);
   q6 = _SIMD_NFMA(z6, h3, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5, h3));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT z6, h3));
#endif /* __ELPA_USE_FMA__ */
 
#endif /* BLOCK6 */

   _STORE(&q[(nb+2)*ldq],q1);
   _STORE(&q[((nb+2)*ldq)+offset],q2);
   _STORE(&q[((nb+2)*ldq)+2*offset],q3);
   _STORE(&q[((nb+2)*ldq)+3*offset],q4);
   _STORE(&q[((nb+2)*ldq)+4*offset],q5);
   _STORE(&q[((nb+2)*ldq)+5*offset],q6);

#endif /* BLOCK4 || BLOCK6  */

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-2], hh[nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-2]);
#endif

   q1 = _LOAD(&q[(nb+3)*ldq]);
   q2 = _LOAD(&q[((nb+3)*ldq)+offset]);
   q3 = _LOAD(&q[((nb+3)*ldq)+2*offset]);
   q4 = _LOAD(&q[((nb+3)*ldq)+3*offset]);
   q5 = _LOAD(&q[((nb+3)*ldq)+4*offset]);
   q6 = _LOAD(&q[((nb+3)*ldq)+5*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
   q4 = _SIMD_NFMA(x4, h1, q4);
   q5 = _SIMD_NFMA(x5, h1, q5);
   q6 = _SIMD_NFMA(x6, h1, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT x5, h1));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT x6, h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
   q5 = _SIMD_NFMA(y5, h2, q5);
   q6 = _SIMD_NFMA(y6, h2, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT y5, h2));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT y6, h2));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[(nb+3)*ldq],q1);
   _STORE(&q[((nb+3)*ldq)+offset],q2);
   _STORE(&q[((nb+3)*ldq)+2*offset],q3);
   _STORE(&q[((nb+3)*ldq)+3*offset],q4);
   _STORE(&q[((nb+3)*ldq)+4*offset],q5);
   _STORE(&q[((nb+3)*ldq)+5*offset],q6);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-1], hh[nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-1]);
#endif

   q1 = _LOAD(&q[(nb+4)*ldq]);
   q2 = _LOAD(&q[((nb+4)*ldq)+offset]);
   q3 = _LOAD(&q[((nb+4)*ldq)+2*offset]);
   q4 = _LOAD(&q[((nb+4)*ldq)+3*offset]);
   q5 = _LOAD(&q[((nb+4)*ldq)+4*offset]);
   q6 = _LOAD(&q[((nb+4)*ldq)+5*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
   q4 = _SIMD_NFMA(x4, h1, q4);
   q5 = _SIMD_NFMA(x5, h1, q5);
   q6 = _SIMD_NFMA(x6, h1, q6);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT x5, h1));
   q6 = _SIMD_SUB( ADDITIONAL_ARGUMENT q6, _SIMD_MUL( ADDITIONAL_ARGUMENT x6, h1));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[(nb+4)*ldq],q1);
   _STORE(&q[((nb+4)*ldq)+offset],q2);
   _STORE(&q[((nb+4)*ldq)+2*offset],q3);
   _STORE(&q[((nb+4)*ldq)+3*offset],q4);
   _STORE(&q[((nb+4)*ldq)+4*offset],q5);
   _STORE(&q[((nb+4)*ldq)+5*offset],q6);

#endif /* BLOCK6 */
}

#undef ROW_LENGTH
#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 10
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 20
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 20
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 40
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 40
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 80
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */

/*
 * Unrolled kernel that computes
 * ROW_LENGTH rows of Q simultaneously, a
 * matrix Vector product with two householder
 */
#ifdef BLOCK2
/*
 * vectors + a rank 2 update is performed
 */
#endif
#ifdef BLOCK4
/*
 * vectors + a rank 1 update is performed
 */
#endif
__forceinline void CONCAT_8ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq, int ldh, 
#ifdef BLOCK2
               DATA_TYPE s)
#endif
#ifdef BLOCK4
               DATA_TYPE s_1_2, DATA_TYPE s_1_3, DATA_TYPE s_2_3, DATA_TYPE s_1_4, DATA_TYPE s_2_4, DATA_TYPE s_3_4)
#endif
#ifdef BLOCK6
               DATA_TYPE_PTR scalarprods)
#endif
  {
#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Matrix Vector Multiplication, Q [ ROW_LENGTH x nb+1] * hh
    // hh contains two householder vectors, with offset 1
    /////////////////////////////////////////////////////
#endif
#if defined(BLOCK4) || defined(BLOCK6)
    /////////////////////////////////////////////////////
    // Matrix Vector Multiplication, Q [ ROW_LENGTH x nb+3] * hh
    // hh contains four householder vectors
    /////////////////////////////////////////////////////
#endif

    int i;
#ifdef BLOCK2
#if VEC_SET == SSE_128
    // Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == VSX_SSE
    __SIMD_DATATYPE sign = vec_splats(MONE);
#endif

#if VEC_SET == NEON_ARCH64_128
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = vdupq_n_f64(MONE);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = vdupq_n_f32(MONE);
#endif
#endif

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set1_epi64x(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if  VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_512 */

#if VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = svdup_f64(MONE);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = svdup_f32(MONE);
#endif
#endif

    __SIMD_DATATYPE x1 = _LOAD(&q[ldq]);
    __SIMD_DATATYPE x2 = _LOAD(&q[ldq+offset]);
    __SIMD_DATATYPE x3 = _LOAD(&q[ldq+2*offset]);
    __SIMD_DATATYPE x4 = _LOAD(&q[ldq+3*offset]);
    __SIMD_DATATYPE x5 = _LOAD(&q[ldq+4*offset]);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h1 = _SIMD_SET1(hh[ldh+1]);
#endif
#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h1 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif
    __SIMD_DATATYPE h2;

#ifdef __ELPA_USE_FMA__
    __SIMD_DATATYPE q1 = _LOAD(q);
    __SIMD_DATATYPE y1 = _SIMD_FMA(x1, h1, q1);
    __SIMD_DATATYPE q2 = _LOAD(&q[offset]);
    __SIMD_DATATYPE y2 = _SIMD_FMA(x2, h1, q2);
    __SIMD_DATATYPE q3 = _LOAD(&q[2*offset]);
    __SIMD_DATATYPE y3 = _SIMD_FMA(x3, h1, q3);
    __SIMD_DATATYPE q4 = _LOAD(&q[3*offset]);
    __SIMD_DATATYPE y4 = _SIMD_FMA(x4, h1, q4);
    __SIMD_DATATYPE q5 = _LOAD(&q[4*offset]);
    __SIMD_DATATYPE y5 = _SIMD_FMA(x5, h1, q5);
#else
    __SIMD_DATATYPE q1 = _LOAD(q);
    __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
    __SIMD_DATATYPE q2 = _LOAD(&q[offset]);
    __SIMD_DATATYPE y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
    __SIMD_DATATYPE q3 = _LOAD(&q[2*offset]);
    __SIMD_DATATYPE y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
    __SIMD_DATATYPE q4 = _LOAD(&q[3*offset]);
    __SIMD_DATATYPE y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
    __SIMD_DATATYPE q5 = _LOAD(&q[4*offset]);
    __SIMD_DATATYPE y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT x5, h1));
#endif
#endif /* BLOCK2 */

#ifdef BLOCK4
    __SIMD_DATATYPE a1_1 = _LOAD(&q[ldq*3]);
    __SIMD_DATATYPE a2_1 = _LOAD(&q[ldq*2]);
    __SIMD_DATATYPE a3_1 = _LOAD(&q[ldq]);  
    __SIMD_DATATYPE a4_1 = _LOAD(&q[0]);    

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_2_1 = _SIMD_SET1(hh[ldh+1]);    
    __SIMD_DATATYPE h_3_2 = _SIMD_SET1(hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET1(hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_SET1(hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET1(hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_2_1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_2_1 = _SIMD_BROADCAST(&hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w1 = _SIMD_FMA(a3_1, h_4_3, a4_1);
    w1 = _SIMD_FMA(a2_1, h_4_2, w1);
    w1 = _SIMD_FMA(a1_1, h_4_1, w1);
    register __SIMD_DATATYPE z1 = _SIMD_FMA(a2_1, h_3_2, a3_1);
    z1 = _SIMD_FMA(a1_1, h_3_1, z1);
    register __SIMD_DATATYPE y1 = _SIMD_FMA(a1_1, h_2_1, a2_1);
    register __SIMD_DATATYPE x1 = a1_1;
#else
    register __SIMD_DATATYPE w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_4_3));
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_4_2));                          
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_4_1));                          
    register __SIMD_DATATYPE z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_3_2));
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_3_1));                          
    register __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_2_1));
    register __SIMD_DATATYPE x1 = a1_1; 
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE a1_2 = _LOAD(&q[(ldq*3)+offset]);                  
    __SIMD_DATATYPE a2_2 = _LOAD(&q[(ldq*2)+offset]);
    __SIMD_DATATYPE a3_2 = _LOAD(&q[ldq+offset]);
    __SIMD_DATATYPE a4_2 = _LOAD(&q[0+offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w2 = _SIMD_FMA(a3_2, h_4_3, a4_2);
    w2 = _SIMD_FMA(a2_2, h_4_2, w2);
    w2 = _SIMD_FMA(a1_2, h_4_1, w2);
    register __SIMD_DATATYPE z2 = _SIMD_FMA(a2_2, h_3_2, a3_2);
    z2 = _SIMD_FMA(a1_2, h_3_1, z2);
    register __SIMD_DATATYPE y2 = _SIMD_FMA(a1_2, h_2_1, a2_2);
    register __SIMD_DATATYPE x2 = a1_2;
#else
    register __SIMD_DATATYPE w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_4_3));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_4_2));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_4_1));
    register __SIMD_DATATYPE z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_3_2));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_3_1));
    register __SIMD_DATATYPE y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_2_1));
    register __SIMD_DATATYPE x2 = a1_2;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE a1_3 = _LOAD(&q[(ldq*3)+2*offset]);
    __SIMD_DATATYPE a2_3 = _LOAD(&q[(ldq*2)+2*offset]);
    __SIMD_DATATYPE a3_3 = _LOAD(&q[ldq+2*offset]);
    __SIMD_DATATYPE a4_3 = _LOAD(&q[0+2*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w3 = _SIMD_FMA(a3_3, h_4_3, a4_3);
    w3 = _SIMD_FMA(a2_3, h_4_2, w3);
    w3 = _SIMD_FMA(a1_3, h_4_1, w3);
    register __SIMD_DATATYPE z3 = _SIMD_FMA(a2_3, h_3_2, a3_3);
    z3 = _SIMD_FMA(a1_3, h_3_1, z3);
    register __SIMD_DATATYPE y3 = _SIMD_FMA(a1_3, h_2_1, a2_3);
    register __SIMD_DATATYPE x3 = a1_3;
#else
    register __SIMD_DATATYPE w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_3, h_4_3));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_4_2));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_4_1));
    register __SIMD_DATATYPE z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_3_2));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_3_1));
    register __SIMD_DATATYPE y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_2_1));
    register __SIMD_DATATYPE x3 = a1_3;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE a1_4 = _LOAD(&q[(ldq*3)+3*offset]);
    __SIMD_DATATYPE a2_4 = _LOAD(&q[(ldq*2)+3*offset]);
    __SIMD_DATATYPE a3_4 = _LOAD(&q[ldq+3*offset]);
    __SIMD_DATATYPE a4_4 = _LOAD(&q[0+3*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w4 = _SIMD_FMA(a3_4, h_4_3, a4_4);
    w4 = _SIMD_FMA(a2_4, h_4_2, w4);
    w4 = _SIMD_FMA(a1_4, h_4_1, w4);
    register __SIMD_DATATYPE z4 = _SIMD_FMA(a2_4, h_3_2, a3_4);
    z4 = _SIMD_FMA(a1_4, h_3_1, z4);
    register __SIMD_DATATYPE y4 = _SIMD_FMA(a1_4, h_2_1, a2_4);
    register __SIMD_DATATYPE x4 = a1_4;
#else
    register __SIMD_DATATYPE w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_4, h_4_3));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_4_2));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_4_1));
    register __SIMD_DATATYPE z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_3_2));
    z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_3_1));
    register __SIMD_DATATYPE y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_2_1));
    register __SIMD_DATATYPE x4 = a1_4;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE a1_5 = _LOAD(&q[(ldq*3)+4*offset]);
    __SIMD_DATATYPE a2_5 = _LOAD(&q[(ldq*2)+4*offset]);
    __SIMD_DATATYPE a3_5 = _LOAD(&q[ldq+4*offset]);
    __SIMD_DATATYPE a4_5 = _LOAD(&q[0+4*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w5 = _SIMD_FMA(a3_5, h_4_3, a4_5);
    w5 = _SIMD_FMA(a2_5, h_4_2, w5);
    w5 = _SIMD_FMA(a1_5, h_4_1, w5);
    register __SIMD_DATATYPE z5 = _SIMD_FMA(a2_5, h_3_2, a3_5);
    z5 = _SIMD_FMA(a1_5, h_3_1, z5);
    register __SIMD_DATATYPE y5 = _SIMD_FMA(a1_5, h_2_1, a2_5);
    register __SIMD_DATATYPE x5 = a1_5;
#else
    register __SIMD_DATATYPE w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_5, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_5, h_4_3));
    w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT w5, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_5, h_4_2));
    w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT w5, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_5, h_4_1));
    register __SIMD_DATATYPE z5 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_5, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_5, h_3_2));
    z5 = _SIMD_ADD( ADDITIONAL_ARGUMENT z5, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_5, h_3_1));
    register __SIMD_DATATYPE y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_5, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_5, h_2_1));
    register __SIMD_DATATYPE x5 = a1_5;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE q1;
    __SIMD_DATATYPE q2;
    __SIMD_DATATYPE q3;
    __SIMD_DATATYPE q4;
    __SIMD_DATATYPE q5;

    __SIMD_DATATYPE h1;
    __SIMD_DATATYPE h2;
    __SIMD_DATATYPE h3;
    __SIMD_DATATYPE h4;
#endif /* BLOCK4 */

#ifdef BLOCK6
    
    __SIMD_DATATYPE a1_1 = _LOAD(&q[ldq*5]);
    __SIMD_DATATYPE a2_1 = _LOAD(&q[ldq*4]);
    __SIMD_DATATYPE a3_1 = _LOAD(&q[ldq*3]);
    __SIMD_DATATYPE a4_1 = _LOAD(&q[ldq*2]);
    __SIMD_DATATYPE a5_1 = _LOAD(&q[ldq]);  
    __SIMD_DATATYPE a6_1 = _LOAD(&q[0]);    

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_6_5 = _SIMD_SET1(hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_SET1(hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_SET1(hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_SET1(hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_SET1(hh[(ldh*5)+5]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_6_5 = _SIMD_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_6_5 = _SIMD_BROADCAST(&hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_BROADCAST(&hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_BROADCAST(&hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_BROADCAST(&hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_BROADCAST(&hh[(ldh*5)+5]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t1 = _SIMD_FMA(a5_1, h_6_5, a6_1);
    t1 = _SIMD_FMA(a4_1, h_6_4, t1);
    t1 = _SIMD_FMA(a3_1, h_6_3, t1);
    t1 = _SIMD_FMA(a2_1, h_6_2, t1);
    t1 = _SIMD_FMA(a1_1, h_6_1, t1);
#else
    register __SIMD_DATATYPE t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_1, h_6_5)); 
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_1, h_6_4));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_6_3));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_6_2));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_6_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_5_4 = _SIMD_SET1(hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_SET1(hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_SET1(hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_SET1(hh[(ldh*4)+4]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_5_4 = _SIMD_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif


#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_5_4 = _SIMD_BROADCAST(&hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_BROADCAST(&hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_BROADCAST(&hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_BROADCAST(&hh[(ldh*4)+4]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE v1 = _SIMD_FMA(a4_1, h_5_4, a5_1);
    v1 = _SIMD_FMA(a3_1, h_5_3, v1);
    v1 = _SIMD_FMA(a2_1, h_5_2, v1);
    v1 = _SIMD_FMA(a1_1, h_5_1, v1);
#else
    register __SIMD_DATATYPE v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_1, h_5_4)); 
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_5_3));
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_5_2));
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_5_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_4_3 = _SIMD_SET1(hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET1(hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_4_3 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_4_3 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w1 = _SIMD_FMA(a3_1, h_4_3, a4_1);
    w1 = _SIMD_FMA(a2_1, h_4_2, w1);
    w1 = _SIMD_FMA(a1_1, h_4_1, w1);
#else
    register __SIMD_DATATYPE w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_4_3)); 
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_4_2));
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_4_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_2_1 = _SIMD_SET1(hh[ldh+1]);    
    __SIMD_DATATYPE h_3_2 = _SIMD_SET1(hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_2_1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_2_1 = _SIMD_BROADCAST(&hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE z1 = _SIMD_FMA(a2_1, h_3_2, a3_1);
    z1 = _SIMD_FMA(a1_1, h_3_1, z1);
    register __SIMD_DATATYPE y1 = _SIMD_FMA(a1_1, h_2_1, a2_1);
#else
    register __SIMD_DATATYPE z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_3_2));
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_3_1));
    register __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_2_1)); 
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x1 = a1_1;

    __SIMD_DATATYPE a1_2 = _LOAD(&q[(ldq*5)+offset]);
    __SIMD_DATATYPE a2_2 = _LOAD(&q[(ldq*4)+offset]);
    __SIMD_DATATYPE a3_2 = _LOAD(&q[(ldq*3)+offset]);
    __SIMD_DATATYPE a4_2 = _LOAD(&q[(ldq*2)+offset]);
    __SIMD_DATATYPE a5_2 = _LOAD(&q[(ldq)+offset]);
    __SIMD_DATATYPE a6_2 = _LOAD(&q[offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t2 = _SIMD_FMA(a5_2, h_6_5, a6_2);
    t2 = _SIMD_FMA(a4_2, h_6_4, t2);
    t2 = _SIMD_FMA(a3_2, h_6_3, t2);
    t2 = _SIMD_FMA(a2_2, h_6_2, t2);
    t2 = _SIMD_FMA(a1_2, h_6_1, t2);
    register __SIMD_DATATYPE v2 = _SIMD_FMA(a4_2, h_5_4, a5_2);
    v2 = _SIMD_FMA(a3_2, h_5_3, v2);
    v2 = _SIMD_FMA(a2_2, h_5_2, v2);
    v2 = _SIMD_FMA(a1_2, h_5_1, v2);
    register __SIMD_DATATYPE w2 = _SIMD_FMA(a3_2, h_4_3, a4_2);
    w2 = _SIMD_FMA(a2_2, h_4_2, w2);
    w2 = _SIMD_FMA(a1_2, h_4_1, w2);
    register __SIMD_DATATYPE z2 = _SIMD_FMA(a2_2, h_3_2, a3_2);
    z2 = _SIMD_FMA(a1_2, h_3_1, z2);
    register __SIMD_DATATYPE y2 = _SIMD_FMA(a1_2, h_2_1, a2_2);
#else
    register __SIMD_DATATYPE t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_2, h_6_5));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_2, h_6_4));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_6_3));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_6_2));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_6_1));
    register __SIMD_DATATYPE v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_2, h_5_4));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_5_3));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_5_2));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_5_1));
    register __SIMD_DATATYPE w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_4_3));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_4_2));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_4_1));
    register __SIMD_DATATYPE z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_3_2));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_3_1));
    register __SIMD_DATATYPE y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_2_1));
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x2 = a1_2;

    __SIMD_DATATYPE a1_3 = _LOAD(&q[(ldq*5)+2*offset]);
    __SIMD_DATATYPE a2_3 = _LOAD(&q[(ldq*4)+2*offset]);
    __SIMD_DATATYPE a3_3 = _LOAD(&q[(ldq*3)+2*offset]);
    __SIMD_DATATYPE a4_3 = _LOAD(&q[(ldq*2)+2*offset]);
    __SIMD_DATATYPE a5_3 = _LOAD(&q[(ldq)+2*offset]);
    __SIMD_DATATYPE a6_3 = _LOAD(&q[2*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t3 = _SIMD_FMA(a5_3, h_6_5, a6_3);
    t3 = _SIMD_FMA(a4_3, h_6_4, t3);
    t3 = _SIMD_FMA(a3_3, h_6_3, t3);
    t3 = _SIMD_FMA(a2_3, h_6_2, t3);
    t3 = _SIMD_FMA(a1_3, h_6_1, t3);
    register __SIMD_DATATYPE v3 = _SIMD_FMA(a4_3, h_5_4, a5_3);
    v3 = _SIMD_FMA(a3_3, h_5_3, v3);
    v3 = _SIMD_FMA(a2_3, h_5_2, v3);
    v3 = _SIMD_FMA(a1_3, h_5_1, v3);
    register __SIMD_DATATYPE w3 = _SIMD_FMA(a3_3, h_4_3, a4_3);
    w3 = _SIMD_FMA(a2_3, h_4_2, w3);
    w3 = _SIMD_FMA(a1_3, h_4_1, w3);
    register __SIMD_DATATYPE z3 = _SIMD_FMA(a2_3, h_3_2, a3_3);
    z3 = _SIMD_FMA(a1_3, h_3_1, z3);
    register __SIMD_DATATYPE y3 = _SIMD_FMA(a1_3, h_2_1, a2_3);
#else
    register __SIMD_DATATYPE t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_3, h_6_5));
    t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_3, h_6_4));
    t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_3, h_6_3));
    t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_6_2));
    t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_6_1));
    register __SIMD_DATATYPE v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_3, h_5_4));
    v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_3, h_5_3));
    v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_5_2));
    v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_5_1));
    register __SIMD_DATATYPE w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_3, h_4_3));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_4_2));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_4_1));
    register __SIMD_DATATYPE z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_3_2));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_3_1));
    register __SIMD_DATATYPE y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_2_1));
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x3 = a1_3;

    __SIMD_DATATYPE a1_4 = _LOAD(&q[(ldq*5)+3*offset]);
    __SIMD_DATATYPE a2_4 = _LOAD(&q[(ldq*4)+3*offset]);
    __SIMD_DATATYPE a3_4 = _LOAD(&q[(ldq*3)+3*offset]);
    __SIMD_DATATYPE a4_4 = _LOAD(&q[(ldq*2)+3*offset]);
    __SIMD_DATATYPE a5_4 = _LOAD(&q[(ldq)+3*offset]);
    __SIMD_DATATYPE a6_4 = _LOAD(&q[3*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t4 = _SIMD_FMA(a5_4, h_6_5, a6_4);
    t4 = _SIMD_FMA(a4_4, h_6_4, t4);
    t4 = _SIMD_FMA(a3_4, h_6_3, t4);
    t4 = _SIMD_FMA(a2_4, h_6_2, t4);
    t4 = _SIMD_FMA(a1_4, h_6_1, t4);
    register __SIMD_DATATYPE v4 = _SIMD_FMA(a4_4, h_5_4, a5_4);
    v4 = _SIMD_FMA(a3_4, h_5_3, v4);
    v4 = _SIMD_FMA(a2_4, h_5_2, v4);
    v4 = _SIMD_FMA(a1_4, h_5_1, v4);
    register __SIMD_DATATYPE w4 = _SIMD_FMA(a3_4, h_4_3, a4_4);
    w4 = _SIMD_FMA(a2_4, h_4_2, w4);
    w4 = _SIMD_FMA(a1_4, h_4_1, w4);
    register __SIMD_DATATYPE z4 = _SIMD_FMA(a2_4, h_3_2, a3_4);
    z4 = _SIMD_FMA(a1_4, h_3_1, z4);
    register __SIMD_DATATYPE y4 = _SIMD_FMA(a1_4, h_2_1, a2_4);
#else
    register __SIMD_DATATYPE t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_4, h_6_5));
    t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT t4, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_4, h_6_4));
    t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT t4, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_4, h_6_3));
    t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT t4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_6_2));
    t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT t4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_6_1));
    register __SIMD_DATATYPE v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_4, h_5_4));
    v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT v4, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_4, h_5_3));
    v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT v4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_5_2));
    v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT v4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_5_1));
    register __SIMD_DATATYPE w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_4, h_4_3));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_4_2));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_4_1));
    register __SIMD_DATATYPE z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_3_2));
    z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_3_1));
    register __SIMD_DATATYPE y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_2_1));
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x4 = a1_4;

    __SIMD_DATATYPE a1_5 = _LOAD(&q[(ldq*5)+4*offset]);
    __SIMD_DATATYPE a2_5 = _LOAD(&q[(ldq*4)+4*offset]);
    __SIMD_DATATYPE a3_5 = _LOAD(&q[(ldq*3)+4*offset]);
    __SIMD_DATATYPE a4_5 = _LOAD(&q[(ldq*2)+4*offset]);
    __SIMD_DATATYPE a5_5 = _LOAD(&q[(ldq)+4*offset]);
    __SIMD_DATATYPE a6_5 = _LOAD(&q[4*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t5 = _SIMD_FMA(a5_5, h_6_5, a6_5);
    t5 = _SIMD_FMA(a4_5, h_6_4, t5);
    t5 = _SIMD_FMA(a3_5, h_6_3, t5);
    t5 = _SIMD_FMA(a2_5, h_6_2, t5);
    t5 = _SIMD_FMA(a1_5, h_6_1, t5);
    register __SIMD_DATATYPE v5 = _SIMD_FMA(a4_5, h_5_4, a5_5);
    v5 = _SIMD_FMA(a3_5, h_5_3, v5);
    v5 = _SIMD_FMA(a2_5, h_5_2, v5);
    v5 = _SIMD_FMA(a1_5, h_5_1, v5);
    register __SIMD_DATATYPE w5 = _SIMD_FMA(a3_5, h_4_3, a4_5);
    w5 = _SIMD_FMA(a2_5, h_4_2, w5);
    w5 = _SIMD_FMA(a1_5, h_4_1, w5);
    register __SIMD_DATATYPE z5 = _SIMD_FMA(a2_5, h_3_2, a3_5);
    z5 = _SIMD_FMA(a1_5, h_3_1, z5);
    register __SIMD_DATATYPE y5 = _SIMD_FMA(a1_5, h_2_1, a2_5);
#else
    register __SIMD_DATATYPE t5 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_5, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_5, h_6_5));
    t5 = _SIMD_ADD( ADDITIONAL_ARGUMENT t5, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_5, h_6_4));
    t5 = _SIMD_ADD( ADDITIONAL_ARGUMENT t5, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_5, h_6_3));
    t5 = _SIMD_ADD( ADDITIONAL_ARGUMENT t5, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_5, h_6_2));
    t5 = _SIMD_ADD( ADDITIONAL_ARGUMENT t5, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_5, h_6_1));
    register __SIMD_DATATYPE v5 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_5, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_5, h_5_4));
    v5 = _SIMD_ADD( ADDITIONAL_ARGUMENT v5, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_5, h_5_3));
    v5 = _SIMD_ADD( ADDITIONAL_ARGUMENT v5, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_5, h_5_2));
    v5 = _SIMD_ADD( ADDITIONAL_ARGUMENT v5, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_5, h_5_1));
    register __SIMD_DATATYPE w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_5, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_5, h_4_3));
    w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT w5, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_5, h_4_2));
    w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT w5, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_5, h_4_1));
    register __SIMD_DATATYPE z5 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_5, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_5, h_3_2));
    z5 = _SIMD_ADD( ADDITIONAL_ARGUMENT z5, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_5, h_3_1));
    register __SIMD_DATATYPE y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_5, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_5, h_2_1));
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x5 = a1_5;


    __SIMD_DATATYPE q1;
    __SIMD_DATATYPE q2;
    __SIMD_DATATYPE q3;
    __SIMD_DATATYPE q4;
    __SIMD_DATATYPE q5;

    __SIMD_DATATYPE h1;
    __SIMD_DATATYPE h2;
    __SIMD_DATATYPE h3;
    __SIMD_DATATYPE h4;
    __SIMD_DATATYPE h5;
    __SIMD_DATATYPE h6;

#endif /* BLOCK6 */


    for(i = BLOCK; i < nb; i++)
      {
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h1 = _SIMD_SET1(hh[i-(BLOCK-1)]);
        h2 = _SIMD_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == SPARC64_SSE
        h1 = _SIMD_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
        h2 = _SIMD_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif
#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h1 = _SIMD_BROADCAST(&hh[i-(BLOCK-1)]);
        h2 = _SIMD_BROADCAST(&hh[ldh+i-(BLOCK-2)]);
#endif /*   VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

        q1 = _LOAD(&q[i*ldq]);
        q2 = _LOAD(&q[(i*ldq)+offset]);
        q3 = _LOAD(&q[(i*ldq)+2*offset]);
        q4 = _LOAD(&q[(i*ldq)+3*offset]);
        q5 = _LOAD(&q[(i*ldq)+4*offset]);
#ifdef __ELPA_USE_FMA__
        x1 = _SIMD_FMA(q1, h1, x1);
        y1 = _SIMD_FMA(q1, h2, y1);
        x2 = _SIMD_FMA(q2, h1, x2);
        y2 = _SIMD_FMA(q2, h2, y2);
        x3 = _SIMD_FMA(q3, h1, x3);
        y3 = _SIMD_FMA(q3, h2, y3);
        x4 = _SIMD_FMA(q4, h1, x4);
        y4 = _SIMD_FMA(q4, h2, y4);
        x5 = _SIMD_FMA(q5, h1, x5);
        y5 = _SIMD_FMA(q5, h2, y5);
#else
        x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
        y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
        x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
        y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
        x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
        y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
        x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
        y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
        x5 = _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h1));
        y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT y5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h2));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h3 = _SIMD_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
        h3 = _SIMD_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h3 = _SIMD_BROADCAST(&hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
        z1 = _SIMD_FMA(q1, h3, z1);
        z2 = _SIMD_FMA(q2, h3, z2);
        z3 = _SIMD_FMA(q3, h3, z3);
        z4 = _SIMD_FMA(q4, h3, z4);
        z5 = _SIMD_FMA(q5, h3, z5);
#else
        z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
        z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
        z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h3));
        z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h3));
        z5 = _SIMD_ADD( ADDITIONAL_ARGUMENT z5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h4 = _SIMD_SET1(hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == SPARC64_SSE
        h4 = _SIMD_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h4 = _SIMD_BROADCAST(&hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
        w1 = _SIMD_FMA(q1, h4, w1);
        w2 = _SIMD_FMA(q2, h4, w2);
        w3 = _SIMD_FMA(q3, h4, w3);
        w4 = _SIMD_FMA(q4, h4, w4);
        w5 = _SIMD_FMA(q5, h4, w5);
#else
        w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4));
        w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h4));
        w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h4));
        w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h4));
        w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT w5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h4));
#endif /* __ELPA_USE_FMA__ */
			
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h5 = _SIMD_SET1(hh[(ldh*4)+i-1]);
#endif
#if VEC_SET == SPARC64_SSE
        h5 = _SIMD_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h5 = _SIMD_BROADCAST(&hh[(ldh*4)+i-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
        v1 = _SIMD_FMA(q1, h5, v1);
        v2 = _SIMD_FMA(q2, h5, v2);
        v3 = _SIMD_FMA(q3, h5, v3);
        v4 = _SIMD_FMA(q4, h5, v4);
        v5 = _SIMD_FMA(q5, h5, v5);
#else
        v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h5));
        v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h5));
        v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h5));
        v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT v4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h5));
        v5 = _SIMD_ADD( ADDITIONAL_ARGUMENT v5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h6 = _SIMD_SET1(hh[(ldh*5)+i]);
#endif

#if VEC_SET == SPARC64_SSE
        h6 = _SIMD_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif


#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h6 = _SIMD_BROADCAST(&hh[(ldh*5)+i-(BLOCK-6)]);
#endif

#ifdef __ELPA_USE_FMA__
        t1 = _SIMD_FMA(q1, h6, t1);
        t2 = _SIMD_FMA(q2, h6, t2);
        t3 = _SIMD_FMA(q3, h6, t3);
        t4 = _SIMD_FMA(q4, h6, t4);
        t5 = _SIMD_FMA(q5, h6, t5);
#else
        t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h6));
        t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h6));
        t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h6));
        t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT t4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h6));
        t5 = _SIMD_ADD( ADDITIONAL_ARGUMENT t5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h6));
#endif /* __ELPA_USE_FMA__ */	

#endif /* BLOCK6 */
      }
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif 
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-1)]);
#endif

    q1 = _LOAD(&q[nb*ldq]);
    q2 = _LOAD(&q[(nb*ldq)+offset]);
    q3 = _LOAD(&q[(nb*ldq)+2*offset]);
    q4 = _LOAD(&q[(nb*ldq)+3*offset]);
    q5 = _LOAD(&q[(nb*ldq)+4*offset]);
#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
    x5 = _SIMD_FMA(q5, h1, x5);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
    x5 = _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h1));
#endif /* __ELPA_USE_FMA__ */

#if defined(BLOCK4) || defined(BLOCK6)
    
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
    y4 = _SIMD_FMA(q4, h2, y4);
    y5 = _SIMD_FMA(q5, h2, y5);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
    y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
    y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT y5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
    z2 = _SIMD_FMA(q2, h3, z2);
    z3 = _SIMD_FMA(q3, h3, z3);
    z4 = _SIMD_FMA(q4, h3, z4);
    z5 = _SIMD_FMA(q5, h3, z5);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h3));
    z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h3));
    z5 = _SIMD_ADD( ADDITIONAL_ARGUMENT z5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h3));
#endif

#ifdef BLOCK4
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-2]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-2], hh[nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-2]);
#endif

    q1 = _LOAD(&q[(nb+1)*ldq]);
    q2 = _LOAD(&q[((nb+1)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+1)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+1)*ldq)+3*offset]);
    q5 = _LOAD(&q[((nb+1)*ldq)+4*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
    x5 = _SIMD_FMA(q5, h1, x5);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
    x5 = _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[(ldh*1)+nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[(ldh*1)+nb-1], hh[(ldh*1)+nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[(ldh*1)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
    y4 = _SIMD_FMA(q4, h2, y4);
    y5 = _SIMD_FMA(q5, h2, y5);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
    y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
    y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT y5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-1], hh[nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-1]);
#endif

    q1 = _LOAD(&q[(nb+2)*ldq]);
    q2 = _LOAD(&q[((nb+2)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+2)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+2)*ldq)+3*offset]);
    q5 = _LOAD(&q[((nb+2)*ldq)+4*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
    x5 = _SIMD_FMA(q5, h1, x5);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
    x5 = _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h1));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 */
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h4 = _SIMD_SET1(hh[(ldh*3)+nb-2]);
#endif

#if VEC_SET == SPARC64_SSE
    h4 = _SIMD_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    w1 = _SIMD_FMA(q1, h4, w1);
    w2 = _SIMD_FMA(q2, h4, w2);
    w3 = _SIMD_FMA(q3, h4, w3);
    w4 = _SIMD_FMA(q4, h4, w4);
    w5 = _SIMD_FMA(q5, h4, w5);
#else
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4)); 
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h4));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h4));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h4));
    w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT w5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h5 = _SIMD_SET1(hh[(ldh*4)+nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h5 = _SIMD_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h5 = _SIMD_BROADCAST(&hh[(ldh*4)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    v1 = _SIMD_FMA(q1, h5, v1);
    v2 = _SIMD_FMA(q2, h5, v2);
    v3 = _SIMD_FMA(q3, h5, v3);
    v4 = _SIMD_FMA(q4, h5, v4);
    v5 = _SIMD_FMA(q5, h5, v5);
#else
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h5));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h5));
    v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h5));
    v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT v4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h5));
    v5 = _SIMD_ADD( ADDITIONAL_ARGUMENT v5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-4]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-4], hh[nb-4]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-2)]);
#endif

    q1 = _LOAD(&q[(nb+1)*ldq]);
    q2 = _LOAD(&q[((nb+1)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+1)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+1)*ldq)+3*offset]);
    q5 = _LOAD(&q[((nb+1)*ldq)+4*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
    x5 = _SIMD_FMA(q5, h1, x5);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
    x5 = _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-3]);
#endif
#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-3], hh[ldh+nb-3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
    y4 = _SIMD_FMA(q4, h2, y4);
    y5 = _SIMD_FMA(q5, h2, y5);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
    y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
    y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT y5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
    z2 = _SIMD_FMA(q2, h3, z2);
    z3 = _SIMD_FMA(q3, h3, z3);
    z4 = _SIMD_FMA(q4, h3, z4);
    z5 = _SIMD_FMA(q5, h3, z5);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h3));
    z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h3));
    z5 = _SIMD_ADD( ADDITIONAL_ARGUMENT z5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h4 = _SIMD_SET1(hh[(ldh*3)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
    h4 = _SIMD_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    w1 = _SIMD_FMA(q1, h4, w1);
    w2 = _SIMD_FMA(q2, h4, w2);
    w3 = _SIMD_FMA(q3, h4, w3);
    w4 = _SIMD_FMA(q4, h4, w4);
    w5 = _SIMD_FMA(q5, h4, w5);
#else
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h4));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h4));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h4));
    w5 = _SIMD_ADD( ADDITIONAL_ARGUMENT w5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-3]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-3], hh[nb-3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-3]);
#endif

    q1 = _LOAD(&q[(nb+2)*ldq]);
    q2 = _LOAD(&q[((nb+2)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+2)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+2)*ldq)+3*offset]);
    q5 = _LOAD(&q[((nb+2)*ldq)+4*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
    x5 = _SIMD_FMA(q5, h1, x5);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
    x5 = _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-2]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
    y4 = _SIMD_FMA(q4, h2, y4);
    y5 = _SIMD_FMA(q5, h2, y5);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
    y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
    y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT y5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h2));
#endif /* __ELPA_USE_FMA__ */
 
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
    z2 = _SIMD_FMA(q2, h3, z2);
    z3 = _SIMD_FMA(q3, h3, z3);
    z4 = _SIMD_FMA(q4, h3, z4);
    z5 = _SIMD_FMA(q5, h3, z5);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h3));
    z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h3));
    z5 = _SIMD_ADD( ADDITIONAL_ARGUMENT z5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-2]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-2], hh[nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-4)]);
#endif

    q1 = _LOAD(&q[(nb+3)*ldq]);
    q2 = _LOAD(&q[((nb+3)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+3)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+3)*ldq)+3*offset]);
    q5 = _LOAD(&q[((nb+3)*ldq)+4*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
    x5 = _SIMD_FMA(q5, h1, x5);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
    x5 = _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
    y4 = _SIMD_FMA(q4, h2, y4);
    y5 = _SIMD_FMA(q5, h2, y5);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
    y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
    y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT y5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-1], hh[nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-5)]);
#endif

    q1 = _LOAD(&q[(nb+4)*ldq]);
    q2 = _LOAD(&q[((nb+4)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+4)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+4)*ldq)+3*offset]);
    q5 = _LOAD(&q[((nb+4)*ldq)+4*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
    x5 = _SIMD_FMA(q5, h1, x5);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
    x5 = _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT q5,h1));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Rank-2 update of Q [ROW_LENGTH x nb+1]
    /////////////////////////////////////////////////////
#endif
#ifdef BLOCK4
    /////////////////////////////////////////////////////
    // Rank-1 update of Q [ROW_LENGTH x nb+3]
    /////////////////////////////////////////////////////
#endif
#ifdef BLOCK6
    /////////////////////////////////////////////////////
    // Apply tau, correct wrong calculation using pre-calculated scalar products
    /////////////////////////////////////////////////////
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE tau1 = _SIMD_SET1(hh[0]);

    __SIMD_DATATYPE tau2 = _SIMD_SET1(hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_SET1(hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_SET1(hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_SET1(hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_SET1(hh[ldh*5]);       
#endif

#ifdef BLOCK2    
    __SIMD_DATATYPE vs = _SIMD_SET1(s);
#endif
#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET1(s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET1(s_1_3);  
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET1(s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET1(s_1_4);  
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET1(s_2_4);  
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET1(s_3_4);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET1(scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET1(scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET1(scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET1(scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET1(scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET1(scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_SET1(scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_SET1(scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_SET1(scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_SET1(scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_SET1(scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_SET1(scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_SET1(scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_SET1(scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_SET1(scalarprods[14]);
#endif
#endif /* VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE */

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE tau1 = _SIMD_SET(hh[0], hh[0]);

    __SIMD_DATATYPE tau2 = _SIMD_SET(hh[ldh], hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_SET(hh[ldh*2], hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_SET(hh[ldh*3], hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_SET(hh[ldh*4], hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_SET(hh[ldh*5], hh[ldh*5]);
#endif

#ifdef BLOCK2
    __SIMD_DATATYPE vs = _SIMD_SET(s, s);
#endif
#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET(s_1_2, s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET(s_1_3, s_1_3);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET(s_2_3, s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET(s_1_4, s_1_4);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET(s_2_4, s_2_4);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET(s_3_4, s_3_4);

#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET(scalarprods[0], scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET(scalarprods[1], scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET(scalarprods[2], scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET(scalarprods[3], scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET(scalarprods[4], scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET(scalarprods[5], scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_SET(scalarprods[6], scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_SET(scalarprods[7], scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_SET(scalarprods[8], scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_SET(scalarprods[9], scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_SET(scalarprods[10], scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_SET(scalarprods[11], scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_SET(scalarprods[12], scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_SET(scalarprods[13], scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_SET(scalarprods[14], scalarprods[14]);
#endif
#endif /*  VEC_SET == SPARC64_SSE  */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   __SIMD_DATATYPE tau1 = _SIMD_BROADCAST(hh);
   __SIMD_DATATYPE tau2 = _SIMD_BROADCAST(&hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_BROADCAST(&hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_BROADCAST(&hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_BROADCAST(&hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_BROADCAST(&hh[ldh*5]);
#endif

#ifdef BLOCK2  
   __SIMD_DATATYPE vs = _SIMD_BROADCAST(&s);
#endif

#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_BROADCAST(&s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_BROADCAST(&s_1_3);
   __SIMD_DATATYPE vs_2_3 = _SIMD_BROADCAST(&s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_BROADCAST(&s_1_4);
   __SIMD_DATATYPE vs_2_4 = _SIMD_BROADCAST(&s_2_4);
   __SIMD_DATATYPE vs_3_4 = _SIMD_BROADCAST(&s_3_4);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_BROADCAST(&scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_BROADCAST(&scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_BROADCAST(&scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_BROADCAST(&scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_BROADCAST(&scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_BROADCAST(&scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_BROADCAST(&scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_BROADCAST(&scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_BROADCAST(&scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_BROADCAST(&scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_BROADCAST(&scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_BROADCAST(&scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_BROADCAST(&scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_BROADCAST(&scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_BROADCAST(&scalarprods[14]);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == VSX_SSE || VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _XOR(tau1, sign);
#endif

#if VEC_SET == SPARC64_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
    h1 = _SIMD_NEG( ADDITIONAL_ARGUMENT tau1);
#endif

#if VEC_SET == AVX_512
#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi64((__SIMD_INTEGER) tau1, (__SIMD_INTEGER) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi32((__SIMD_INTEGER) tau1, (__SIMD_INTEGER) sign);
#endif
#endif /* HAVE_AVX512_XEON_PHI */

#ifdef HAVE_AVX512_XEON
    h1 = _XOR(tau1, sign);
#endif
#endif /* VEC_SET == AVX_512 */

#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau1;
#endif

   x1 = _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1);
   x2 = _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1);
   x3 = _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1);
   x4 = _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1);
   x5 = _SIMD_MUL( ADDITIONAL_ARGUMENT x5, h1);

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == VSX_SSE || VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _XOR(tau2, sign);
#endif

#if VEC_SET == SPARC64_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
   h1 = _SIMD_NEG( ADDITIONAL_ARGUMENT tau2);
#endif

#if VEC_SET == AVX_512
#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi64((__SIMD_INTEGER) tau2, (__SIMD_INTEGER) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi32((__SIMD_INTEGER) tau2, (__SIMD_INTEGER) sign);
#endif
#endif /* HAVE_AVX512_XEON_PHI */

#ifdef HAVE_AVX512_XEON
    h1 = _XOR(tau2, sign);
#endif
#endif /* VEC_SET == AVX_512 */
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs);
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau2;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_2);
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK2

#ifdef __ELPA_USE_FMA__
   y1 = _SIMD_FMA(y1, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_FMA(y2, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
   y3 = _SIMD_FMA(y3, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2));
   y4 = _SIMD_FMA(y4, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2));
   y5 = _SIMD_FMA(y5, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2));
#else
   y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
   y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2));
   y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2));
   y5 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y5,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2));
#endif
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)

#ifdef __ELPA_USE_FMA__
   y1 = _SIMD_FMSUB(y1, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_FMSUB(y2, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
   y3 = _SIMD_FMSUB(y3, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2));
   y4 = _SIMD_FMSUB(y4, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2));
   y5 = _SIMD_FMSUB(y5, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2));
#else   
   y1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
   y3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2));
   y4 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2));
   y5 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y5,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2));
#endif /* __ELPA_USE_FMA__ */

   h1 = tau3;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_3);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_2_3);

#ifdef __ELPA_USE_FMA__
   z1 = _SIMD_FMSUB(z1, h1, _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)));
   z2 = _SIMD_FMSUB(z2, h1, _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)));
   z3 = _SIMD_FMSUB(z3, h1, _SIMD_FMA(y3, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2)));
   z4 = _SIMD_FMSUB(z4, h1, _SIMD_FMA(y4, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2)));
   z5 = _SIMD_FMSUB(z5, h1, _SIMD_FMA(y5, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2)));
#else
   z1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)));
   z2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)));
   z3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2)));
   z4 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2)));
   z5 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z5,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y5,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2)));
#endif /* __ELPA_USE_FMA__ */

   h1 = tau4;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_4);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_2_4);
   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_3_4);

#ifdef __ELPA_USE_FMA__
   w1 = _SIMD_FMSUB(w1, h1, _SIMD_FMA(z1, h4, _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   w2 = _SIMD_FMSUB(w2, h1, _SIMD_FMA(z2, h4, _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
   w3 = _SIMD_FMSUB(w3, h1, _SIMD_FMA(z3, h4, _SIMD_FMA(y3, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2))));
   w4 = _SIMD_FMSUB(w4, h1, _SIMD_FMA(z4, h4, _SIMD_FMA(y4, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2))));
   w5 = _SIMD_FMSUB(w5, h1, _SIMD_FMA(z5, h4, _SIMD_FMA(y5, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2))));
#else
   w1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   w2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
   w3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w3,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2))));
   w4 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w4,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2))));
   w5 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w5,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z5,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y5,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2))));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_1_5); 
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_2_5);
   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_3_5);
   h5 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_4_5);

#ifdef __ELPA_USE_FMA__
   v1 = _SIMD_FMSUB(v1, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w1, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   v2 = _SIMD_FMSUB(v2, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w2, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
   v3 = _SIMD_FMSUB(v3, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w3, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4)), _SIMD_FMA(y3, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2))));
   v4 = _SIMD_FMSUB(v4, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w4, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h4)), _SIMD_FMA(y4, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2))));
   v5 = _SIMD_FMSUB(v5, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w5, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5,h4)), _SIMD_FMA(y5, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2))));
#else
   v1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v1,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   v2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v2,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
   v3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v3,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w3,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2))));
   v4 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v4,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w4,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2))));
   v5 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v5,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w5,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z5,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y5,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2))));
#endif /* __ELPA_USE_FMA__ */

   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_1_6);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_2_6);
   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_3_6);
   h5 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_4_6);
   h6 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_5_6);

#ifdef __ELPA_USE_FMA__
   t1 = _SIMD_FMSUB(t1, tau6, _SIMD_FMA(v1, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w1, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)))));
   t2 = _SIMD_FMSUB(t2, tau6, _SIMD_FMA(v2, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w2, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)))));
   t3 = _SIMD_FMSUB(t3, tau6, _SIMD_FMA(v3, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w3, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4)), _SIMD_FMA(y3, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2)))));
   t4 = _SIMD_FMSUB(t4, tau6, _SIMD_FMA(v4, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w4, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h4)), _SIMD_FMA(y4, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2)))));
   t5 = _SIMD_FMSUB(t5, tau6, _SIMD_FMA(v5, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w5, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5,h4)), _SIMD_FMA(y5, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2)))));
#else
   t1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t1,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v1,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)))));
   t2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t2,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v2,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)))));
   t3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t3,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v3,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w3,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2)))));
   t4 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t4,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v4,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w4,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2)))));
   t5 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t5,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v5,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w5,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z5,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y5,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h2)))));
#endif /* __ELPA_USE_FMA__ */

   /////////////////////////////////////////////////////
   // Rank-1 update of Q [ ROW_LENGTH x nb+3]
   /////////////////////////////////////////////////////
#endif /* BLOCK6 */

   q1 = _LOAD(&q[0]);
#ifdef BLOCK2
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, y1);
#endif
#ifdef BLOCK4
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, w1);
#endif
#ifdef BLOCK6
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, t1); 
#endif
   _STORE(&q[0],q1);
   q2 = _LOAD(&q[offset]);
#ifdef BLOCK2
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, y2);
#endif
#ifdef BLOCK4
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, w2);
#endif
#ifdef BLOCK6
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, t2);
#endif
   _STORE(&q[offset],q2);
   q3 = _LOAD(&q[2*offset]);
#ifdef BLOCK2
   q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, y3);
#endif
#ifdef BLOCK4
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, w3);
#endif
#ifdef BLOCK6
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, t3);
#endif
   _STORE(&q[2*offset],q3);
   q4 = _LOAD(&q[3*offset]);
#ifdef BLOCK2
   q4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, y4);
#endif
#ifdef BLOCK4
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, w4);
#endif
#ifdef BLOCK6
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, t4);
#endif
   _STORE(&q[3*offset],q4);
   q5 = _LOAD(&q[4*offset]);
#ifdef BLOCK2
   q5 = _SIMD_ADD( ADDITIONAL_ARGUMENT q5, y5);
#endif
#ifdef BLOCK4
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, w5);
#endif
#ifdef BLOCK6
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, t5);
#endif
   _STORE(&q[4*offset],q5);

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif

   q1 = _LOAD(&q[ldq]);
   q2 = _LOAD(&q[ldq+offset]);
   q3 = _LOAD(&q[ldq+2*offset]);
   q4 = _LOAD(&q[ldq+3*offset]);
   q5 = _LOAD(&q[ldq+4*offset]);
#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_FMA(y1, h2, x1));
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_FMA(y2, h2, x2));
   q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_FMA(y3, h2, x3));
   q4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, _SIMD_FMA(y4, h2, x4));
   q5 = _SIMD_ADD( ADDITIONAL_ARGUMENT q5, _SIMD_FMA(y5, h2, x5));
#else
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2)));
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2)));
   q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2)));
   q4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2)));
   q5 = _SIMD_ADD( ADDITIONAL_ARGUMENT q5, _SIMD_ADD( ADDITIONAL_ARGUMENT x5, _SIMD_MUL( ADDITIONAL_ARGUMENT y5, h2)));
#endif /* __ELPA_USE_FMA__ */
   _STORE(&q[ldq],q1);
   _STORE(&q[ldq+offset],q2);
   _STORE(&q[ldq+2*offset],q3);
   _STORE(&q[ldq+3*offset],q4);
   _STORE(&q[ldq+4*offset],q5);
#endif /* BLOCK2 */

#ifdef BLOCK4
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
#endif
   q1 = _LOAD(&q[ldq]);
   q2 = _LOAD(&q[ldq+offset]);
   q3 = _LOAD(&q[ldq+2*offset]);
   q4 = _LOAD(&q[ldq+3*offset]);
   q5 = _LOAD(&q[ldq+4*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_FMA(w1, h4, z1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_FMA(w2, h4, z2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_FMA(w3, h4, z3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_FMA(w4, h4, z4));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_FMA(w5, h4, z5));
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4)));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4)));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4)));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4)));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_ADD( ADDITIONAL_ARGUMENT z5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5, h4)));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq],q1);
   _STORE(&q[ldq+offset],q2);
   _STORE(&q[ldq+2*offset],q3);
   _STORE(&q[ldq+3*offset],q4);
   _STORE(&q[ldq+4*offset],q5);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+2]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
#endif
   q1 = _LOAD(&q[ldq*2]);
   q2 = _LOAD(&q[(ldq*2)+offset]);
   q3 = _LOAD(&q[(ldq*2)+2*offset]);
   q4 = _LOAD(&q[(ldq*2)+3*offset]);
   q5 = _LOAD(&q[(ldq*2)+4*offset]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, y1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, y2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, y3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, y4);
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, y5);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
   q5 = _SIMD_NFMA(w5, h4, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
   q5 = _SIMD_NFMA(z5, h3, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5, h3));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*2],q1);
   _STORE(&q[(ldq*2)+offset],q2);
   _STORE(&q[(ldq*2)+2*offset],q3);
   _STORE(&q[(ldq*2)+3*offset],q4);
   _STORE(&q[(ldq*2)+4*offset],q5);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

   q1 = _LOAD(&q[ldq*3]);
   q2 = _LOAD(&q[(ldq*3)+offset]);
   q3 = _LOAD(&q[(ldq*3)+2*offset]);
   q4 = _LOAD(&q[(ldq*3)+3*offset]);
   q5 = _LOAD(&q[(ldq*3)+4*offset]);

   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, x1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, x2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, x3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, x4);
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, x5);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
   q5 = _SIMD_NFMA(w5, h4, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5, h4));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
   q5 = _SIMD_NFMA(y5, h2, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT y5, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
   q5 = _SIMD_NFMA(z5, h3, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5, h3));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*3], q1);
   _STORE(&q[(ldq*3)+offset], q2);
   _STORE(&q[(ldq*3)+2*offset], q3);
   _STORE(&q[(ldq*3)+3*offset], q4);
   _STORE(&q[(ldq*3)+4*offset], q5);

#endif /* BLOCK4 */

#ifdef BLOCK6
   
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+1]);
#endif

   q1 = _LOAD(&q[ldq]);
   q2 = _LOAD(&q[(ldq+offset)]);
   q3 = _LOAD(&q[(ldq+2*offset)]);
   q4 = _LOAD(&q[(ldq+3*offset)]);
   q5 = _LOAD(&q[(ldq+4*offset)]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, v1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, v2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, v3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, v4);
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, v5);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
   q4 = _SIMD_NFMA(t4, h6, q4);
   q5 = _SIMD_NFMA(t5, h6, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT t5, h6));
#endif

   _STORE(&q[ldq],q1);
   _STORE(&q[(ldq+offset)],q2);
   _STORE(&q[(ldq+2*offset)],q3);
   _STORE(&q[(ldq+3*offset)],q4);
   _STORE(&q[(ldq+4*offset)],q5);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+1]);
#endif
   q1 = _LOAD(&q[ldq*2]);
   q2 = _LOAD(&q[(ldq*2)+offset]);
   q3 = _LOAD(&q[(ldq*2)+2*offset]);
   q4 = _LOAD(&q[(ldq*2)+3*offset]);
   q5 = _LOAD(&q[(ldq*2)+4*offset]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, w1); 
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, w2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, w3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, w4);
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, w5);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
   q4 = _SIMD_NFMA(v4, h5, q4);
   q4 = _SIMD_NFMA(v4, h5, q4);
   q5 = _SIMD_NFMA(v5, h5, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5)); 
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));  
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));  
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));  
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT v5, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
   q4 = _SIMD_NFMA(t4, h6, q4);
   q5 = _SIMD_NFMA(t5, h6, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT t5, h6));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*2],q1);
   _STORE(&q[(ldq*2)+offset],q2);
   _STORE(&q[(ldq*2)+2*offset],q3);
   _STORE(&q[(ldq*2)+3*offset],q4);
   _STORE(&q[(ldq*2)+4*offset],q5);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
#endif

   q1 = _LOAD(&q[ldq*3]);
   q2 = _LOAD(&q[(ldq*3)+offset]);
   q3 = _LOAD(&q[(ldq*3)+2*offset]);
   q4 = _LOAD(&q[(ldq*3)+3*offset]);
   q5 = _LOAD(&q[(ldq*3)+4*offset]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, z1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, z2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, z3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, z4);
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, z5);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
   q5 = _SIMD_NFMA(w5, h4, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
   q4 = _SIMD_NFMA(v4, h5, q4);
   q5 = _SIMD_NFMA(v5, h5, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT v5, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
   q4 = _SIMD_NFMA(t4, h6, q4);
   q5 = _SIMD_NFMA(t5, h6, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT t5, h6));
#endif

   _STORE(&q[ldq*3],q1);
   _STORE(&q[(ldq*3)+offset],q2);
   _STORE(&q[(ldq*3)+2*offset],q3);
   _STORE(&q[(ldq*3)+3*offset],q4);
   _STORE(&q[(ldq*3)+4*offset],q5);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
#endif

   q1 = _LOAD(&q[ldq*4]);
   q2 = _LOAD(&q[(ldq*4)+offset]);
   q3 = _LOAD(&q[(ldq*4)+2*offset]);
   q4 = _LOAD(&q[(ldq*4)+3*offset]);
   q5 = _LOAD(&q[(ldq*4)+4*offset]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, y1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, y2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, y3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, y4);
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, y5);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
   q5 = _SIMD_NFMA(z5, h3, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5, h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
   q5 = _SIMD_NFMA(w5, h4, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5, h4));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
   q4 = _SIMD_NFMA(v4, h5, q4);
   q5 = _SIMD_NFMA(v5, h5, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT v5, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+4]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+4]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
   q4 = _SIMD_NFMA(t4, h6, q4);
   q5 = _SIMD_NFMA(t5, h6, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT t5, h6));
#endif

   _STORE(&q[ldq*4],q1);
   _STORE(&q[(ldq*4)+offset],q2);
   _STORE(&q[(ldq*4)+2*offset],q3);
   _STORE(&q[(ldq*4)+3*offset],q4);
   _STORE(&q[(ldq*4)+4*offset],q5);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[(ldh)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[(ldh)+1], hh[(ldh)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[(ldh)+1]);
#endif
   q1 = _LOAD(&q[ldq*5]);
   q2 = _LOAD(&q[(ldq*5)+offset]);
   q3 = _LOAD(&q[(ldq*5)+2*offset]);
   q4 = _LOAD(&q[(ldq*5)+3*offset]);
   q5 = _LOAD(&q[(ldq*5)+4*offset]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, x1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, x2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, x3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, x4);
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, x5);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
   q5 = _SIMD_NFMA(y5, h2, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT y5, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
   q5 = _SIMD_NFMA(z5, h3, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5, h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
   q5 = _SIMD_NFMA(w5, h4, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+4]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+4]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
   q4 = _SIMD_NFMA(v4, h5, q4);
   q5 = _SIMD_NFMA(v5, h5, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT v5, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+5]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+5]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
   q4 = _SIMD_NFMA(t4, h6, q4);
   q5 = _SIMD_NFMA(t5, h6, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT t5, h6));
#endif

   _STORE(&q[ldq*5],q1);
   _STORE(&q[(ldq*5)+offset],q2);
   _STORE(&q[(ldq*5)+2*offset],q3);
   _STORE(&q[(ldq*5)+3*offset],q4);
   _STORE(&q[(ldq*5)+4*offset],q5);

#endif /* BLOCK6 */

   for (i = BLOCK; i < nb; i++)
   {
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h1 = _SIMD_SET1(hh[i-(BLOCK-1)]);
     h2 = _SIMD_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == SPARC64_SSE
     h1 = _SIMD_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
     h2 = _SIMD_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[i-(BLOCK-1)]);
    h2 = _SIMD_BROADCAST(&hh[ldh+i-(BLOCK-2)]);
#endif

     q1 = _LOAD(&q[i*ldq]);
     q2 = _LOAD(&q[(i*ldq)+offset]);
     q3 = _LOAD(&q[(i*ldq)+2*offset]);
     q4 = _LOAD(&q[(i*ldq)+3*offset]);
     q5 = _LOAD(&q[(i*ldq)+4*offset]);

#ifdef BLOCK2
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_FMA(x1, h1, q1);
     q1 = _SIMD_FMA(y1, h2, q1);
     q2 = _SIMD_FMA(x2, h1, q2);
     q2 = _SIMD_FMA(y2, h2, q2);
     q3 = _SIMD_FMA(x3, h1, q3);
     q3 = _SIMD_FMA(y3, h2, q3);
     q4 = _SIMD_FMA(x4, h1, q4);
     q4 = _SIMD_FMA(y4, h2, q4);
     q5 = _SIMD_FMA(x5, h1, q5);
     q5 = _SIMD_FMA(y5, h2, q5);
#else
     q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2)));
     q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2)));
     q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2)));
     q4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2)));
     q5 = _SIMD_ADD( ADDITIONAL_ARGUMENT q5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y5, h2)));
#endif /* __ELPA_USE_FMA__ */
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
          
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(x1, h1, q1);
     q2 = _SIMD_NFMA(x2, h1, q2);
     q3 = _SIMD_NFMA(x3, h1, q3);
     q4 = _SIMD_NFMA(x4, h1, q4);
     q5 = _SIMD_NFMA(x5, h1, q5);
#else  
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h1));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h1));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h1));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h1));
     q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT x5,h1));
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(y1, h2, q1);
     q2 = _SIMD_NFMA(y2, h2, q2);
     q3 = _SIMD_NFMA(y3, h2, q3);
     q4 = _SIMD_NFMA(y4, h2, q4);
     q5 = _SIMD_NFMA(y5, h2, q5);
#else   
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h2));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h2));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h2));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h2));
     q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT y5,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h3 = _SIMD_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
     h3 = _SIMD_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h3 = _SIMD_BROADCAST(&hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(z1, h3, q1);
     q2 = _SIMD_NFMA(z2, h3, q2);
     q3 = _SIMD_NFMA(z3, h3, q3);
     q4 = _SIMD_NFMA(z4, h3, q4);
     q5 = _SIMD_NFMA(z5, h3, q5);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h3));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h3));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h3));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h3));
     q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h4 = _SIMD_SET1(hh[(ldh*3)+i-(BLOCK-4)]); 
#endif

#if VEC_SET == SPARC64_SSE
     h4 = _SIMD_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h4 = _SIMD_BROADCAST(&hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(w1, h4, q1);
     q2 = _SIMD_NFMA(w2, h4, q2);
     q3 = _SIMD_NFMA(w3, h4, q3);
     q4 = _SIMD_NFMA(w4, h4, q4);
     q5 = _SIMD_NFMA(w5, h4, q5);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h4));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h4));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3,h4));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4,h4));
     q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5,h4));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h5 = _SIMD_SET1(hh[(ldh*4)+i-1]);
#endif
#if VEC_SET == SPARC64_SSE
     h5 = _SIMD_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h5 = _SIMD_BROADCAST(&hh[(ldh*4)+i-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(v1, h5, q1);
     q2 = _SIMD_NFMA(v2, h5, q2);
     q3 = _SIMD_NFMA(v3, h5, q3);
     q4 = _SIMD_NFMA(v4, h5, q4);
     q5 = _SIMD_NFMA(v5, h5, q5);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));
     q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT v5, h5));
#endif /* __ELPA_USE_FMA__ */
 
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h6 = _SIMD_SET1(hh[(ldh*5)+i]);
#endif
#if VEC_SET == SPARC64_SSE
     h6 = _SIMD_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h6 = _SIMD_BROADCAST(&hh[(ldh*5)+i-(BLOCK-6)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(t1, h6, q1);
     q2 = _SIMD_NFMA(t2, h6, q2);
     q3 = _SIMD_NFMA(t3, h6, q3);
     q4 = _SIMD_NFMA(t4, h6, q4);
     q5 = _SIMD_NFMA(t5, h6, q5);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
     q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT t5, h6));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

     _STORE(&q[i*ldq],q1);
     _STORE(&q[(i*ldq)+offset],q2);
     _STORE(&q[(i*ldq)+2*offset],q3);
     _STORE(&q[(i*ldq)+3*offset],q4);
     _STORE(&q[(i*ldq)+4*offset],q5);

   }
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-1)]);
#endif

   q1 = _LOAD(&q[nb*ldq]);
   q2 = _LOAD(&q[(nb*ldq)+offset]);
   q3 = _LOAD(&q[(nb*ldq)+2*offset]);
   q4 = _LOAD(&q[(nb*ldq)+3*offset]);
   q5 = _LOAD(&q[(nb*ldq)+4*offset]);

#ifdef BLOCK2

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_FMA(x1, h1, q1);
   q2 = _SIMD_FMA(x2, h1, q2);
   q3 = _SIMD_FMA(x3, h1, q3);
   q4 = _SIMD_FMA(x4, h1, q4);
   q5 = _SIMD_FMA(x5, h1, q5);
#else
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
   q5 = _SIMD_ADD( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT x5, h1));
#endif
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
   q4 = _SIMD_NFMA(x4, h1, q4);
   q5 = _SIMD_NFMA(x5, h1, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT x5, h1));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
   q5 = _SIMD_NFMA(y5, h2, q5);
#else   
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT y5, h2));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
   q5 = _SIMD_NFMA(z5, h3, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5, h3));
#endif

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
   q5 = _SIMD_NFMA(w5, h4, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
   q4 = _SIMD_NFMA(v4, h5, q4);
   q5 = _SIMD_NFMA(v5, h5, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT v5, h5));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

   _STORE(&q[nb*ldq],q1);
   _STORE(&q[(nb*ldq)+offset],q2);
   _STORE(&q[(nb*ldq)+2*offset],q3);
   _STORE(&q[(nb*ldq)+3*offset],q4);
   _STORE(&q[(nb*ldq)+4*offset],q5);

#if defined(BLOCK4) || defined(BLOCK6)
   
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-2)], hh[nb-(BLOCK-2)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-2)]);
#endif

   q1 = _LOAD(&q[(nb+1)*ldq]);
   q2 = _LOAD(&q[((nb+1)*ldq)+offset]);
   q3 = _LOAD(&q[((nb+1)*ldq)+2*offset]);
   q4 = _LOAD(&q[((nb+1)*ldq)+3*offset]);
   q5 = _LOAD(&q[((nb+1)*ldq)+4*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
   q4 = _SIMD_NFMA(x4, h1, q4);
   q5 = _SIMD_NFMA(x5, h1, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT x5, h1));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-3)], hh[ldh+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
   q5 = _SIMD_NFMA(y5, h2, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT y5, h2));
#endif

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
   q5 = _SIMD_NFMA(z5, h3, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5, h3));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
   q5 = _SIMD_NFMA(w5, h4, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT w5, h4));
#endif /* __ELPA_USE_FMA__ */
#endif /* BLOCK6 */

   _STORE(&q[(nb+1)*ldq],q1);
   _STORE(&q[((nb+1)*ldq)+offset],q2);
   _STORE(&q[((nb+1)*ldq)+2*offset],q3);
   _STORE(&q[((nb+1)*ldq)+3*offset],q4);
   _STORE(&q[((nb+1)*ldq)+4*offset],q5);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-3)], hh[nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-3)]);
#endif

   q1 = _LOAD(&q[(nb+2)*ldq]);
   q2 = _LOAD(&q[((nb+2)*ldq)+offset]);
   q3 = _LOAD(&q[((nb+2)*ldq)+2*offset]);
   q4 = _LOAD(&q[((nb+2)*ldq)+3*offset]);
   q5 = _LOAD(&q[((nb+2)*ldq)+4*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
   q4 = _SIMD_NFMA(x4, h1, q4);
   q5 = _SIMD_NFMA(x5, h1, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT x5, h1));
#endif

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
   q5 = _SIMD_NFMA(y5, h2, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT y5, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
   q5 = _SIMD_NFMA(z5, h3, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT z5, h3));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

   _STORE(&q[(nb+2)*ldq],q1);
   _STORE(&q[((nb+2)*ldq)+offset],q2);
   _STORE(&q[((nb+2)*ldq)+2*offset],q3);
   _STORE(&q[((nb+2)*ldq)+3*offset],q4);
   _STORE(&q[((nb+2)*ldq)+4*offset],q5);

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-2], hh[nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-2]);
#endif

   q1 = _LOAD(&q[(nb+3)*ldq]);
   q2 = _LOAD(&q[((nb+3)*ldq)+offset]);
   q3 = _LOAD(&q[((nb+3)*ldq)+2*offset]);
   q4 = _LOAD(&q[((nb+3)*ldq)+3*offset]);
   q5 = _LOAD(&q[((nb+3)*ldq)+4*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
   q4 = _SIMD_NFMA(x4, h1, q4);
   q5 = _SIMD_NFMA(x5, h1, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT x5, h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
   q5 = _SIMD_NFMA(y5, h2, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT y5, h2));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[(nb+3)*ldq],q1);
   _STORE(&q[((nb+3)*ldq)+offset],q2);
   _STORE(&q[((nb+3)*ldq)+2*offset],q3);
   _STORE(&q[((nb+3)*ldq)+3*offset],q4);
   _STORE(&q[((nb+3)*ldq)+4*offset],q5);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-1], hh[nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-1]);
#endif

   q1 = _LOAD(&q[(nb+4)*ldq]);
   q2 = _LOAD(&q[((nb+4)*ldq)+offset]);
   q3 = _LOAD(&q[((nb+4)*ldq)+2*offset]);
   q4 = _LOAD(&q[((nb+4)*ldq)+3*offset]);
   q5 = _LOAD(&q[((nb+4)*ldq)+4*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
   q4 = _SIMD_NFMA(x4, h1, q4);
   q5 = _SIMD_NFMA(x5, h1, q5);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
   q5 = _SIMD_SUB( ADDITIONAL_ARGUMENT q5, _SIMD_MUL( ADDITIONAL_ARGUMENT x5, h1));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[(nb+4)*ldq],q1);
   _STORE(&q[((nb+4)*ldq)+offset],q2);
   _STORE(&q[((nb+4)*ldq)+2*offset],q3);
   _STORE(&q[((nb+4)*ldq)+3*offset],q4);
   _STORE(&q[((nb+4)*ldq)+4*offset],q5);

#endif /* BLOCK6 */
}


#undef ROW_LENGTH
#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 16
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 16
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 32
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 32
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 64
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */
/*
 * Unrolled kernel that computes
 * ROW_LENGTH rows of Q simultaneously, a
 * matrix Vector product with two householder
 */
#ifdef BLOCK2
/*
 * vectors + a rank 2 update is performed
 */
#endif
#if defined(BLOCK4) || defined(BLOCK6)
/*
 * vectors + a rank 1 update is performed
 */
#endif

__forceinline void CONCAT_8ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq, int ldh,
#ifdef BLOCK2
               DATA_TYPE s)
#endif
#ifdef BLOCK4
               DATA_TYPE s_1_2, DATA_TYPE s_1_3, DATA_TYPE s_2_3, DATA_TYPE s_1_4, DATA_TYPE s_2_4, DATA_TYPE s_3_4)
#endif
#ifdef BLOCK6
               DATA_TYPE_PTR scalarprods)
#endif
  {
#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Matrix Vector Multiplication, Q [ROW_LENGTH x nb+1] * hh
    // hh contains two householder vectors, with offset 1
    /////////////////////////////////////////////////////
#endif
#if defined(BLOCK4) || defined(BLOCK6)
    /////////////////////////////////////////////////////
    // Matrix Vector Multiplication, Q [ROW_LENGTH x nb+3] * hh
    // hh contains four householder vectors
    /////////////////////////////////////////////////////
#endif

    int i;
#ifdef BLOCK2
#if VEC_SET == SSE_128
    // Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == VSX_SSE
    __SIMD_DATATYPE sign = vec_splats(MONE);
#endif

#if VEC_SET == NEON_ARCH64_128
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = vdupq_n_f64(MONE);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = vdupq_n_f32(MONE);
#endif
#endif

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set1_epi64x(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if  VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_512 */

#if VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = svdup_f64(MONE);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = svdup_f32(MONE);
#endif
#endif

    __SIMD_DATATYPE x1 = _LOAD(&q[ldq]);
    __SIMD_DATATYPE x2 = _LOAD(&q[ldq+offset]);
    __SIMD_DATATYPE x3 = _LOAD(&q[ldq+2*offset]);
    __SIMD_DATATYPE x4 = _LOAD(&q[ldq+3*offset]);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h1 = _SIMD_SET1(hh[ldh+1]);
#endif
#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h1 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif
    __SIMD_DATATYPE h2;

#ifdef __ELPA_USE_FMA__
    __SIMD_DATATYPE q1 = _LOAD(q);
    __SIMD_DATATYPE y1 = _SIMD_FMA(x1, h1, q1);
    __SIMD_DATATYPE q2 = _LOAD(&q[offset]);
    __SIMD_DATATYPE y2 = _SIMD_FMA(x2, h1, q2);
    __SIMD_DATATYPE q3 = _LOAD(&q[2*offset]);
    __SIMD_DATATYPE y3 = _SIMD_FMA(x3, h1, q3);
    __SIMD_DATATYPE q4 = _LOAD(&q[3*offset]);
    __SIMD_DATATYPE y4 = _SIMD_FMA(x4, h1, q4);
#else
    __SIMD_DATATYPE q1 = _LOAD(q);
    __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
    __SIMD_DATATYPE q2 = _LOAD(&q[offset]);
    __SIMD_DATATYPE y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
    __SIMD_DATATYPE q3 = _LOAD(&q[2*offset]);
    __SIMD_DATATYPE y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
    __SIMD_DATATYPE q4 = _LOAD(&q[3*offset]);
    __SIMD_DATATYPE y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
#endif
#endif /* BLOCK2 */

#ifdef BLOCK4
    __SIMD_DATATYPE a1_1 = _LOAD(&q[ldq*3]);
    __SIMD_DATATYPE a2_1 = _LOAD(&q[ldq*2]);
    __SIMD_DATATYPE a3_1 = _LOAD(&q[ldq]);  
    __SIMD_DATATYPE a4_1 = _LOAD(&q[0]);    

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_2_1 = _SIMD_SET1(hh[ldh+1]);    
    __SIMD_DATATYPE h_3_2 = _SIMD_SET1(hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET1(hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_SET1(hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET1(hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_2_1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_2_1 = _SIMD_BROADCAST(&hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w1 = _SIMD_FMA(a3_1, h_4_3, a4_1);
    w1 = _SIMD_FMA(a2_1, h_4_2, w1);
    w1 = _SIMD_FMA(a1_1, h_4_1, w1);
    register __SIMD_DATATYPE z1 = _SIMD_FMA(a2_1, h_3_2, a3_1);
    z1 = _SIMD_FMA(a1_1, h_3_1, z1);
    register __SIMD_DATATYPE y1 = _SIMD_FMA(a1_1, h_2_1, a2_1);
    register __SIMD_DATATYPE x1 = a1_1;
#else
    register __SIMD_DATATYPE w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_4_3));
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_4_2));                          
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_4_1));                          
    register __SIMD_DATATYPE z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_3_2));
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_3_1));                          
    register __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_2_1));
    register __SIMD_DATATYPE x1 = a1_1;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE a1_2 = _LOAD(&q[(ldq*3)+offset]);                  
    __SIMD_DATATYPE a2_2 = _LOAD(&q[(ldq*2)+offset]);
    __SIMD_DATATYPE a3_2 = _LOAD(&q[ldq+offset]);
    __SIMD_DATATYPE a4_2 = _LOAD(&q[0+offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w2 = _SIMD_FMA(a3_2, h_4_3, a4_2);
    w2 = _SIMD_FMA(a2_2, h_4_2, w2);
    w2 = _SIMD_FMA(a1_2, h_4_1, w2);
    register __SIMD_DATATYPE z2 = _SIMD_FMA(a2_2, h_3_2, a3_2);
    z2 = _SIMD_FMA(a1_2, h_3_1, z2);
    register __SIMD_DATATYPE y2 = _SIMD_FMA(a1_2, h_2_1, a2_2);
    register __SIMD_DATATYPE x2 = a1_2;
#else
    register __SIMD_DATATYPE w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_4_3));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_4_2));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_4_1));
    register __SIMD_DATATYPE z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_3_2));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_3_1));
    register __SIMD_DATATYPE y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_2_1));
    register __SIMD_DATATYPE x2 = a1_2;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE a1_3 = _LOAD(&q[(ldq*3)+2*offset]);
    __SIMD_DATATYPE a2_3 = _LOAD(&q[(ldq*2)+2*offset]);
    __SIMD_DATATYPE a3_3 = _LOAD(&q[ldq+2*offset]);
    __SIMD_DATATYPE a4_3 = _LOAD(&q[0+2*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w3 = _SIMD_FMA(a3_3, h_4_3, a4_3);
    w3 = _SIMD_FMA(a2_3, h_4_2, w3);
    w3 = _SIMD_FMA(a1_3, h_4_1, w3);
    register __SIMD_DATATYPE z3 = _SIMD_FMA(a2_3, h_3_2, a3_3);
    z3 = _SIMD_FMA(a1_3, h_3_1, z3);
    register __SIMD_DATATYPE y3 = _SIMD_FMA(a1_3, h_2_1, a2_3);
    register __SIMD_DATATYPE x3 = a1_3;
#else
    register __SIMD_DATATYPE w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_3, h_4_3));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_4_2));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_4_1));
    register __SIMD_DATATYPE z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_3_2));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_3_1));
    register __SIMD_DATATYPE y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_2_1));
    register __SIMD_DATATYPE x3 = a1_3;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE a1_4 = _LOAD(&q[(ldq*3)+3*offset]);
    __SIMD_DATATYPE a2_4 = _LOAD(&q[(ldq*2)+3*offset]);
    __SIMD_DATATYPE a3_4 = _LOAD(&q[ldq+3*offset]);
    __SIMD_DATATYPE a4_4 = _LOAD(&q[0+3*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w4 = _SIMD_FMA(a3_4, h_4_3, a4_4);
    w4 = _SIMD_FMA(a2_4, h_4_2, w4);
    w4 = _SIMD_FMA(a1_4, h_4_1, w4);
    register __SIMD_DATATYPE z4 = _SIMD_FMA(a2_4, h_3_2, a3_4);
    z4 = _SIMD_FMA(a1_4, h_3_1, z4);
    register __SIMD_DATATYPE y4 = _SIMD_FMA(a1_4, h_2_1, a2_4);
    register __SIMD_DATATYPE x4 = a1_4;
#else
    register __SIMD_DATATYPE w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_4, h_4_3));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_4_2));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_4_1));
    register __SIMD_DATATYPE z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_3_2));
    z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_3_1));
    register __SIMD_DATATYPE y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_2_1));
    register __SIMD_DATATYPE x4 = a1_4;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE q1;
    __SIMD_DATATYPE q2;
    __SIMD_DATATYPE q3;
    __SIMD_DATATYPE q4;

    __SIMD_DATATYPE h1;
    __SIMD_DATATYPE h2;
    __SIMD_DATATYPE h3;
    __SIMD_DATATYPE h4;
#endif /* BLOCK4 */

#ifdef BLOCK6
    
    __SIMD_DATATYPE a1_1 = _LOAD(&q[ldq*5]);
    __SIMD_DATATYPE a2_1 = _LOAD(&q[ldq*4]);
    __SIMD_DATATYPE a3_1 = _LOAD(&q[ldq*3]);
    __SIMD_DATATYPE a4_1 = _LOAD(&q[ldq*2]);
    __SIMD_DATATYPE a5_1 = _LOAD(&q[ldq]);  
    __SIMD_DATATYPE a6_1 = _LOAD(&q[0]);    

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_6_5 = _SIMD_SET1(hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_SET1(hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_SET1(hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_SET1(hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_SET1(hh[(ldh*5)+5]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_6_5 = _SIMD_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_6_5 = _SIMD_BROADCAST(&hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_BROADCAST(&hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_BROADCAST(&hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_BROADCAST(&hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_BROADCAST(&hh[(ldh*5)+5]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t1 = _SIMD_FMA(a5_1, h_6_5, a6_1);
    t1 = _SIMD_FMA(a4_1, h_6_4, t1);
    t1 = _SIMD_FMA(a3_1, h_6_3, t1);
    t1 = _SIMD_FMA(a2_1, h_6_2, t1);
    t1 = _SIMD_FMA(a1_1, h_6_1, t1);
#else
    register __SIMD_DATATYPE t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_1, h_6_5)); 
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_1, h_6_4));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_6_3));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_6_2));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_6_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_5_4 = _SIMD_SET1(hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_SET1(hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_SET1(hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_SET1(hh[(ldh*4)+4]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_5_4 = _SIMD_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_5_4 = _SIMD_BROADCAST(&hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_BROADCAST(&hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_BROADCAST(&hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_BROADCAST(&hh[(ldh*4)+4]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE v1 = _SIMD_FMA(a4_1, h_5_4, a5_1);
    v1 = _SIMD_FMA(a3_1, h_5_3, v1);
    v1 = _SIMD_FMA(a2_1, h_5_2, v1);
    v1 = _SIMD_FMA(a1_1, h_5_1, v1);
#else
    register __SIMD_DATATYPE v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_1, h_5_4)); 
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_5_3));
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_5_2));
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_5_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_4_3 = _SIMD_SET1(hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET1(hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_4_3 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_4_3 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w1 = _SIMD_FMA(a3_1, h_4_3, a4_1);
    w1 = _SIMD_FMA(a2_1, h_4_2, w1);
    w1 = _SIMD_FMA(a1_1, h_4_1, w1);
#else
    register __SIMD_DATATYPE w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_4_3)); 
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_4_2));
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_4_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_2_1 = _SIMD_SET1(hh[ldh+1]);    
    __SIMD_DATATYPE h_3_2 = _SIMD_SET1(hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_2_1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_2_1 = _SIMD_BROADCAST(&hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE z1 = _SIMD_FMA(a2_1, h_3_2, a3_1);
    z1 = _SIMD_FMA(a1_1, h_3_1, z1);
    register __SIMD_DATATYPE y1 = _SIMD_FMA(a1_1, h_2_1, a2_1);
#else
    register __SIMD_DATATYPE z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_3_2));
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_3_1));
    register __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_2_1)); 
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x1 = a1_1;

    __SIMD_DATATYPE a1_2 = _LOAD(&q[(ldq*5)+offset]);
    __SIMD_DATATYPE a2_2 = _LOAD(&q[(ldq*4)+offset]);
    __SIMD_DATATYPE a3_2 = _LOAD(&q[(ldq*3)+offset]);
    __SIMD_DATATYPE a4_2 = _LOAD(&q[(ldq*2)+offset]);
    __SIMD_DATATYPE a5_2 = _LOAD(&q[(ldq)+offset]);
    __SIMD_DATATYPE a6_2 = _LOAD(&q[offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t2 = _SIMD_FMA(a5_2, h_6_5, a6_2);
    t2 = _SIMD_FMA(a4_2, h_6_4, t2);
    t2 = _SIMD_FMA(a3_2, h_6_3, t2);
    t2 = _SIMD_FMA(a2_2, h_6_2, t2);
    t2 = _SIMD_FMA(a1_2, h_6_1, t2);
    register __SIMD_DATATYPE v2 = _SIMD_FMA(a4_2, h_5_4, a5_2);
    v2 = _SIMD_FMA(a3_2, h_5_3, v2);
    v2 = _SIMD_FMA(a2_2, h_5_2, v2);
    v2 = _SIMD_FMA(a1_2, h_5_1, v2);
    register __SIMD_DATATYPE w2 = _SIMD_FMA(a3_2, h_4_3, a4_2);
    w2 = _SIMD_FMA(a2_2, h_4_2, w2);
    w2 = _SIMD_FMA(a1_2, h_4_1, w2);
    register __SIMD_DATATYPE z2 = _SIMD_FMA(a2_2, h_3_2, a3_2);
    z2 = _SIMD_FMA(a1_2, h_3_1, z2);
    register __SIMD_DATATYPE y2 = _SIMD_FMA(a1_2, h_2_1, a2_2);
#else
    register __SIMD_DATATYPE t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_2, h_6_5));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_2, h_6_4));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_6_3));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_6_2));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_6_1));
    register __SIMD_DATATYPE v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_2, h_5_4));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_5_3));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_5_2));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_5_1));
    register __SIMD_DATATYPE w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_4_3));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_4_2));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_4_1));
    register __SIMD_DATATYPE z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_3_2));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_3_1));
    register __SIMD_DATATYPE y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_2_1));
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x2 = a1_2;

    __SIMD_DATATYPE a1_3 = _LOAD(&q[(ldq*5)+2*offset]);
    __SIMD_DATATYPE a2_3 = _LOAD(&q[(ldq*4)+2*offset]);
    __SIMD_DATATYPE a3_3 = _LOAD(&q[(ldq*3)+2*offset]);
    __SIMD_DATATYPE a4_3 = _LOAD(&q[(ldq*2)+2*offset]);
    __SIMD_DATATYPE a5_3 = _LOAD(&q[(ldq)+2*offset]);
    __SIMD_DATATYPE a6_3 = _LOAD(&q[2*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t3 = _SIMD_FMA(a5_3, h_6_5, a6_3);
    t3 = _SIMD_FMA(a4_3, h_6_4, t3);
    t3 = _SIMD_FMA(a3_3, h_6_3, t3);
    t3 = _SIMD_FMA(a2_3, h_6_2, t3);
    t3 = _SIMD_FMA(a1_3, h_6_1, t3);
    register __SIMD_DATATYPE v3 = _SIMD_FMA(a4_3, h_5_4, a5_3);
    v3 = _SIMD_FMA(a3_3, h_5_3, v3);
    v3 = _SIMD_FMA(a2_3, h_5_2, v3);
    v3 = _SIMD_FMA(a1_3, h_5_1, v3);
    register __SIMD_DATATYPE w3 = _SIMD_FMA(a3_3, h_4_3, a4_3);
    w3 = _SIMD_FMA(a2_3, h_4_2, w3);
    w3 = _SIMD_FMA(a1_3, h_4_1, w3);
    register __SIMD_DATATYPE z3 = _SIMD_FMA(a2_3, h_3_2, a3_3);
    z3 = _SIMD_FMA(a1_3, h_3_1, z3);
    register __SIMD_DATATYPE y3 = _SIMD_FMA(a1_3, h_2_1, a2_3);
#else
    register __SIMD_DATATYPE t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_3, h_6_5));
    t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_3, h_6_4));
    t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_3, h_6_3));
    t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_6_2));
    t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_6_1));
    register __SIMD_DATATYPE v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_3, h_5_4));
    v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_3, h_5_3));
    v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_5_2));
    v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_5_1));
    register __SIMD_DATATYPE w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_3, h_4_3));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_4_2));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_4_1));
    register __SIMD_DATATYPE z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_3_2));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_3_1));
    register __SIMD_DATATYPE y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_2_1));
#endif /* __ELPA_USE_FMA__ */
 
    register __SIMD_DATATYPE x3 = a1_3;

    __SIMD_DATATYPE a1_4 = _LOAD(&q[(ldq*5)+3*offset]);
    __SIMD_DATATYPE a2_4 = _LOAD(&q[(ldq*4)+3*offset]);
    __SIMD_DATATYPE a3_4 = _LOAD(&q[(ldq*3)+3*offset]);
    __SIMD_DATATYPE a4_4 = _LOAD(&q[(ldq*2)+3*offset]);
    __SIMD_DATATYPE a5_4 = _LOAD(&q[(ldq)+3*offset]);
    __SIMD_DATATYPE a6_4 = _LOAD(&q[3*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t4 = _SIMD_FMA(a5_4, h_6_5, a6_4);
    t4 = _SIMD_FMA(a4_4, h_6_4, t4);
    t4 = _SIMD_FMA(a3_4, h_6_3, t4);
    t4 = _SIMD_FMA(a2_4, h_6_2, t4);
    t4 = _SIMD_FMA(a1_4, h_6_1, t4);
    register __SIMD_DATATYPE v4 = _SIMD_FMA(a4_4, h_5_4, a5_4);
    v4 = _SIMD_FMA(a3_4, h_5_3, v4);
    v4 = _SIMD_FMA(a2_4, h_5_2, v4);
    v4 = _SIMD_FMA(a1_4, h_5_1, v4);
    register __SIMD_DATATYPE w4 = _SIMD_FMA(a3_4, h_4_3, a4_4);
    w4 = _SIMD_FMA(a2_4, h_4_2, w4);
    w4 = _SIMD_FMA(a1_4, h_4_1, w4);
    register __SIMD_DATATYPE z4 = _SIMD_FMA(a2_4, h_3_2, a3_4);
    z4 = _SIMD_FMA(a1_4, h_3_1, z4);
    register __SIMD_DATATYPE y4 = _SIMD_FMA(a1_4, h_2_1, a2_4);
#else
    register __SIMD_DATATYPE t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_4, h_6_5));
    t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT t4, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_4, h_6_4));
    t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT t4, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_4, h_6_3));
    t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT t4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_6_2));
    t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT t4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_6_1));
    register __SIMD_DATATYPE v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_4, h_5_4));
    v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT v4, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_4, h_5_3));
    v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT v4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_5_2));
    v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT v4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_5_1));
    register __SIMD_DATATYPE w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_4, h_4_3));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_4_2));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_4_1));
    register __SIMD_DATATYPE z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_4, h_3_2));
    z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_3_1));
    register __SIMD_DATATYPE y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_4, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_4, h_2_1));
#endif /* __ELPA_USE_FMA__ */
 
    register __SIMD_DATATYPE x4 = a1_4;


    __SIMD_DATATYPE q1;
    __SIMD_DATATYPE q2;
    __SIMD_DATATYPE q3;
    __SIMD_DATATYPE q4;

    __SIMD_DATATYPE h1;
    __SIMD_DATATYPE h2;
    __SIMD_DATATYPE h3;
    __SIMD_DATATYPE h4;
    __SIMD_DATATYPE h5;
    __SIMD_DATATYPE h6;

#endif /* BLOCK6 */

    for(i = BLOCK; i < nb; i++)
      {
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h1 = _SIMD_SET1(hh[i-(BLOCK-1)]);
        h2 = _SIMD_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == SPARC64_SSE
        h1 = _SIMD_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
        h2 = _SIMD_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif
#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h1 = _SIMD_BROADCAST(&hh[i-(BLOCK-1)]);
        h2 = _SIMD_BROADCAST(&hh[ldh+i-(BLOCK-2)]);
#endif /*   VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

        q1 = _LOAD(&q[i*ldq]);
        q2 = _LOAD(&q[(i*ldq)+offset]);
        q3 = _LOAD(&q[(i*ldq)+2*offset]);
        q4 = _LOAD(&q[(i*ldq)+3*offset]);

#ifdef __ELPA_USE_FMA__
        x1 = _SIMD_FMA(q1, h1, x1);
        y1 = _SIMD_FMA(q1, h2, y1);
        x2 = _SIMD_FMA(q2, h1, x2);
        y2 = _SIMD_FMA(q2, h2, y2);
        x3 = _SIMD_FMA(q3, h1, x3);
        y3 = _SIMD_FMA(q3, h2, y3);
        x4 = _SIMD_FMA(q4, h1, x4);
        y4 = _SIMD_FMA(q4, h2, y4);
#else
        x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
        y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
        x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
        y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
        x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
        y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
        x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
        y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
#endif /* __ELPA_USE_FMA__ */

#if defined(BLOCK4) || defined(BLOCK6)
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h3 = _SIMD_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
        h3 = _SIMD_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h3 = _SIMD_BROADCAST(&hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
        z1 = _SIMD_FMA(q1, h3, z1);
        z2 = _SIMD_FMA(q2, h3, z2);
        z3 = _SIMD_FMA(q3, h3, z3);
        z4 = _SIMD_FMA(q4, h3, z4);
#else
        z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
        z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
        z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h3));
        z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h4 = _SIMD_SET1(hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == SPARC64_SSE
        h4 = _SIMD_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h4 = _SIMD_BROADCAST(&hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
        w1 = _SIMD_FMA(q1, h4, w1);
        w2 = _SIMD_FMA(q2, h4, w2);
        w3 = _SIMD_FMA(q3, h4, w3);
        w4 = _SIMD_FMA(q4, h4, w4);
#else
        w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4));
        w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h4));
        w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h4));
        w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h4));
#endif /* __ELPA_USE_FMA__ */
		
#endif /* BLOCK4 || BLOCK6 */
#ifdef BLOCK6

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h5 = _SIMD_SET1(hh[(ldh*4)+i-(BLOCK-5)]);
#endif
#if VEC_SET == SPARC64_SSE
        h5 = _SIMD_SET(hh[(ldh*4)+i-(BLOCK-5)], hh[(ldh*4)+i-(BLOCK-5)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h5 = _SIMD_BROADCAST(&hh[(ldh*4)+i-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
        v1 = _SIMD_FMA(q1, h5, v1);
        v2 = _SIMD_FMA(q2, h5, v2);
        v3 = _SIMD_FMA(q3, h5, v3);
        v4 = _SIMD_FMA(q4, h5, v4);
#else
        v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h5));
        v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h5));
        v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h5));
        v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT v4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h6 = _SIMD_SET1(hh[(ldh*5)+i]);
#endif

#if VEC_SET == SPARC64_SSE
        h6 = _SIMD_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
	h6 = _SIMD_BROADCAST(&hh[(ldh*5)+i-(BLOCK-6)]);
#endif

#ifdef __ELPA_USE_FMA__
        t1 = _SIMD_FMA(q1, h6, t1);
        t2 = _SIMD_FMA(q2, h6, t2);
        t3 = _SIMD_FMA(q3, h6, t3);
        t4 = _SIMD_FMA(q4, h6, t4);
#else
        t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h6));
        t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h6));
        t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h6));
        t4 = _SIMD_ADD( ADDITIONAL_ARGUMENT t4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h6));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */
      }
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-1)]);
#endif

    q1 = _LOAD(&q[nb*ldq]);
    q2 = _LOAD(&q[(nb*ldq)+offset]);
    q3 = _LOAD(&q[(nb*ldq)+2*offset]);
    q4 = _LOAD(&q[(nb*ldq)+3*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
#endif /* __ELPA_USE_FMA__ */

#if defined(BLOCK4) || defined(BLOCK6)
    
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
    y4 = _SIMD_FMA(q4, h2, y4);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
    y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
    z2 = _SIMD_FMA(q2, h3, z2);
    z3 = _SIMD_FMA(q3, h3, z3);
    z4 = _SIMD_FMA(q4, h3, z4);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h3));
    z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h3));
#endif

#ifdef BLOCK4
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-2]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-2], hh[nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-2]);
#endif

    q1 = _LOAD(&q[(nb+1)*ldq]);
    q2 = _LOAD(&q[((nb+1)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+1)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+1)*ldq)+3*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[(ldh*1)+nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[(ldh*1)+nb-1], hh[(ldh*1)+nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[(ldh*1)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
    y4 = _SIMD_FMA(q4, h2, y4);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
    y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-1], hh[nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-1]);
#endif

    q1 = _LOAD(&q[(nb+2)*ldq]);
    q2 = _LOAD(&q[((nb+2)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+2)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+2)*ldq)+3*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 */
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h4 = _SIMD_SET1(hh[(ldh*3)+nb-(BLOCK-4)]);
#endif

#if VEC_SET == SPARC64_SSE
    h4 = _SIMD_SET(hh[(ldh*3)+nb-(BLOCK-4)], hh[(ldh*3)+nb-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    w1 = _SIMD_FMA(q1, h4, w1);
    w2 = _SIMD_FMA(q2, h4, w2);
    w3 = _SIMD_FMA(q3, h4, w3);
    w4 = _SIMD_FMA(q4, h4, w4);
#else
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4)); 
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h4));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h4));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h5 = _SIMD_SET1(hh[(ldh*4)+nb-(BLOCK-5)]);
#endif

#if VEC_SET == SPARC64_SSE
    h5 = _SIMD_SET(hh[(ldh*4)+nb-(BLOCK-5)], hh[(ldh*4)+nb-(BLOCK-5)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h5 = _SIMD_BROADCAST(&hh[(ldh*4)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    v1 = _SIMD_FMA(q1, h5, v1);
    v2 = _SIMD_FMA(q2, h5, v2);
    v3 = _SIMD_FMA(q3, h5, v3);
    v4 = _SIMD_FMA(q4, h5, v4);
#else
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h5));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h5));
    v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h5));
    v4 = _SIMD_ADD( ADDITIONAL_ARGUMENT v4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-(BLOCK-2)]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-(BLOCK-2)], hh[nb-(BLOCK-2)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-2)]);
#endif

    q1 = _LOAD(&q[(nb+1)*ldq]);
    q2 = _LOAD(&q[((nb+1)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+1)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+1)*ldq)+3*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-3)]);
#endif
#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-3)], hh[ldh+nb-(BLOCK-3)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
    y4 = _SIMD_FMA(q4, h2, y4);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
    y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-4)]);
#endif
#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-4)], hh[(ldh*2)+nb-(BLOCK-4)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
    z2 = _SIMD_FMA(q2, h3, z2);
    z3 = _SIMD_FMA(q3, h3, z3);
    z4 = _SIMD_FMA(q4, h3, z4);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h3));
    z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h4 = _SIMD_SET1(hh[(ldh*3)+nb-(BLOCK-5)]);
#endif
#if VEC_SET == SPARC64_SSE
    h4 = _SIMD_SET(hh[(ldh*3)+nb-(BLOCK-5)], hh[(ldh*3)+nb-(BLOCK-5)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    w1 = _SIMD_FMA(q1, h4, w1);
    w2 = _SIMD_FMA(q2, h4, w2);
    w3 = _SIMD_FMA(q3, h4, w3);
    w4 = _SIMD_FMA(q4, h4, w4);
#else
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h4));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h4));
    w4 = _SIMD_ADD( ADDITIONAL_ARGUMENT w4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-(BLOCK-3)]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-(BLOCK-3)], hh[nb-(BLOCK-3)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-3)]);
#endif

    q1 = _LOAD(&q[(nb+2)*ldq]);
    q2 = _LOAD(&q[((nb+2)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+2)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+2)*ldq)+3*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-4)]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-4)], hh[ldh+nb-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
    y4 = _SIMD_FMA(q4, h2, y4);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
    y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-5)]);
#endif

#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-5)], hh[(ldh*2)+nb-(BLOCK-5)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
    z2 = _SIMD_FMA(q2, h3, z2);
    z3 = _SIMD_FMA(q3, h3, z3);
    z4 = _SIMD_FMA(q4, h3, z4);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h3));
    z4 = _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-(BLOCK-4)]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-(BLOCK-4)], hh[nb-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-4)]);
#endif

    q1 = _LOAD(&q[(nb+3)*ldq]);
    q2 = _LOAD(&q[((nb+3)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+3)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+3)*ldq)+3*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-5)]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-5)], hh[ldh+nb-(BLOCK-5)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
    y4 = _SIMD_FMA(q4, h2, y4);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
    y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT y4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-(BLOCK-5)]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-(BLOCK-5)], hh[nb-(BLOCK-5)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-5)]);
#endif

    q1 = _LOAD(&q[(nb+4)*ldq]);
    q2 = _LOAD(&q[((nb+4)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+4)*ldq)+2*offset]);
    q4 = _LOAD(&q[((nb+4)*ldq)+3*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
    x4 = _SIMD_FMA(q4, h1, x4);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
    x4 = _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT q4,h1));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Rank-2 update of Q [ ROW_LENGTH x nb+1]
    /////////////////////////////////////////////////////
#endif
#ifdef BLOCK4
    /////////////////////////////////////////////////////
    // Rank-1 update of Q [ ROW_LENGTH x nb+3]
    /////////////////////////////////////////////////////
#endif
#ifdef BLOCK6
    /////////////////////////////////////////////////////
    // Apply tau, correct wrong calculation using pre-calculated scalar products
    /////////////////////////////////////////////////////
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE tau1 = _SIMD_SET1(hh[0]);

    __SIMD_DATATYPE tau2 = _SIMD_SET1(hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_SET1(hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_SET1(hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_SET1(hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_SET1(hh[ldh*5]);       
#endif

#ifdef BLOCK2    
    __SIMD_DATATYPE vs = _SIMD_SET1(s);
#endif
#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET1(s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET1(s_1_3);  
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET1(s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET1(s_1_4);  
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET1(s_2_4);  
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET1(s_3_4);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET1(scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET1(scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET1(scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET1(scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET1(scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET1(scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_SET1(scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_SET1(scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_SET1(scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_SET1(scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_SET1(scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_SET1(scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_SET1(scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_SET1(scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_SET1(scalarprods[14]);
#endif
#endif /* VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE */

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE tau1 = _SIMD_SET(hh[0], hh[0]);

    __SIMD_DATATYPE tau2 = _SIMD_SET(hh[ldh], hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_SET(hh[ldh*2], hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_SET(hh[ldh*3], hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_SET(hh[ldh*4], hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_SET(hh[ldh*5], hh[ldh*5]);
#endif

#ifdef BLOCK2
    __SIMD_DATATYPE vs = _SIMD_SET(s, s);
#endif
#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET(s_1_2, s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET(s_1_3, s_1_3);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET(s_2_3, s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET(s_1_4, s_1_4);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET(s_2_4, s_2_4);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET(s_3_4, s_3_4);

#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET(scalarprods[0], scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET(scalarprods[1], scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET(scalarprods[2], scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET(scalarprods[3], scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET(scalarprods[4], scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET(scalarprods[5], scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_SET(scalarprods[6], scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_SET(scalarprods[7], scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_SET(scalarprods[8], scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_SET(scalarprods[9], scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_SET(scalarprods[10], scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_SET(scalarprods[11], scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_SET(scalarprods[12], scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_SET(scalarprods[13], scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_SET(scalarprods[14], scalarprods[14]);
#endif
#endif /* VEC_SET == SPARC64_SSE */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   __SIMD_DATATYPE tau1 = _SIMD_BROADCAST(hh);
   __SIMD_DATATYPE tau2 = _SIMD_BROADCAST(&hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_BROADCAST(&hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_BROADCAST(&hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_BROADCAST(&hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_BROADCAST(&hh[ldh*5]);
#endif

#ifdef BLOCK2  
   __SIMD_DATATYPE vs = _SIMD_BROADCAST(&s);
#endif

#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_BROADCAST(&s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_BROADCAST(&s_1_3);
   __SIMD_DATATYPE vs_2_3 = _SIMD_BROADCAST(&s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_BROADCAST(&s_1_4);
   __SIMD_DATATYPE vs_2_4 = _SIMD_BROADCAST(&s_2_4);
   __SIMD_DATATYPE vs_3_4 = _SIMD_BROADCAST(&s_3_4);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_BROADCAST(&scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_BROADCAST(&scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_BROADCAST(&scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_BROADCAST(&scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_BROADCAST(&scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_BROADCAST(&scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_BROADCAST(&scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_BROADCAST(&scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_BROADCAST(&scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_BROADCAST(&scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_BROADCAST(&scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_BROADCAST(&scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_BROADCAST(&scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_BROADCAST(&scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_BROADCAST(&scalarprods[14]);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == VSX_SSE || VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _XOR(tau1, sign);
#endif

#if VEC_SET == SPARC64_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
    h1 = _SIMD_NEG( ADDITIONAL_ARGUMENT tau1);
#endif

#if VEC_SET == AVX_512
#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi64((__SIMD_INTEGER) tau1, (__SIMD_INTEGER) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi32((__SIMD_INTEGER) tau1, (__SIMD_INTEGER) sign);
#endif
#endif /* HAVE_AVX512_XEON_PHI */

#ifdef HAVE_AVX512_XEON
    h1 = _XOR(tau1, sign);
#endif
#endif /* VEC_SET == AVX_512 */

#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau1;
#endif

   x1 = _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1);
   x2 = _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1);
   x3 = _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1);
   x4 = _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1);

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == VSX_SSE || VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _XOR(tau2, sign);
#endif

#if VEC_SET == SPARC64_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
   h1 = _SIMD_NEG( ADDITIONAL_ARGUMENT tau2);
#endif

#if VEC_SET == AVX_512
#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi64((__SIMD_INTEGER) tau2, (__SIMD_INTEGER) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi32((__SIMD_INTEGER) tau2, (__SIMD_INTEGER) sign);
#endif
#endif /* HAVE_AVX512_XEON_PHI */

#ifdef HAVE_AVX512_XEON
    h1 = _XOR(tau2, sign);
#endif
#endif /* VEC_SET == AVX_512 */

   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs);
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau2;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_2);
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK2

#ifdef __ELPA_USE_FMA__
   y1 = _SIMD_FMA(y1, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_FMA(y2, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
   y3 = _SIMD_FMA(y3, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2));
   y4 = _SIMD_FMA(y4, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2));
#else
   y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
   y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2));
   y4 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2));
#endif
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)

#ifdef __ELPA_USE_FMA__
   y1 = _SIMD_FMSUB(y1, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_FMSUB(y2, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
   y3 = _SIMD_FMSUB(y3, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2));
   y4 = _SIMD_FMSUB(y4, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2));
#else   
   y1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
   y3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2));
   y4 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2));
#endif /* __ELPA_USE_FMA__ */

   h1 = tau3;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_3);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_2_3);

#ifdef __ELPA_USE_FMA__
   z1 = _SIMD_FMSUB(z1, h1, _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)));
   z2 = _SIMD_FMSUB(z2, h1, _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)));
   z3 = _SIMD_FMSUB(z3, h1, _SIMD_FMA(y3, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2)));
   z4 = _SIMD_FMSUB(z4, h1, _SIMD_FMA(y4, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2)));
#else
   z1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)));
   z2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)));
   z3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2)));
   z4 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2)));
#endif /* __ELPA_USE_FMA__ */

   h1 = tau4;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_4);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_2_4);
   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_3_4);

#ifdef __ELPA_USE_FMA__
   w1 = _SIMD_FMSUB(w1, h1, _SIMD_FMA(z1, h4, _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   w2 = _SIMD_FMSUB(w2, h1, _SIMD_FMA(z2, h4, _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
   w3 = _SIMD_FMSUB(w3, h1, _SIMD_FMA(z3, h4, _SIMD_FMA(y3, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2))));
   w4 = _SIMD_FMSUB(w4, h1, _SIMD_FMA(z4, h4, _SIMD_FMA(y4, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2))));
#else
   w1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   w2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
   w3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w3,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2))));
   w4 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w4,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2))));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_1_5); 
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_2_5);
   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_3_5);
   h5 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_4_5);

#ifdef __ELPA_USE_FMA__
   v1 = _SIMD_FMSUB(v1, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w1, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   v2 = _SIMD_FMSUB(v2, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w2, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
   v3 = _SIMD_FMSUB(v3, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w3, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4)), _SIMD_FMA(y3, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2))));
   v4 = _SIMD_FMSUB(v4, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w4, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h4)), _SIMD_FMA(y4, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2))));
#else
   v1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v1,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   v2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v2,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
   v3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v3,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w3,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2))));
   v4 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v4,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w4,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2))));
#endif /* __ELPA_USE_FMA__ */

   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_1_6);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_2_6);
   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_3_6);
   h5 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_4_6);
   h6 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_5_6);

#ifdef __ELPA_USE_FMA__
   t1 = _SIMD_FMSUB(t1, tau6, _SIMD_FMA(v1, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w1, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)))));
   t2 = _SIMD_FMSUB(t2, tau6, _SIMD_FMA(v2, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w2, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)))));
   t3 = _SIMD_FMSUB(t3, tau6, _SIMD_FMA(v3, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w3, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4)), _SIMD_FMA(y3, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2)))));
   t4 = _SIMD_FMSUB(t4, tau6, _SIMD_FMA(v4, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w4, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h4)), _SIMD_FMA(y4, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2)))));
#else
   t1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t1,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v1,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)))));
   t2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t2,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v2,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)))));
   t3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t3,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v3,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w3,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2)))));
   t4 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t4,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v4,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w4,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h2)))));
#endif /* __ELPA_USE_FMA__ */

    /////////////////////////////////////////////////////
   // Rank-1 update of Q [ ROW_LENGTH x nb+3]
   /////////////////////////////////////////////////////
#endif /* BLOCK6 */

   q1 = _LOAD(&q[0]);
   q2 = _LOAD(&q[offset]);
   q3 = _LOAD(&q[2*offset]);
   q4 = _LOAD(&q[3*offset]);

#ifdef BLOCK2
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, y1);
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, y2);
   q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, y3);
   q4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, y4);
#endif
#ifdef BLOCK4
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, w1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, w2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, w3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, w4);
#endif
#ifdef BLOCK6
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, t1); 
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, t2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, t3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, t4);
#endif

   _STORE(&q[0],q1);
   _STORE(&q[offset],q2);
   _STORE(&q[2*offset],q3);
   _STORE(&q[3*offset],q4);

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif

   q1 = _LOAD(&q[ldq]);
   q2 = _LOAD(&q[ldq+offset]);
   q3 = _LOAD(&q[ldq+2*offset]);
   q4 = _LOAD(&q[ldq+3*offset]);
 
#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_FMA(y1, h2, x1));
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_FMA(y2, h2, x2));
   q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_FMA(y3, h2, x3));
   q4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, _SIMD_FMA(y4, h2, x4));
#else
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2)));
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2)));
   q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2)));
   q4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, _SIMD_ADD( ADDITIONAL_ARGUMENT x4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2)));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq],q1);
   _STORE(&q[ldq+offset],q2);
   _STORE(&q[ldq+2*offset],q3);
   _STORE(&q[ldq+3*offset],q4);
#endif /* BLOCK2 */

#ifdef BLOCK4
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
#endif
   q1 = _LOAD(&q[ldq]);
   q2 = _LOAD(&q[ldq+offset]);
   q3 = _LOAD(&q[ldq+2*offset]);
   q4 = _LOAD(&q[ldq+3*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_FMA(w1, h4, z1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_FMA(w2, h4, z2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_FMA(w3, h4, z3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_FMA(w4, h4, z4));
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4)));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4)));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4)));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_ADD( ADDITIONAL_ARGUMENT z4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4)));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq],q1);
   _STORE(&q[ldq+offset],q2);
   _STORE(&q[ldq+2*offset],q3);
   _STORE(&q[ldq+3*offset],q4);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+2]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
#endif

   q1 = _LOAD(&q[ldq*2]);
   q2 = _LOAD(&q[(ldq*2)+offset]);
   q3 = _LOAD(&q[(ldq*2)+2*offset]);
   q4 = _LOAD(&q[(ldq*2)+3*offset]);

   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, y1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, y2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, y3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, y4);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*2],q1);
   _STORE(&q[(ldq*2)+offset],q2);
   _STORE(&q[(ldq*2)+2*offset],q3);
   _STORE(&q[(ldq*2)+3*offset],q4);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

   q1 = _LOAD(&q[ldq*3]);
   q2 = _LOAD(&q[(ldq*3)+offset]);
   q3 = _LOAD(&q[(ldq*3)+2*offset]);
   q4 = _LOAD(&q[(ldq*3)+3*offset]);

   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, x1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, x2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, x3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, x4);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*3], q1);
   _STORE(&q[(ldq*3)+offset], q2);
   _STORE(&q[(ldq*3)+2*offset], q3);
   _STORE(&q[(ldq*3)+3*offset], q4);

#endif /* BLOCK4 */

#ifdef BLOCK6
   
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+1]);
#endif

   q1 = _LOAD(&q[ldq]);
   q2 = _LOAD(&q[(ldq+offset)]);
   q3 = _LOAD(&q[(ldq+2*offset)]);
   q4 = _LOAD(&q[(ldq+3*offset)]);

   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, v1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, v2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, v3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, v4);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
   q4 = _SIMD_NFMA(t4, h6, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
#endif

   _STORE(&q[ldq],q1);
   _STORE(&q[(ldq+offset)],q2);
   _STORE(&q[(ldq+2*offset)],q3);
   _STORE(&q[(ldq+3*offset)],q4);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+1]);
#endif
   q1 = _LOAD(&q[ldq*2]);
   q2 = _LOAD(&q[(ldq*2)+offset]);
   q3 = _LOAD(&q[(ldq*2)+2*offset]);
   q4 = _LOAD(&q[(ldq*2)+3*offset]);
 
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, w1); 
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, w2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, w3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, w4);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
   q4 = _SIMD_NFMA(v4, h5, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5)); 
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));  
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));  
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
   q4 = _SIMD_NFMA(t4, h6, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*2],q1);
   _STORE(&q[(ldq*2)+offset],q2);
   _STORE(&q[(ldq*2)+2*offset],q3);
   _STORE(&q[(ldq*2)+3*offset],q4);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
#endif

   q1 = _LOAD(&q[ldq*3]);
   q2 = _LOAD(&q[(ldq*3)+offset]);
   q3 = _LOAD(&q[(ldq*3)+2*offset]);
   q4 = _LOAD(&q[(ldq*3)+3*offset]);

   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, z1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, z2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, z3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, z4);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
   q4 = _SIMD_NFMA(v4, h5, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
   q4 = _SIMD_NFMA(t4, h6, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
#endif

   _STORE(&q[ldq*3],q1);
   _STORE(&q[(ldq*3)+offset],q2);
   _STORE(&q[(ldq*3)+2*offset],q3);
   _STORE(&q[(ldq*3)+3*offset],q4);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
#endif

   q1 = _LOAD(&q[ldq*4]);
   q2 = _LOAD(&q[(ldq*4)+offset]);
   q3 = _LOAD(&q[(ldq*4)+2*offset]);
   q4 = _LOAD(&q[(ldq*4)+3*offset]);

   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, y1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, y2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, y3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, y4);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
   q4 = _SIMD_NFMA(v4, h5, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+4]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+4]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
   q4 = _SIMD_NFMA(t4, h6, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
#endif

   _STORE(&q[ldq*4],q1);
   _STORE(&q[(ldq*4)+offset],q2);
   _STORE(&q[(ldq*4)+2*offset],q3);
   _STORE(&q[(ldq*4)+3*offset],q4);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[(ldh)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[(ldh)+1], hh[(ldh)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[(ldh)+1]);
#endif
   q1 = _LOAD(&q[ldq*5]);
   q2 = _LOAD(&q[(ldq*5)+offset]);
   q3 = _LOAD(&q[(ldq*5)+2*offset]);
   q4 = _LOAD(&q[(ldq*5)+3*offset]);

   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, x1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, x2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, x3);
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, x4);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+4]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+4]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
   q4 = _SIMD_NFMA(v4, h5, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+5]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+5]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
   q4 = _SIMD_NFMA(t4, h6, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
#endif

   _STORE(&q[ldq*5],q1);
   _STORE(&q[(ldq*5)+offset],q2);
   _STORE(&q[(ldq*5)+2*offset],q3);
   _STORE(&q[(ldq*5)+3*offset],q4);

#endif /* BLOCK6 */

   for (i = BLOCK; i < nb; i++)
   {
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h1 = _SIMD_SET1(hh[i-(BLOCK-1)]);
     h2 = _SIMD_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == SPARC64_SSE
     h1 = _SIMD_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
     h2 = _SIMD_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[i-(BLOCK-1)]);
    h2 = _SIMD_BROADCAST(&hh[ldh+i-(BLOCK-2)]);
#endif

     q1 = _LOAD(&q[i*ldq]);
     q2 = _LOAD(&q[(i*ldq)+offset]);
     q3 = _LOAD(&q[(i*ldq)+2*offset]);
     q4 = _LOAD(&q[(i*ldq)+3*offset]);

#ifdef BLOCK2
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_FMA(x1, h1, q1);
     q1 = _SIMD_FMA(y1, h2, q1);
     q2 = _SIMD_FMA(x2, h1, q2);
     q2 = _SIMD_FMA(y2, h2, q2);
     q3 = _SIMD_FMA(x3, h1, q3);
     q3 = _SIMD_FMA(y3, h2, q3);
     q4 = _SIMD_FMA(x4, h1, q4);
     q4 = _SIMD_FMA(y4, h2, q4);
#else
     q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2)));
     q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2)));
     q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2)));
     q4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2)));
#endif /* __ELPA_USE_FMA__ */
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
     
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(x1, h1, q1);
     q2 = _SIMD_NFMA(x2, h1, q2);
     q3 = _SIMD_NFMA(x3, h1, q3);
     q4 = _SIMD_NFMA(x4, h1, q4);
#else   
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h1));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h1));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h1));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4,h1));
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(y1, h2, q1);
     q2 = _SIMD_NFMA(y2, h2, q2);
     q3 = _SIMD_NFMA(y3, h2, q3);
     q4 = _SIMD_NFMA(y4, h2, q4);
#else    
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h2));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h2));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h2));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h3 = _SIMD_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
     h3 = _SIMD_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h3 = _SIMD_BROADCAST(&hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(z1, h3, q1);
     q2 = _SIMD_NFMA(z2, h3, q2);
     q3 = _SIMD_NFMA(z3, h3, q3);
     q4 = _SIMD_NFMA(z4, h3, q4);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h3));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h3));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h3));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h4 = _SIMD_SET1(hh[(ldh*3)+i-(BLOCK-4)]); 
#endif

#if VEC_SET == SPARC64_SSE
     h4 = _SIMD_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h4 = _SIMD_BROADCAST(&hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(w1, h4, q1);
     q2 = _SIMD_NFMA(w2, h4, q2);
     q3 = _SIMD_NFMA(w3, h4, q3);
     q4 = _SIMD_NFMA(w4, h4, q4);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h4));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h4));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3,h4));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4,h4));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 || BLOCK6*/

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h5 = _SIMD_SET1(hh[(ldh*4)+i-1]);
#endif
#if VEC_SET == SPARC64_SSE
     h5 = _SIMD_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h5 = _SIMD_BROADCAST(&hh[(ldh*4)+i-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(v1, h5, q1);
     q2 = _SIMD_NFMA(v2, h5, q2);
     q3 = _SIMD_NFMA(v3, h5, q3);
     q4 = _SIMD_NFMA(v4, h5, q4);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));
#endif /* __ELPA_USE_FMA__ */
 
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h6 = _SIMD_SET1(hh[(ldh*5)+i]);
#endif
#if VEC_SET == SPARC64_SSE
     h6 = _SIMD_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h6 = _SIMD_BROADCAST(&hh[(ldh*5)+i-(BLOCK-6)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(t1, h6, q1);
     q2 = _SIMD_NFMA(t2, h6, q2);
     q3 = _SIMD_NFMA(t3, h6, q3);
     q4 = _SIMD_NFMA(t4, h6, q4);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
     q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT t4, h6));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

     _STORE(&q[i*ldq],q1);
     _STORE(&q[(i*ldq)+offset],q2);
     _STORE(&q[(i*ldq)+2*offset],q3);
     _STORE(&q[(i*ldq)+3*offset],q4);

   }
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-1)]);
#endif

   q1 = _LOAD(&q[nb*ldq]);
   q2 = _LOAD(&q[(nb*ldq)+offset]);
   q3 = _LOAD(&q[(nb*ldq)+2*offset]);
   q4 = _LOAD(&q[(nb*ldq)+3*offset]);

#ifdef BLOCK2

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_FMA(x1, h1, q1);
   q2 = _SIMD_FMA(x2, h1, q2);
   q3 = _SIMD_FMA(x3, h1, q3);
   q4 = _SIMD_FMA(x4, h1, q4);
#else
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_ADD( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
#endif
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
   q4 = _SIMD_NFMA(x4, h1, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
#else   
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
#endif

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
   q4 = _SIMD_NFMA(v4, h5, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT v4, h5));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

   _STORE(&q[nb*ldq],q1);
   _STORE(&q[(nb*ldq)+offset],q2);
   _STORE(&q[(nb*ldq)+2*offset],q3);
   _STORE(&q[(nb*ldq)+3*offset],q4);

#if defined(BLOCK4) || defined(BLOCK6)
   
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-2)], hh[nb-(BLOCK-2)]);
#endif
  
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-2)]);
#endif

   q1 = _LOAD(&q[(nb+1)*ldq]);
   q2 = _LOAD(&q[((nb+1)*ldq)+offset]);
   q3 = _LOAD(&q[((nb+1)*ldq)+2*offset]);
   q4 = _LOAD(&q[((nb+1)*ldq)+3*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
   q4 = _SIMD_NFMA(x4, h1, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-3)], hh[ldh+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
#endif

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
#else 
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
   q4 = _SIMD_NFMA(w4, h4, q4);
#else   
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT w4, h4));
#endif /* __ELPA_USE_FMA__ */
#endif /* BLOCK6 */

   _STORE(&q[(nb+1)*ldq],q1);
   _STORE(&q[((nb+1)*ldq)+offset],q2);
   _STORE(&q[((nb+1)*ldq)+2*offset],q3);
   _STORE(&q[((nb+1)*ldq)+3*offset],q4);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-3)], hh[nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-3)]);
#endif

   q1 = _LOAD(&q[(nb+2)*ldq]);
   q2 = _LOAD(&q[((nb+2)*ldq)+offset]);
   q3 = _LOAD(&q[((nb+2)*ldq)+2*offset]);
   q4 = _LOAD(&q[((nb+2)*ldq)+3*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
   q4 = _SIMD_NFMA(x4, h1, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
#endif

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
   q4 = _SIMD_NFMA(z4, h3, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT z4, h3));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */
   _STORE(&q[(nb+2)*ldq],q1);
   _STORE(&q[((nb+2)*ldq)+offset],q2);
   _STORE(&q[((nb+2)*ldq)+2*offset],q3);
   _STORE(&q[((nb+2)*ldq)+3*offset],q4);

#endif /* BLOCK4 || BLOCK6*/


#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-2], hh[nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-2]);
#endif

   q1 = _LOAD(&q[(nb+3)*ldq]);
   q2 = _LOAD(&q[((nb+3)*ldq)+offset]);
   q3 = _LOAD(&q[((nb+3)*ldq)+2*offset]);
   q4 = _LOAD(&q[((nb+3)*ldq)+3*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
   q4 = _SIMD_NFMA(x4, h1, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
   q4 = _SIMD_NFMA(y4, h2, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT y4, h2));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[(nb+3)*ldq],q1);
   _STORE(&q[((nb+3)*ldq)+offset],q2);
   _STORE(&q[((nb+3)*ldq)+2*offset],q3);
   _STORE(&q[((nb+3)*ldq)+3*offset],q4);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-1], hh[nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-1]);
#endif

   q1 = _LOAD(&q[(nb+4)*ldq]);
   q2 = _LOAD(&q[((nb+4)*ldq)+offset]);
   q3 = _LOAD(&q[((nb+4)*ldq)+2*offset]);
   q4 = _LOAD(&q[((nb+4)*ldq)+3*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
   q4 = _SIMD_NFMA(x4, h1, q4);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
   q4 = _SIMD_SUB( ADDITIONAL_ARGUMENT q4, _SIMD_MUL( ADDITIONAL_ARGUMENT x4, h1));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[(nb+4)*ldq],q1);
   _STORE(&q[((nb+4)*ldq)+offset],q2);
   _STORE(&q[((nb+4)*ldq)+2*offset],q3);
   _STORE(&q[((nb+4)*ldq)+3*offset],q4);

#endif /* BLOCK6 */
}

#undef ROW_LENGTH
#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 6
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 12
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 12
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 24
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 24
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 48
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */
/*
 * Unrolled kernel that computes
 * ROW_LENGTH rows of Q simultaneously, a
 * matrix Vector product with two householder
 */
#ifdef BLOCK2
/*
 * vectors + a rank 2 update is performed
 */
#endif
#ifdef BLOCK4
/*
 * vectors + a rank 1 update is performed
 */
#endif

__forceinline void CONCAT_8ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq, int ldh,
#ifdef BLOCK2
               DATA_TYPE s)
#endif
#ifdef BLOCK4
               DATA_TYPE s_1_2, DATA_TYPE s_1_3, DATA_TYPE s_2_3, DATA_TYPE s_1_4, DATA_TYPE s_2_4, DATA_TYPE s_3_4)
#endif
#ifdef BLOCK6
               DATA_TYPE_PTR scalarprods)
#endif
  {
#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Matrix Vector Multiplication, Q [ ROW_LENGTH x nb+1] * hh
    // hh contains two householder vectors, with offset 1
    /////////////////////////////////////////////////////
#endif
#if defined(BLOCK4) || defined(BLOCK6)
    /////////////////////////////////////////////////////
    // Matrix Vector Multiplication, Q [ ROW_LENGTH x nb+3] * hh
    // hh contains four householder vectors
    /////////////////////////////////////////////////////
#endif

    int i;
#ifdef BLOCK2
#if VEC_SET == SSE_128 
    // Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == VSX_SSE
    __SIMD_DATATYPE sign = vec_splats(MONE);
#endif

#if VEC_SET == NEON_ARCH64_128
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = vdupq_n_f64(MONE);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = vdupq_n_f32(MONE);
#endif
#endif

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set1_epi64x(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if  VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_512 */

#if VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = svdup_f64(MONE);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = svdup_f32(MONE);
#endif
#endif

    __SIMD_DATATYPE x1 = _LOAD(&q[ldq]);
    __SIMD_DATATYPE x2 = _LOAD(&q[ldq+offset]);
    __SIMD_DATATYPE x3 = _LOAD(&q[ldq+2*offset]);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h1 = _SIMD_SET1(hh[ldh+1]);
#endif
#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h1 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif
    __SIMD_DATATYPE h2;

    __SIMD_DATATYPE q1 = _LOAD(q);
    __SIMD_DATATYPE q2 = _LOAD(&q[offset]);
    __SIMD_DATATYPE q3 = _LOAD(&q[2*offset]);

#ifdef __ELPA_USE_FMA__
    __SIMD_DATATYPE y1 = _SIMD_FMA(x1, h1, q1);
    __SIMD_DATATYPE y2 = _SIMD_FMA(x2, h1, q2);
    __SIMD_DATATYPE y3 = _SIMD_FMA(x3, h1, q3);
#else
    __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
    __SIMD_DATATYPE y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
    __SIMD_DATATYPE y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
#endif
#endif /* BLOCK2 */

#ifdef BLOCK4
    __SIMD_DATATYPE a1_1 = _LOAD(&q[ldq*3]);
    __SIMD_DATATYPE a2_1 = _LOAD(&q[ldq*2]);
    __SIMD_DATATYPE a3_1 = _LOAD(&q[ldq]);  
    __SIMD_DATATYPE a4_1 = _LOAD(&q[0]);    

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_2_1 = _SIMD_SET1(hh[ldh+1]);    
    __SIMD_DATATYPE h_3_2 = _SIMD_SET1(hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET1(hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_SET1(hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET1(hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_2_1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_2_1 = _SIMD_BROADCAST(&hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w1 = _SIMD_FMA(a3_1, h_4_3, a4_1);
    w1 = _SIMD_FMA(a2_1, h_4_2, w1);
    w1 = _SIMD_FMA(a1_1, h_4_1, w1);
    register __SIMD_DATATYPE z1 = _SIMD_FMA(a2_1, h_3_2, a3_1);
    z1 = _SIMD_FMA(a1_1, h_3_1, z1);
    register __SIMD_DATATYPE y1 = _SIMD_FMA(a1_1, h_2_1, a2_1);
    register __SIMD_DATATYPE x1 = a1_1;
#else
    register __SIMD_DATATYPE w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_4_3));
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_4_2));                          
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_4_1));                          
    register __SIMD_DATATYPE z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_3_2));
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_3_1));                          
    register __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_2_1));
    register __SIMD_DATATYPE x1 = a1_1;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE a1_2 = _LOAD(&q[(ldq*3)+offset]);                  
    __SIMD_DATATYPE a2_2 = _LOAD(&q[(ldq*2)+offset]);
    __SIMD_DATATYPE a3_2 = _LOAD(&q[ldq+offset]);
    __SIMD_DATATYPE a4_2 = _LOAD(&q[0+offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w2 = _SIMD_FMA(a3_2, h_4_3, a4_2);
    w2 = _SIMD_FMA(a2_2, h_4_2, w2);
    w2 = _SIMD_FMA(a1_2, h_4_1, w2);
    register __SIMD_DATATYPE z2 = _SIMD_FMA(a2_2, h_3_2, a3_2);
    z2 = _SIMD_FMA(a1_2, h_3_1, z2);
    register __SIMD_DATATYPE y2 = _SIMD_FMA(a1_2, h_2_1, a2_2);
    register __SIMD_DATATYPE x2 = a1_2;
#else
    register __SIMD_DATATYPE w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_4_3));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_4_2));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_4_1));
    register __SIMD_DATATYPE z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_3_2));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_3_1));
    register __SIMD_DATATYPE y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_2_1));
    register __SIMD_DATATYPE x2 = a1_2;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE a1_3 = _LOAD(&q[(ldq*3)+2*offset]);
    __SIMD_DATATYPE a2_3 = _LOAD(&q[(ldq*2)+2*offset]);
    __SIMD_DATATYPE a3_3 = _LOAD(&q[ldq+2*offset]);
    __SIMD_DATATYPE a4_3 = _LOAD(&q[0+2*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w3 = _SIMD_FMA(a3_3, h_4_3, a4_3);
    w3 = _SIMD_FMA(a2_3, h_4_2, w3);
    w3 = _SIMD_FMA(a1_3, h_4_1, w3);
    register __SIMD_DATATYPE z3 = _SIMD_FMA(a2_3, h_3_2, a3_3);
    z3 = _SIMD_FMA(a1_3, h_3_1, z3);
    register __SIMD_DATATYPE y3 = _SIMD_FMA(a1_3, h_2_1, a2_3);
    register __SIMD_DATATYPE x3 = a1_3;
#else
    register __SIMD_DATATYPE w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_3, h_4_3));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_4_2));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_4_1));
    register __SIMD_DATATYPE z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_3_2));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_3_1));
    register __SIMD_DATATYPE y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_2_1));
    register __SIMD_DATATYPE x3 = a1_3;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE q1;
    __SIMD_DATATYPE q2;
    __SIMD_DATATYPE q3;

    __SIMD_DATATYPE h1;
    __SIMD_DATATYPE h2;
    __SIMD_DATATYPE h3;
    __SIMD_DATATYPE h4;
#endif /* BLOCK4 */

#ifdef BLOCK6
    
    __SIMD_DATATYPE a1_1 = _LOAD(&q[ldq*5]);
    __SIMD_DATATYPE a2_1 = _LOAD(&q[ldq*4]);
    __SIMD_DATATYPE a3_1 = _LOAD(&q[ldq*3]);
    __SIMD_DATATYPE a4_1 = _LOAD(&q[ldq*2]);
    __SIMD_DATATYPE a5_1 = _LOAD(&q[ldq]);  
    __SIMD_DATATYPE a6_1 = _LOAD(&q[0]);    

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_6_5 = _SIMD_SET1(hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_SET1(hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_SET1(hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_SET1(hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_SET1(hh[(ldh*5)+5]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_6_5 = _SIMD_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_6_5 = _SIMD_BROADCAST(&hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_BROADCAST(&hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_BROADCAST(&hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_BROADCAST(&hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_BROADCAST(&hh[(ldh*5)+5]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t1 = _SIMD_FMA(a5_1, h_6_5, a6_1);
    t1 = _SIMD_FMA(a4_1, h_6_4, t1);
    t1 = _SIMD_FMA(a3_1, h_6_3, t1);
    t1 = _SIMD_FMA(a2_1, h_6_2, t1);
    t1 = _SIMD_FMA(a1_1, h_6_1, t1);
#else
    register __SIMD_DATATYPE t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_1, h_6_5)); 
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_1, h_6_4));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_6_3));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_6_2));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_6_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_5_4 = _SIMD_SET1(hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_SET1(hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_SET1(hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_SET1(hh[(ldh*4)+4]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_5_4 = _SIMD_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_5_4 = _SIMD_BROADCAST(&hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_BROADCAST(&hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_BROADCAST(&hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_BROADCAST(&hh[(ldh*4)+4]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE v1 = _SIMD_FMA(a4_1, h_5_4, a5_1);
    v1 = _SIMD_FMA(a3_1, h_5_3, v1);
    v1 = _SIMD_FMA(a2_1, h_5_2, v1);
    v1 = _SIMD_FMA(a1_1, h_5_1, v1);
#else
    register __SIMD_DATATYPE v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_1, h_5_4)); 
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_5_3));
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_5_2));
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_5_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_4_3 = _SIMD_SET1(hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET1(hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_4_3 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_4_3 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w1 = _SIMD_FMA(a3_1, h_4_3, a4_1);
    w1 = _SIMD_FMA(a2_1, h_4_2, w1);
    w1 = _SIMD_FMA(a1_1, h_4_1, w1);
#else
    register __SIMD_DATATYPE w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_4_3)); 
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_4_2));
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_4_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_2_1 = _SIMD_SET1(hh[ldh+1]);    
    __SIMD_DATATYPE h_3_2 = _SIMD_SET1(hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_2_1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_2_1 = _SIMD_BROADCAST(&hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE z1 = _SIMD_FMA(a2_1, h_3_2, a3_1);
    z1 = _SIMD_FMA(a1_1, h_3_1, z1);
    register __SIMD_DATATYPE y1 = _SIMD_FMA(a1_1, h_2_1, a2_1);
#else
    register __SIMD_DATATYPE z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_3_2));
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_3_1));
    register __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_2_1)); 
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x1 = a1_1;

    __SIMD_DATATYPE a1_2 = _LOAD(&q[(ldq*5)+offset]);
    __SIMD_DATATYPE a2_2 = _LOAD(&q[(ldq*4)+offset]);
    __SIMD_DATATYPE a3_2 = _LOAD(&q[(ldq*3)+offset]);
    __SIMD_DATATYPE a4_2 = _LOAD(&q[(ldq*2)+offset]);
    __SIMD_DATATYPE a5_2 = _LOAD(&q[(ldq)+offset]);
    __SIMD_DATATYPE a6_2 = _LOAD(&q[offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t2 = _SIMD_FMA(a5_2, h_6_5, a6_2);
    t2 = _SIMD_FMA(a4_2, h_6_4, t2);
    t2 = _SIMD_FMA(a3_2, h_6_3, t2);
    t2 = _SIMD_FMA(a2_2, h_6_2, t2);
    t2 = _SIMD_FMA(a1_2, h_6_1, t2);
    register __SIMD_DATATYPE v2 = _SIMD_FMA(a4_2, h_5_4, a5_2);
    v2 = _SIMD_FMA(a3_2, h_5_3, v2);
    v2 = _SIMD_FMA(a2_2, h_5_2, v2);
    v2 = _SIMD_FMA(a1_2, h_5_1, v2);
    register __SIMD_DATATYPE w2 = _SIMD_FMA(a3_2, h_4_3, a4_2);
    w2 = _SIMD_FMA(a2_2, h_4_2, w2);
    w2 = _SIMD_FMA(a1_2, h_4_1, w2);
    register __SIMD_DATATYPE z2 = _SIMD_FMA(a2_2, h_3_2, a3_2);
    z2 = _SIMD_FMA(a1_2, h_3_1, z2);
    register __SIMD_DATATYPE y2 = _SIMD_FMA(a1_2, h_2_1, a2_2);
#else
    register __SIMD_DATATYPE t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_2, h_6_5));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_2, h_6_4));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_6_3));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_6_2));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_6_1));
    register __SIMD_DATATYPE v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_2, h_5_4));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_5_3));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_5_2));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_5_1));
    register __SIMD_DATATYPE w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_4_3));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_4_2));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_4_1));
    register __SIMD_DATATYPE z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_3_2));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_3_1));
    register __SIMD_DATATYPE y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_2_1));
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x2 = a1_2;

    __SIMD_DATATYPE a1_3 = _LOAD(&q[(ldq*5)+2*offset]);
    __SIMD_DATATYPE a2_3 = _LOAD(&q[(ldq*4)+2*offset]);
    __SIMD_DATATYPE a3_3 = _LOAD(&q[(ldq*3)+2*offset]);
    __SIMD_DATATYPE a4_3 = _LOAD(&q[(ldq*2)+2*offset]);
    __SIMD_DATATYPE a5_3 = _LOAD(&q[(ldq)+2*offset]);
    __SIMD_DATATYPE a6_3 = _LOAD(&q[2*offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t3 = _SIMD_FMA(a5_3, h_6_5, a6_3);
    t3 = _SIMD_FMA(a4_3, h_6_4, t3);
    t3 = _SIMD_FMA(a3_3, h_6_3, t3);
    t3 = _SIMD_FMA(a2_3, h_6_2, t3);
    t3 = _SIMD_FMA(a1_3, h_6_1, t3);
    register __SIMD_DATATYPE v3 = _SIMD_FMA(a4_3, h_5_4, a5_3);
    v3 = _SIMD_FMA(a3_3, h_5_3, v3);
    v3 = _SIMD_FMA(a2_3, h_5_2, v3);
    v3 = _SIMD_FMA(a1_3, h_5_1, v3);
    register __SIMD_DATATYPE w3 = _SIMD_FMA(a3_3, h_4_3, a4_3);
    w3 = _SIMD_FMA(a2_3, h_4_2, w3);
    w3 = _SIMD_FMA(a1_3, h_4_1, w3);
    register __SIMD_DATATYPE z3 = _SIMD_FMA(a2_3, h_3_2, a3_3);
    z3 = _SIMD_FMA(a1_3, h_3_1, z3);
    register __SIMD_DATATYPE y3 = _SIMD_FMA(a1_3, h_2_1, a2_3);
#else
    register __SIMD_DATATYPE t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_3, h_6_5));
    t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_3, h_6_4));
    t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_3, h_6_3));
    t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_6_2));
    t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_6_1));
    register __SIMD_DATATYPE v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_3, h_5_4));
    v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_3, h_5_3));
    v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_5_2));
    v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_5_1));
    register __SIMD_DATATYPE w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_3, h_4_3));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_4_2));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_4_1));
    register __SIMD_DATATYPE z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_3, h_3_2));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_3_1));
    register __SIMD_DATATYPE y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_3, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_3, h_2_1));
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x3 = a1_3;

    __SIMD_DATATYPE q1;
    __SIMD_DATATYPE q2;
    __SIMD_DATATYPE q3;

    __SIMD_DATATYPE h1;
    __SIMD_DATATYPE h2;
    __SIMD_DATATYPE h3;
    __SIMD_DATATYPE h4;
    __SIMD_DATATYPE h5;
    __SIMD_DATATYPE h6;

#endif /* BLOCK6 */


    for(i = BLOCK; i < nb; i++)
      {
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h1 = _SIMD_SET1(hh[i-(BLOCK-1)]);
        h2 = _SIMD_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == SPARC64_SSE
        h1 = _SIMD_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
        h2 = _SIMD_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif
#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h1 = _SIMD_BROADCAST(&hh[i-(BLOCK-1)]);
        h2 = _SIMD_BROADCAST(&hh[ldh+i-(BLOCK-2)]);
#endif /*   VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

        q1 = _LOAD(&q[i*ldq]);
        q2 = _LOAD(&q[(i*ldq)+offset]);
        q3 = _LOAD(&q[(i*ldq)+2*offset]);

#ifdef __ELPA_USE_FMA__
        x1 = _SIMD_FMA(q1, h1, x1);
        y1 = _SIMD_FMA(q1, h2, y1);
        x2 = _SIMD_FMA(q2, h1, x2);
        y2 = _SIMD_FMA(q2, h2, y2);
        x3 = _SIMD_FMA(q3, h1, x3);
        y3 = _SIMD_FMA(q3, h2, y3);
#else
        x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
        y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
        x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
        y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
        x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
        y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
#endif /* __ELPA_USE_FMA__ */

#if defined(BLOCK4) || defined(BLOCK6)
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h3 = _SIMD_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
        h3 = _SIMD_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h3 = _SIMD_BROADCAST(&hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
        z1 = _SIMD_FMA(q1, h3, z1);
        z2 = _SIMD_FMA(q2, h3, z2);
        z3 = _SIMD_FMA(q3, h3, z3);
#else	
        z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
        z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
        z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h4 = _SIMD_SET1(hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == SPARC64_SSE
        h4 = _SIMD_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h4 = _SIMD_BROADCAST(&hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
        w1 = _SIMD_FMA(q1, h4, w1);
        w2 = _SIMD_FMA(q2, h4, w2);
        w3 = _SIMD_FMA(q3, h4, w3);
#else
        w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4));
        w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h4));
        w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h4));
#endif /* __ELPA_USE_FMA__ */
	
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h5 = _SIMD_SET1(hh[(ldh*4)+i-1]);
#endif
#if VEC_SET == SPARC64_SSE
        h5 = _SIMD_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h5 = _SIMD_BROADCAST(&hh[(ldh*4)+i-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
        v1 = _SIMD_FMA(q1, h5, v1);
        v2 = _SIMD_FMA(q2, h5, v2);
        v3 = _SIMD_FMA(q3, h5, v3);
#else
        v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h5));
        v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h5));
        v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h6 = _SIMD_SET1(hh[(ldh*5)+i-(BLOCK-6)]);
#endif

#if VEC_SET == SPARC64_SSE
        h6 = _SIMD_SET(hh[(ldh*5)+i-(BLOCK-6)], hh[(ldh*5)+i]-(BLOCK-6));
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h6 = _SIMD_BROADCAST(&hh[(ldh*5)+i-(BLOCK-6)]);
#endif

#ifdef __ELPA_USE_FMA__
        t1 = _SIMD_FMA(q1, h6, t1);
        t2 = _SIMD_FMA(q2, h6, t2);
        t3 = _SIMD_FMA(q3, h6, t3);
#else
        t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h6));
        t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h6));
        t3 = _SIMD_ADD( ADDITIONAL_ARGUMENT t3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h6));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */
      }
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-1)]);
#endif

    q1 = _LOAD(&q[nb*ldq]);
    q2 = _LOAD(&q[(nb*ldq)+offset]);
    q3 = _LOAD(&q[(nb*ldq)+2*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
#endif /* __ELPA_USE_FMA__ */

#if defined(BLOCK4) || defined(BLOCK6)
    
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
    z2 = _SIMD_FMA(q2, h3, z2);
    z3 = _SIMD_FMA(q3, h3, z3);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h3));
#endif /* __ELPA_USE_FMA__ */

#ifdef BLOCK4
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-2]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-2], hh[nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-2]);
#endif
    q1 = _LOAD(&q[(nb+1)*ldq]);
    q2 = _LOAD(&q[((nb+1)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+1)*ldq)+2*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[(ldh*1)+nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[(ldh*1)+nb-1], hh[(ldh*1)+nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[(ldh*1)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-1], hh[nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-1]);
#endif

    q1 = _LOAD(&q[(nb+2)*ldq]);
    q2 = _LOAD(&q[((nb+2)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+2)*ldq)+2*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 */
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h4 = _SIMD_SET1(hh[(ldh*3)+nb-(BLOCK-4)]);
#endif

#if VEC_SET == SPARC64_SSE
    h4 = _SIMD_SET(hh[(ldh*3)+nb-(BLOCK-4)], hh[(ldh*3)+nb-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    w1 = _SIMD_FMA(q1, h4, w1);
    w2 = _SIMD_FMA(q2, h4, w2);
    w3 = _SIMD_FMA(q3, h4, w3);
#else
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4)); 
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h4));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h5 = _SIMD_SET1(hh[(ldh*4)+nb-(BLOCK-5)]);
#endif

#if VEC_SET == SPARC64_SSE
    h5 = _SIMD_SET(hh[(ldh*4)+nb-(BLOCK-5)], hh[(ldh*4)+nb-(BLOCK-5)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h5 = _SIMD_BROADCAST(&hh[(ldh*4)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    v1 = _SIMD_FMA(q1, h5, v1);
    v2 = _SIMD_FMA(q2, h5, v2);
    v3 = _SIMD_FMA(q3, h5, v3);
#else
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h5));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h5));
    v3 = _SIMD_ADD( ADDITIONAL_ARGUMENT v3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-(BLOCK-2)]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-(BLOCK-2)], hh[nb-(BLOCK-2)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-2)]);
#endif

    q1 = _LOAD(&q[(nb+1)*ldq]);
    q2 = _LOAD(&q[((nb+1)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+1)*ldq)+2*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-3)]);
#endif
#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-3)], hh[ldh+nb-(BLOCK-3)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
#endif /* __ELPA_USE_FMA__ */
 
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-4)]);
#endif
#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-4)], hh[(ldh*2)+nb-(BLOCK-4)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
    z2 = _SIMD_FMA(q2, h3, z2);
    z3 = _SIMD_FMA(q3, h3, z3);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h4 = _SIMD_SET1(hh[(ldh*3)+nb-(BLOCK-5)]);
#endif
#if VEC_SET == SPARC64_SSE
    h4 = _SIMD_SET(hh[(ldh*3)+nb-(BLOCK-5)], hh[(ldh*3)+nb-(BLOCK-5)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-(BLOCK-5)]);
#endif
 
#ifdef __ELPA_USE_FMA__
    w1 = _SIMD_FMA(q1, h4, w1);
    w2 = _SIMD_FMA(q2, h4, w2);
    w3 = _SIMD_FMA(q3, h4, w3);
#else
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h4));
    w3 = _SIMD_ADD( ADDITIONAL_ARGUMENT w3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-(BLOCK-3)]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-(BLOCK-3)], hh[nb-(BLOCK-3)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-3)]);
#endif

    q1 = _LOAD(&q[(nb+2)*ldq]);
    q2 = _LOAD(&q[((nb+2)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+2)*ldq)+2*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-4)]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-4)], hh[ldh+nb-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-5)]);
#endif

#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-5)], hh[(ldh*2)+nb-(BLOCK-5)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
    z2 = _SIMD_FMA(q2, h3, z2);
    z3 = _SIMD_FMA(q3, h3, z3);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
    z3 = _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-(BLOCK-4)]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-(BLOCK-4)], hh[nb-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-4)]);
#endif

    q1 = _LOAD(&q[(nb+3)*ldq]);
    q2 = _LOAD(&q[((nb+3)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+3)*ldq)+2*offset]);


#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-5)]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-5)], hh[ldh+nb-(BLOCK-5)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
    y3 = _SIMD_FMA(q3, h2, y3);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
    y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT y3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-(BLOCK-5)]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-(BLOCK-5)], hh[nb-(BLOCK-5)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-5)]);
#endif
 
    q1 = _LOAD(&q[(nb+4)*ldq]);
    q2 = _LOAD(&q[((nb+4)*ldq)+offset]);
    q3 = _LOAD(&q[((nb+4)*ldq)+2*offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
    x3 = _SIMD_FMA(q3, h1, x3);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
    x3 = _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT q3,h1));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Rank-2 update of Q [ ROW_LENGTH x nb+1]
    /////////////////////////////////////////////////////
#endif
#ifdef BLOCK4
    /////////////////////////////////////////////////////
    // Rank-1 update of Q [ ROW_LENGTH x nb+3]
    /////////////////////////////////////////////////////
#endif
#ifdef BLOCK6
    /////////////////////////////////////////////////////
    // Apply tau, correct wrong calculation using pre-calculated scalar products
    /////////////////////////////////////////////////////
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE tau1 = _SIMD_SET1(hh[0]);
    __SIMD_DATATYPE tau2 = _SIMD_SET1(hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_SET1(hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_SET1(hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_SET1(hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_SET1(hh[ldh*5]);       
#endif

#ifdef BLOCK2    
    __SIMD_DATATYPE vs = _SIMD_SET1(s);
#endif
#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET1(s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET1(s_1_3);  
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET1(s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET1(s_1_4);  
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET1(s_2_4);  
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET1(s_3_4);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET1(scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET1(scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET1(scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET1(scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET1(scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET1(scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_SET1(scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_SET1(scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_SET1(scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_SET1(scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_SET1(scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_SET1(scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_SET1(scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_SET1(scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_SET1(scalarprods[14]);
#endif
#endif /* VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE */

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE tau1 = _SIMD_SET(hh[0], hh[0]);
    __SIMD_DATATYPE tau2 = _SIMD_SET(hh[ldh], hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_SET(hh[ldh*2], hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_SET(hh[ldh*3], hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_SET(hh[ldh*4], hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_SET(hh[ldh*5], hh[ldh*5]);
#endif

#ifdef BLOCK2
    __SIMD_DATATYPE vs = _SIMD_SET(s, s);
#endif
#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET(s_1_2, s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET(s_1_3, s_1_3);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET(s_2_3, s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET(s_1_4, s_1_4);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET(s_2_4, s_2_4);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET(s_3_4, s_3_4);

#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET(scalarprods[0], scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET(scalarprods[1], scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET(scalarprods[2], scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET(scalarprods[3], scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET(scalarprods[4], scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET(scalarprods[5], scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_SET(scalarprods[6], scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_SET(scalarprods[7], scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_SET(scalarprods[8], scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_SET(scalarprods[9], scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_SET(scalarprods[10], scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_SET(scalarprods[11], scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_SET(scalarprods[12], scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_SET(scalarprods[13], scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_SET(scalarprods[14], scalarprods[14]);
#endif
#endif /* VEC_SET == SPARC64_SSE */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   __SIMD_DATATYPE tau1 = _SIMD_BROADCAST(hh);
   __SIMD_DATATYPE tau2 = _SIMD_BROADCAST(&hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_BROADCAST(&hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_BROADCAST(&hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_BROADCAST(&hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_BROADCAST(&hh[ldh*5]);
#endif

#ifdef BLOCK2  
   __SIMD_DATATYPE vs = _SIMD_BROADCAST(&s);
#endif

#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_BROADCAST(&s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_BROADCAST(&s_1_3);
   __SIMD_DATATYPE vs_2_3 = _SIMD_BROADCAST(&s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_BROADCAST(&s_1_4);
   __SIMD_DATATYPE vs_2_4 = _SIMD_BROADCAST(&s_2_4);
   __SIMD_DATATYPE vs_3_4 = _SIMD_BROADCAST(&s_3_4);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_BROADCAST(&scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_BROADCAST(&scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_BROADCAST(&scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_BROADCAST(&scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_BROADCAST(&scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_BROADCAST(&scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_BROADCAST(&scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_BROADCAST(&scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_BROADCAST(&scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_BROADCAST(&scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_BROADCAST(&scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_BROADCAST(&scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_BROADCAST(&scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_BROADCAST(&scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_BROADCAST(&scalarprods[14]);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == VSX_SSE || VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _XOR(tau1, sign);
#endif

#if VEC_SET == SPARC64_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
    h1 = _SIMD_NEG( ADDITIONAL_ARGUMENT tau1);
#endif

#if VEC_SET == AVX_512
#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi64((__SIMD_INTEGER) tau1, (__SIMD_INTEGER) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi32((__SIMD_INTEGER) tau1, (__SIMD_INTEGER) sign);
#endif
#endif /* HAVE_AVX512_XEON_PHI */

#ifdef HAVE_AVX512_XEON
    h1 = _XOR(tau1, sign);
#endif
#endif /* VEC_SET == AVX_512 */

#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau1;
#endif

   x1 = _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1);
   x2 = _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1);
   x3 = _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1);

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == VSX_SSE || VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _XOR(tau2, sign);
#endif

#if VEC_SET == SPARC64_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
   h1 = _SIMD_NEG( ADDITIONAL_ARGUMENT tau2);
#endif

#if VEC_SET == AVX_512
#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi64((__SIMD_INTEGER) tau2, (__SIMD_INTEGER) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi32((__SIMD_INTEGER) tau2, (__SIMD_INTEGER) sign);
#endif
#endif /* HAVE_AVX512_XEON_PHI */

#ifdef HAVE_AVX512_XEON
    h1 = _XOR(tau2, sign);
#endif
#endif /* VEC_SET == AVX_512 */
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs);

#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau2;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_2);
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK2

#ifdef __ELPA_USE_FMA__
   y1 = _SIMD_FMA(y1, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_FMA(y2, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
   y3 = _SIMD_FMA(y3, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2));
#else
   y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
   y3 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2));
#endif
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)

#ifdef __ELPA_USE_FMA__
   y1 = _SIMD_FMSUB(y1, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_FMSUB(y2, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
   y3 = _SIMD_FMSUB(y3, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2));
#else   
   y1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
   y3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2));
#endif /* __ELPA_USE_FMA__ */

   h1 = tau3;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_3);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_2_3);

#ifdef __ELPA_USE_FMA__
   z1 = _SIMD_FMSUB(z1, h1, _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)));
   z2 = _SIMD_FMSUB(z2, h1, _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)));
   z3 = _SIMD_FMSUB(z3, h1, _SIMD_FMA(y3, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2)));
#else
   z1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)));
   z2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)));
   z3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2)));
#endif /* __ELPA_USE_FMA__ */

   h1 = tau4;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_4);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_2_4);
   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_3_4);

#ifdef __ELPA_USE_FMA__
   w1 = _SIMD_FMSUB(w1, h1, _SIMD_FMA(z1, h4, _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   w2 = _SIMD_FMSUB(w2, h1, _SIMD_FMA(z2, h4, _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
   w3 = _SIMD_FMSUB(w3, h1, _SIMD_FMA(z3, h4, _SIMD_FMA(y3, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2))));
#else
   w1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)))); 
   w2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
   w3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w3,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2))));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_1_5); 
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_2_5);
   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_3_5);
   h5 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_4_5);

#ifdef __ELPA_USE_FMA__
   v1 = _SIMD_FMSUB(v1, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w1, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   v2 = _SIMD_FMSUB(v2, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w2, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
   v3 = _SIMD_FMSUB(v3, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w3, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4)), _SIMD_FMA(y3, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2))));
#else
   v1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v1,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   v2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v2,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
   v3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v3,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w3,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2))));
#endif /* __ELPA_USE_FMA__ */

   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_1_6);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_2_6);
   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_3_6);
   h5 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_4_6);
   h6 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_5_6);

#ifdef __ELPA_USE_FMA__
   t1 = _SIMD_FMSUB(t1, tau6, _SIMD_FMA(v1, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w1, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)))));
   t2 = _SIMD_FMSUB(t2, tau6, _SIMD_FMA(v2, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w2, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)))));
   t3 = _SIMD_FMSUB(t3, tau6, _SIMD_FMA(v3, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w3, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4)), _SIMD_FMA(y3, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2)))));
#else
   t1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t1,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v1,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)))));
   t2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t2,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v2,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)))));
   t3 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t3,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v3,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w3,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h2)))));
#endif /* __ELPA_USE_FMA__ */

   /////////////////////////////////////////////////////
   // Rank-1 update of Q [ROW_LENGTH x nb+3]
   /////////////////////////////////////////////////////
#endif /* BLOCK6 */

   q1 = _LOAD(&q[0]);
#ifdef BLOCK2
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, y1);
#endif
#ifdef BLOCK4
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, w1);
#endif
#ifdef BLOCK6
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, t1); 
#endif
   _STORE(&q[0],q1);
   q2 = _LOAD(&q[offset]);
#ifdef BLOCK2
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, y2);
#endif
#ifdef BLOCK4
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, w2);
#endif
#ifdef BLOCK6
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, t2);
#endif
   _STORE(&q[offset],q2);
   q3 = _LOAD(&q[2*offset]);
#ifdef BLOCK2
   q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, y3);
#endif
#ifdef BLOCK4
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, w3);
#endif
#ifdef BLOCK6
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, t3);
#endif

   _STORE(&q[2*offset],q3);

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif

   q1 = _LOAD(&q[ldq]);
   q2 = _LOAD(&q[ldq+offset]);
   q3 = _LOAD(&q[ldq+2*offset]);
#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_FMA(y1, h2, x1));
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_FMA(y2, h2, x2));
   q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_FMA(y3, h2, x3));
#else
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2)));
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2)));
   q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_ADD( ADDITIONAL_ARGUMENT x3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2)));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq],q1);
   _STORE(&q[ldq+offset],q2);
   _STORE(&q[ldq+2*offset],q3);

#endif /* BLOCK2 */

#ifdef BLOCK4
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
#endif
   q1 = _LOAD(&q[ldq]);
   q2 = _LOAD(&q[ldq+offset]);
   q3 = _LOAD(&q[ldq+2*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_FMA(w1, h4, z1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_FMA(w2, h4, z2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_FMA(w3, h4, z3));
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4)));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4)));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_ADD( ADDITIONAL_ARGUMENT z3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4)));
#endif /* __ELPA_USE_FMA__ */
 
   _STORE(&q[ldq],q1);
   _STORE(&q[ldq+offset],q2);
   _STORE(&q[ldq+2*offset],q3);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+2]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
#endif

   q1 = _LOAD(&q[ldq*2]);
   q2 = _LOAD(&q[(ldq*2)+offset]);
   q3 = _LOAD(&q[(ldq*2)+2*offset]);

   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, y1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, y2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, y3);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*2],q1);
   _STORE(&q[(ldq*2)+offset],q2);
   _STORE(&q[(ldq*2)+2*offset],q3);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif
   q1 = _LOAD(&q[ldq*3]);
   q2 = _LOAD(&q[(ldq*3)+offset]);
   q3 = _LOAD(&q[(ldq*3)+2*offset]);

   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, x1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, x2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, x3);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*3], q1);
   _STORE(&q[(ldq*3)+offset], q2);
   _STORE(&q[(ldq*3)+2*offset], q3);

#endif /* BLOCK4 */

#ifdef BLOCK6
   
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+1]);
#endif

   q1 = _LOAD(&q[ldq]);
   q2 = _LOAD(&q[(ldq+offset)]);
   q3 = _LOAD(&q[(ldq+2*offset)]);

   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, v1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, v2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, v3);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
#endif /* __ELPA_USE_FMA__ */
 
   _STORE(&q[ldq],q1);
   _STORE(&q[(ldq+offset)],q2);
   _STORE(&q[(ldq+2*offset)],q3);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+1]);
#endif

   q1 = _LOAD(&q[ldq*2]);
   q2 = _LOAD(&q[(ldq*2)+offset]);
   q3 = _LOAD(&q[(ldq*2)+2*offset]);

   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, w1); 
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, w2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, w3);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5)); 
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*2],q1);
   _STORE(&q[(ldq*2)+offset],q2);
   _STORE(&q[(ldq*2)+2*offset],q3);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
#endif
 
   q1 = _LOAD(&q[ldq*3]);
   q2 = _LOAD(&q[(ldq*3)+offset]);
   q3 = _LOAD(&q[(ldq*3)+2*offset]);

   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, z1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, z2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, z3);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*3],q1);
   _STORE(&q[(ldq*3)+offset],q2);
   _STORE(&q[(ldq*3)+2*offset],q3);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
#endif

   q1 = _LOAD(&q[ldq*4]);
   q2 = _LOAD(&q[(ldq*4)+offset]);
   q3 = _LOAD(&q[(ldq*4)+2*offset]);

   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, y1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, y2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, y3);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+4]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+4]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*4],q1);
   _STORE(&q[(ldq*4)+offset],q2);
   _STORE(&q[(ldq*4)+2*offset],q3);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[(ldh)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[(ldh)+1], hh[(ldh)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[(ldh)+1]);
#endif
   q1 = _LOAD(&q[ldq*5]);
   q2 = _LOAD(&q[(ldq*5)+offset]);
   q3 = _LOAD(&q[(ldq*5)+2*offset]);

   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, x1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, x2);
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, x3);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+4]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+4]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+5]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+5]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
   q3 = _SIMD_NFMA(t3, h6, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*5],q1);
   _STORE(&q[(ldq*5)+offset],q2);
   _STORE(&q[(ldq*5)+2*offset],q3);

#endif /* BLOCK6 */

   for (i = BLOCK; i < nb; i++)
   {
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h1 = _SIMD_SET1(hh[i-(BLOCK-1)]);
     h2 = _SIMD_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == SPARC64_SSE
     h1 = _SIMD_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
     h2 = _SIMD_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[i-(BLOCK-1)]);
    h2 = _SIMD_BROADCAST(&hh[ldh+i-(BLOCK-2)]);
#endif

     q1 = _LOAD(&q[i*ldq]);
     q2 = _LOAD(&q[(i*ldq)+offset]);
     q3 = _LOAD(&q[(i*ldq)+2*offset]);

#ifdef BLOCK2
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_FMA(x1, h1, q1);
     q1 = _SIMD_FMA(y1, h2, q1);
     q2 = _SIMD_FMA(x2, h1, q2);
     q2 = _SIMD_FMA(y2, h2, q2);
     q3 = _SIMD_FMA(x3, h1, q3);
     q3 = _SIMD_FMA(y3, h2, q3);
#else
     q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2)));
     q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2)));
     q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2)));
#endif /* __ELPA_USE_FMA__ */
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(x1, h1, q1);
     q2 = _SIMD_NFMA(x2, h1, q2);
     q3 = _SIMD_NFMA(x3, h1, q3);
#else     
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h1));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h1));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3,h1));
#endif /* __ELPA_USE_FMA__ */

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(y1, h2, q1);
     q2 = _SIMD_NFMA(y2, h2, q2);
     q3 = _SIMD_NFMA(y3, h2, q3);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h2));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h2));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h3 = _SIMD_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
     h3 = _SIMD_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h3 = _SIMD_BROADCAST(&hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(z1, h3, q1);
     q2 = _SIMD_NFMA(z2, h3, q2);
     q3 = _SIMD_NFMA(z3, h3, q3);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h3));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h3));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h4 = _SIMD_SET1(hh[(ldh*3)+i-(BLOCK-4)]); 
#endif

#if VEC_SET == SPARC64_SSE
     h4 = _SIMD_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h4 = _SIMD_BROADCAST(&hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(w1, h4, q1);
     q2 = _SIMD_NFMA(w2, h4, q2);
     q3 = _SIMD_NFMA(w3, h4, q3);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h4));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h4));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3,h4));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h5 = _SIMD_SET1(hh[(ldh*4)+i-1]);
#endif
#if VEC_SET == SPARC64_SSE
     h5 = _SIMD_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h5 = _SIMD_BROADCAST(&hh[(ldh*4)+i-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(v1, h5, q1);
     q2 = _SIMD_NFMA(v2, h5, q2);
     q3 = _SIMD_NFMA(v3, h5, q3);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h6 = _SIMD_SET1(hh[(ldh*5)+i]);
#endif
#if VEC_SET == SPARC64_SSE
     h6 = _SIMD_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h6 = _SIMD_BROADCAST(&hh[(ldh*5)+i-(BLOCK-6)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(t1, h6, q1);
     q2 = _SIMD_NFMA(t2, h6, q2);
     q3 = _SIMD_NFMA(t3, h6, q3);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
     q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT t3, h6));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */
     _STORE(&q[i*ldq],q1);
     _STORE(&q[(i*ldq)+offset],q2);
     _STORE(&q[(i*ldq)+2*offset],q3);

   }
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-1)]);
#endif

   q1 = _LOAD(&q[nb*ldq]);
   q2 = _LOAD(&q[(nb*ldq)+offset]);
   q3 = _LOAD(&q[(nb*ldq)+2*offset]);

#ifdef BLOCK2

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_FMA(x1, h1, q1);
   q2 = _SIMD_FMA(x2, h1, q2);
   q3 = _SIMD_FMA(x3, h1, q3);
#else
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_ADD( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
#endif
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
#else   
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
#endif

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-2]);
#endif


#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
   q3 = _SIMD_NFMA(v3, h5, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT v3, h5));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

   _STORE(&q[nb*ldq],q1);
   _STORE(&q[(nb*ldq)+offset],q2);
   _STORE(&q[(nb*ldq)+2*offset],q3);

#if defined(BLOCK4) || defined(BLOCK6)
   
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-2)], hh[nb-(BLOCK-2)]);
#endif
   
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-2)]);
#endif

   q1 = _LOAD(&q[(nb+1)*ldq]);
   q2 = _LOAD(&q[((nb+1)*ldq)+offset]);
   q3 = _LOAD(&q[((nb+1)*ldq)+2*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-3)], hh[ldh+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
#endif

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
#else 
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
   q3 = _SIMD_NFMA(w3, h4, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT w3, h4));
#endif /* __ELPA_USE_FMA__ */
#endif /* BLOCK6 */

   _STORE(&q[(nb+1)*ldq],q1);
   _STORE(&q[((nb+1)*ldq)+offset],q2);
   _STORE(&q[((nb+1)*ldq)+2*offset],q3);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-3)], hh[nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-3)]);
#endif

   q1 = _LOAD(&q[(nb+2)*ldq]);
   q2 = _LOAD(&q[((nb+2)*ldq)+offset]);
   q3 = _LOAD(&q[((nb+2)*ldq)+2*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
#endif

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
   q3 = _SIMD_NFMA(z3, h3, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT z3, h3));
#endif /* __ELPA_USE_FMA__ */
#endif /* BLOCK6 */

   _STORE(&q[(nb+2)*ldq],q1);
   _STORE(&q[((nb+2)*ldq)+offset],q2);
   _STORE(&q[((nb+2)*ldq)+2*offset],q3);

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-2], hh[nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-2]);
#endif

   q1 = _LOAD(&q[(nb+3)*ldq]);
   q2 = _LOAD(&q[((nb+3)*ldq)+offset]);
   q3 = _LOAD(&q[((nb+3)*ldq)+2*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
   q3 = _SIMD_NFMA(y3, h2, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT y3, h2));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[(nb+3)*ldq],q1);
   _STORE(&q[((nb+3)*ldq)+offset],q2);
   _STORE(&q[((nb+3)*ldq)+2*offset],q3);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-1], hh[nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-1]);
#endif

   q1 = _LOAD(&q[(nb+4)*ldq]);
   q2 = _LOAD(&q[((nb+4)*ldq)+offset]);
   q3 = _LOAD(&q[((nb+4)*ldq)+2*offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
   q3 = _SIMD_NFMA(x3, h1, q3);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
   q3 = _SIMD_SUB( ADDITIONAL_ARGUMENT q3, _SIMD_MUL( ADDITIONAL_ARGUMENT x3, h1));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[(nb+4)*ldq],q1);
   _STORE(&q[((nb+4)*ldq)+offset],q2);
   _STORE(&q[((nb+4)*ldq)+2*offset],q3);

#endif /* BLOCK6 */
}


#undef ROW_LENGTH
#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 8
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 16
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 16
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 32
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */
/*
 * Unrolled kernel that computes
 * ROW_LENGTH rows of Q simultaneously, a
 * matrix Vector product with two householder
 */
#ifdef BLOCK2
/*
 * vectors + a rank 2 update is performed
 */
#endif
#if defined(BLOCK4) || defined(BLOCK6)
/*
 * vectors + a rank 1 update is performed
 */
#endif

__forceinline void CONCAT_8ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq, int ldh,
#ifdef BLOCK2
               DATA_TYPE s)
#endif
#ifdef BLOCK4
               DATA_TYPE s_1_2, DATA_TYPE s_1_3, DATA_TYPE s_2_3, DATA_TYPE s_1_4, DATA_TYPE s_2_4, DATA_TYPE s_3_4)
#endif 
#ifdef BLOCK6
               DATA_TYPE_PTR scalarprods)
#endif 
  {
#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Matrix Vector Multiplication, Q [ ROW_LENGTH x nb+1] * hh
    // hh contains two householder vectors, with offset 1
    /////////////////////////////////////////////////////
#endif
#if defined(BLOCK4) || defined(BLOCK6)
    /////////////////////////////////////////////////////
    // Matrix Vector Multiplication, Q [ ROW_LENGTH x nb+3] * hh
    // hh contains four householder vectors
    /////////////////////////////////////////////////////
#endif

    int i;
#ifdef BLOCK2
#if VEC_SET == SSE_128
    // Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == VSX_SSE
    __SIMD_DATATYPE sign = vec_splats(MONE);
#endif

#if VEC_SET == NEON_ARCH64_128
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = vdupq_n_f64(MONE);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = vdupq_n_f32(MONE);
#endif
#endif

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set1_epi64x(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if  VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_512 */

#if VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = svdup_f64(MONE);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = svdup_f32(MONE);
#endif
#endif

    __SIMD_DATATYPE x1 = _LOAD(&q[ldq]);
    __SIMD_DATATYPE x2 = _LOAD(&q[ldq+offset]);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h1 = _SIMD_SET1(hh[ldh+1]);
#endif
#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h1 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif

    __SIMD_DATATYPE h2;
#ifdef __ELPA_USE_FMA__
    __SIMD_DATATYPE q1 = _LOAD(q);
    __SIMD_DATATYPE y1 = _SIMD_FMA(x1, h1, q1);
    __SIMD_DATATYPE q2 = _LOAD(&q[offset]);
    __SIMD_DATATYPE y2 = _SIMD_FMA(x2, h1, q2);
#else
    __SIMD_DATATYPE q1 = _LOAD(q);
    __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
    __SIMD_DATATYPE q2 = _LOAD(&q[offset]);
    __SIMD_DATATYPE y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
#endif
#endif /* BLOCK2 */

#ifdef BLOCK4
    __SIMD_DATATYPE a1_1 = _LOAD(&q[ldq*3]);
    __SIMD_DATATYPE a2_1 = _LOAD(&q[ldq*2]);
    __SIMD_DATATYPE a3_1 = _LOAD(&q[ldq]);  
    __SIMD_DATATYPE a4_1 = _LOAD(&q[0]);    

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_2_1 = _SIMD_SET1(hh[ldh+1]);    
    __SIMD_DATATYPE h_3_2 = _SIMD_SET1(hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET1(hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_SET1(hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET1(hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_2_1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_2_1 = _SIMD_BROADCAST(&hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w1 = _SIMD_FMA(a3_1, h_4_3, a4_1);
    w1 = _SIMD_FMA(a2_1, h_4_2, w1);
    w1 = _SIMD_FMA(a1_1, h_4_1, w1);
    register __SIMD_DATATYPE z1 = _SIMD_FMA(a2_1, h_3_2, a3_1);
    z1 = _SIMD_FMA(a1_1, h_3_1, z1);
    register __SIMD_DATATYPE y1 = _SIMD_FMA(a1_1, h_2_1, a2_1);
    register __SIMD_DATATYPE x1 = a1_1;
#else
    register __SIMD_DATATYPE w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_4_3));
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_4_2));                          
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_4_1));                          
    register __SIMD_DATATYPE z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_3_2));
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_3_1));                          
    register __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_2_1));
    register __SIMD_DATATYPE x1 = a1_1;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE a1_2 = _LOAD(&q[(ldq*3)+offset]);                  
    __SIMD_DATATYPE a2_2 = _LOAD(&q[(ldq*2)+offset]);
    __SIMD_DATATYPE a3_2 = _LOAD(&q[ldq+offset]);
    __SIMD_DATATYPE a4_2 = _LOAD(&q[0+offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w2 = _SIMD_FMA(a3_2, h_4_3, a4_2);
    w2 = _SIMD_FMA(a2_2, h_4_2, w2);
    w2 = _SIMD_FMA(a1_2, h_4_1, w2);
    register __SIMD_DATATYPE z2 = _SIMD_FMA(a2_2, h_3_2, a3_2);
    z2 = _SIMD_FMA(a1_2, h_3_1, z2);
    register __SIMD_DATATYPE y2 = _SIMD_FMA(a1_2, h_2_1, a2_2);
    register __SIMD_DATATYPE x2 = a1_2;
#else
    register __SIMD_DATATYPE w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_4_3));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_4_2));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_4_1));
    register __SIMD_DATATYPE z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_3_2));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_3_1));
    register __SIMD_DATATYPE y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_2_1));
    register __SIMD_DATATYPE x2 = a1_2;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE q1;
    __SIMD_DATATYPE q2;

    __SIMD_DATATYPE h1;
    __SIMD_DATATYPE h2;
    __SIMD_DATATYPE h3;
    __SIMD_DATATYPE h4;
#endif /* BLOCK4 */

#ifdef BLOCK6
    
    __SIMD_DATATYPE a1_1 = _LOAD(&q[ldq*5]);
    __SIMD_DATATYPE a2_1 = _LOAD(&q[ldq*4]);
    __SIMD_DATATYPE a3_1 = _LOAD(&q[ldq*3]);
    __SIMD_DATATYPE a4_1 = _LOAD(&q[ldq*2]);
    __SIMD_DATATYPE a5_1 = _LOAD(&q[ldq]);  
    __SIMD_DATATYPE a6_1 = _LOAD(&q[0]);    

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_6_5 = _SIMD_SET1(hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_SET1(hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_SET1(hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_SET1(hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_SET1(hh[(ldh*5)+5]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_6_5 = _SIMD_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_6_5 = _SIMD_BROADCAST(&hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_BROADCAST(&hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_BROADCAST(&hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_BROADCAST(&hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_BROADCAST(&hh[(ldh*5)+5]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t1 = _SIMD_FMA(a5_1, h_6_5, a6_1);
    t1 = _SIMD_FMA(a4_1, h_6_4, t1);
    t1 = _SIMD_FMA(a3_1, h_6_3, t1);
    t1 = _SIMD_FMA(a2_1, h_6_2, t1);
    t1 = _SIMD_FMA(a1_1, h_6_1, t1);
#else
    register __SIMD_DATATYPE t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_1, h_6_5)); 
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_1, h_6_4));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_6_3));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_6_2));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_6_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_5_4 = _SIMD_SET1(hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_SET1(hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_SET1(hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_SET1(hh[(ldh*4)+4]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_5_4 = _SIMD_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_5_4 = _SIMD_BROADCAST(&hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_BROADCAST(&hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_BROADCAST(&hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_BROADCAST(&hh[(ldh*4)+4]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE v1 = _SIMD_FMA(a4_1, h_5_4, a5_1);
    v1 = _SIMD_FMA(a3_1, h_5_3, v1);
    v1 = _SIMD_FMA(a2_1, h_5_2, v1);
    v1 = _SIMD_FMA(a1_1, h_5_1, v1);
#else
    register __SIMD_DATATYPE v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_1, h_5_4)); 
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_5_3));
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_5_2));
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_5_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_4_3 = _SIMD_SET1(hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET1(hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_4_3 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_4_3 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w1 = _SIMD_FMA(a3_1, h_4_3, a4_1);
    w1 = _SIMD_FMA(a2_1, h_4_2, w1);
    w1 = _SIMD_FMA(a1_1, h_4_1, w1);
#else
    register __SIMD_DATATYPE w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_4_3)); 
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_4_2));
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_4_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_2_1 = _SIMD_SET1(hh[ldh+1]);    
    __SIMD_DATATYPE h_3_2 = _SIMD_SET1(hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_2_1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_2_1 = _SIMD_BROADCAST(&hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE z1 = _SIMD_FMA(a2_1, h_3_2, a3_1);
    z1 = _SIMD_FMA(a1_1, h_3_1, z1);
    register __SIMD_DATATYPE y1 = _SIMD_FMA(a1_1, h_2_1, a2_1);
#else
    register __SIMD_DATATYPE z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_3_2));
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_3_1));
    register __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_2_1)); 
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x1 = a1_1;

    __SIMD_DATATYPE a1_2 = _LOAD(&q[(ldq*5)+offset]);
    __SIMD_DATATYPE a2_2 = _LOAD(&q[(ldq*4)+offset]);
    __SIMD_DATATYPE a3_2 = _LOAD(&q[(ldq*3)+offset]);
    __SIMD_DATATYPE a4_2 = _LOAD(&q[(ldq*2)+offset]);
    __SIMD_DATATYPE a5_2 = _LOAD(&q[(ldq)+offset]);
    __SIMD_DATATYPE a6_2 = _LOAD(&q[offset]);

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t2 = _SIMD_FMA(a5_2, h_6_5, a6_2);
    t2 = _SIMD_FMA(a4_2, h_6_4, t2);
    t2 = _SIMD_FMA(a3_2, h_6_3, t2);
    t2 = _SIMD_FMA(a2_2, h_6_2, t2);
    t2 = _SIMD_FMA(a1_2, h_6_1, t2);
    register __SIMD_DATATYPE v2 = _SIMD_FMA(a4_2, h_5_4, a5_2);
    v2 = _SIMD_FMA(a3_2, h_5_3, v2);
    v2 = _SIMD_FMA(a2_2, h_5_2, v2);
    v2 = _SIMD_FMA(a1_2, h_5_1, v2);
    register __SIMD_DATATYPE w2 = _SIMD_FMA(a3_2, h_4_3, a4_2);
    w2 = _SIMD_FMA(a2_2, h_4_2, w2);
    w2 = _SIMD_FMA(a1_2, h_4_1, w2);
    register __SIMD_DATATYPE z2 = _SIMD_FMA(a2_2, h_3_2, a3_2);
    z2 = _SIMD_FMA(a1_2, h_3_1, z2);
    register __SIMD_DATATYPE y2 = _SIMD_FMA(a1_2, h_2_1, a2_2);
#else
    register __SIMD_DATATYPE t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_2, h_6_5));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_2, h_6_4));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_6_3));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_6_2));
    t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_6_1));
    register __SIMD_DATATYPE v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_2, h_5_4));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_5_3));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_5_2));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_5_1));
    register __SIMD_DATATYPE w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_2, h_4_3));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_4_2));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_4_1));
    register __SIMD_DATATYPE z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_2, h_3_2));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_3_1));
    register __SIMD_DATATYPE y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_2, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_2, h_2_1));
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x2 = a1_2;

    __SIMD_DATATYPE q1;
    __SIMD_DATATYPE q2;

    __SIMD_DATATYPE h1;
    __SIMD_DATATYPE h2;
    __SIMD_DATATYPE h3;
    __SIMD_DATATYPE h4;
    __SIMD_DATATYPE h5;
    __SIMD_DATATYPE h6;

#endif /* BLOCK6 */

    for(i = BLOCK; i < nb; i++)
      {
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h1 = _SIMD_SET1(hh[i-(BLOCK-1)]);
        h2 = _SIMD_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == SPARC64_SSE
        h1 = _SIMD_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
        h2 = _SIMD_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif
#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h1 = _SIMD_BROADCAST(&hh[i-(BLOCK-1)]);
        h2 = _SIMD_BROADCAST(&hh[ldh+i-(BLOCK-2)]);
#endif /*   VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

        q1 = _LOAD(&q[i*ldq]);
        q2 = _LOAD(&q[(i*ldq)+offset]);
#ifdef __ELPA_USE_FMA__
        x1 = _SIMD_FMA(q1, h1, x1);
        y1 = _SIMD_FMA(q1, h2, y1);
        x2 = _SIMD_FMA(q2, h1, x2);
        y2 = _SIMD_FMA(q2, h2, y2);
#else
        x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
        y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
        x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
        y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
#endif /* __ELPA_USE_FMA__ */

#if defined(BLOCK4) || defined(BLOCK6)
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h3 = _SIMD_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
        h3 = _SIMD_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h3 = _SIMD_BROADCAST(&hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
        z1 = _SIMD_FMA(q1, h3, z1);
        z2 = _SIMD_FMA(q2, h3, z2);
#else	
        z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
        z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h4 = _SIMD_SET1(hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == SPARC64_SSE
        h4 = _SIMD_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h4 = _SIMD_BROADCAST(&hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
        w1 = _SIMD_FMA(q1, h4, w1);
        w2 = _SIMD_FMA(q2, h4, w2);
#else
        w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4));
        w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h4));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h5 = _SIMD_SET1(hh[(ldh*4)+i-1]);
#endif
#if VEC_SET == SPARC64_SSE
        h5 = _SIMD_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h5 = _SIMD_BROADCAST(&hh[(ldh*4)+i-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
        v1 = _SIMD_FMA(q1, h5, v1);
        v2 = _SIMD_FMA(q2, h5, v2);
#else
        v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h5));
        v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h6 = _SIMD_SET1(hh[(ldh*5)+i-(BLOCK-6)]);
#endif

#if VEC_SET == SPARC64_SSE
        h6 = _SIMD_SET(hh[(ldh*5)+i], hh[(ldh*5)+i-(BLOCK-6)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h6 = _SIMD_BROADCAST(&hh[(ldh*5)+i-(BLOCK-6)]);
#endif

#ifdef __ELPA_USE_FMA__
        t1 = _SIMD_FMA(q1, h6, t1);
        t2 = _SIMD_FMA(q2, h6, t2);
#else
        t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h6));
        t2 = _SIMD_ADD( ADDITIONAL_ARGUMENT t2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h6));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */
      }
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-1)]);
#endif

    q1 = _LOAD(&q[nb*ldq]);
    q2 = _LOAD(&q[(nb*ldq)+offset]);
#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
#endif /* __ELPA_USE_FMA__ */

#if defined(BLOCK4) || defined(BLOCK6)
    
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
    z2 = _SIMD_FMA(q2, h3, z2);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
#endif /* __ELPA_USE_FMA__ */

#ifdef BLOCK4

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-2]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-2], hh[nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-2]);
#endif

    q1 = _LOAD(&q[(nb+1)*ldq]);
    q2 = _LOAD(&q[((nb+1)*ldq)+offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[(ldh*1)+nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[(ldh*1)+nb-1], hh[(ldh*1)+nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[(ldh*1)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-1], hh[nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-1]);
#endif

    q1 = _LOAD(&q[(nb+2)*ldq]);
    q2 = _LOAD(&q[((nb+2)*ldq)+offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 */
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h4 = _SIMD_SET1(hh[(ldh*3)+nb-(BLOCK-4)]);
#endif

#if VEC_SET == SPARC64_SSE
    h4 = _SIMD_SET(hh[(ldh*3)+nb-(BLOCK-4)], hh[(ldh*3)+nb-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    w1 = _SIMD_FMA(q1, h4, w1);
    w2 = _SIMD_FMA(q2, h4, w2);
#else
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4)); 
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h5 = _SIMD_SET1(hh[(ldh*4)+nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h5 = _SIMD_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h5 = _SIMD_BROADCAST(&hh[(ldh*4)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    v1 = _SIMD_FMA(q1, h5, v1);
    v2 = _SIMD_FMA(q2, h5, v2);
#else
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h5));
    v2 = _SIMD_ADD( ADDITIONAL_ARGUMENT v2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-(BLOCK-2)]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-(BLOCK-2)], hh[nb-(BLOCK-2)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-2)]);
#endif

    q1 = _LOAD(&q[(nb+1)*ldq]);
    q2 = _LOAD(&q[((nb+1)*ldq)+offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-3)]);
#endif
#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-3)], hh[ldh+nb-(BLOCK-3)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-4)]);
#endif
#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-4)], hh[(ldh*2)+nb-(BLOCK-4)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
    z2 = _SIMD_FMA(q2, h3, z2);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h4 = _SIMD_SET1(hh[(ldh*3)+nb-(BLOCK-5)]);
#endif
#if VEC_SET == SPARC64_SSE
    h4 = _SIMD_SET(hh[(ldh*3)+nb-(BLOCK-5)], hh[(ldh*3)+nb-(BLOCK-5)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    w1 = _SIMD_FMA(q1, h4, w1);
    w2 = _SIMD_FMA(q2, h4, w2);
#else
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4));
    w2 = _SIMD_ADD( ADDITIONAL_ARGUMENT w2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-(BLOCK-3)]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-(BLOCK-3)], hh[nb-(BLOCK-3)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-3)]);
#endif

    q1 = _LOAD(&q[(nb+2)*ldq]);
    q2 = _LOAD(&q[((nb+2)*ldq)+offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-4)]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-4)], hh[ldh+nb-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-5)]);
#endif

#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-5)], hh[(ldh*2)+nb-(BLOCK-5)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
    z2 = _SIMD_FMA(q2, h3, z2);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
    z2 = _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-2]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-2], hh[nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-4)]);
#endif

    q1 = _LOAD(&q[(nb+3)*ldq]);
    q2 = _LOAD(&q[((nb+3)*ldq)+offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
    y2 = _SIMD_FMA(q2, h2, y2);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
    y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT y2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-1], hh[nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-5)]);
#endif

    q1 = _LOAD(&q[(nb+4)*ldq]);
    q2 = _LOAD(&q[((nb+4)*ldq)+offset]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
    x2 = _SIMD_FMA(q2, h1, x2);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
    x2 = _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT q2,h1));
#endif /* __ELPA_USE_FMA__ */
#endif /* BLOCK6 */

#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Rank-2 update of Q [ ROW_LENGTH x nb+1]
    /////////////////////////////////////////////////////
#endif
#ifdef BLOCK4
    /////////////////////////////////////////////////////
    // Rank-1 update of Q [ ROW_LENGTH x nb+3]
    /////////////////////////////////////////////////////
#endif
#ifdef BLOCK6
    /////////////////////////////////////////////////////
    // Apply tau, correct wrong calculation using pre-calculated scalar products
    /////////////////////////////////////////////////////
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE tau1 = _SIMD_SET1(hh[0]);
    __SIMD_DATATYPE tau2 = _SIMD_SET1(hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_SET1(hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_SET1(hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_SET1(hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_SET1(hh[ldh*5]);       
#endif

#ifdef BLOCK2    
    __SIMD_DATATYPE vs = _SIMD_SET1(s);
#endif
#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET1(s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET1(s_1_3);  
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET1(s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET1(s_1_4);  
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET1(s_2_4);  
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET1(s_3_4);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET1(scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET1(scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET1(scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET1(scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET1(scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET1(scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_SET1(scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_SET1(scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_SET1(scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_SET1(scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_SET1(scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_SET1(scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_SET1(scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_SET1(scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_SET1(scalarprods[14]);
#endif
#endif /* VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE */

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE tau1 = _SIMD_SET(hh[0], hh[0]);
    __SIMD_DATATYPE tau2 = _SIMD_SET(hh[ldh], hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_SET(hh[ldh*2], hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_SET(hh[ldh*3], hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_SET(hh[ldh*4], hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_SET(hh[ldh*5], hh[ldh*5]);
#endif

#ifdef BLOCK2
    __SIMD_DATATYPE vs = _SIMD_SET(s, s);
#endif
#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET(s_1_2, s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET(s_1_3, s_1_3);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET(s_2_3, s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET(s_1_4, s_1_4);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET(s_2_4, s_2_4);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET(s_3_4, s_3_4);

#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET(scalarprods[0], scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET(scalarprods[1], scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET(scalarprods[2], scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET(scalarprods[3], scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET(scalarprods[4], scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET(scalarprods[5], scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_SET(scalarprods[6], scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_SET(scalarprods[7], scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_SET(scalarprods[8], scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_SET(scalarprods[9], scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_SET(scalarprods[10], scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_SET(scalarprods[11], scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_SET(scalarprods[12], scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_SET(scalarprods[13], scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_SET(scalarprods[14], scalarprods[14]);
#endif
#endif /*  VEC_SET == SPARC64_SSE */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   __SIMD_DATATYPE tau1 = _SIMD_BROADCAST(hh);
   __SIMD_DATATYPE tau2 = _SIMD_BROADCAST(&hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_BROADCAST(&hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_BROADCAST(&hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_BROADCAST(&hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_BROADCAST(&hh[ldh*5]);
#endif

#ifdef BLOCK2  
   __SIMD_DATATYPE vs = _SIMD_BROADCAST(&s);
#endif

#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_BROADCAST(&s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_BROADCAST(&s_1_3);
   __SIMD_DATATYPE vs_2_3 = _SIMD_BROADCAST(&s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_BROADCAST(&s_1_4);
   __SIMD_DATATYPE vs_2_4 = _SIMD_BROADCAST(&s_2_4);
   __SIMD_DATATYPE vs_3_4 = _SIMD_BROADCAST(&s_3_4);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_BROADCAST(&scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_BROADCAST(&scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_BROADCAST(&scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_BROADCAST(&scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_BROADCAST(&scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_BROADCAST(&scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_BROADCAST(&scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_BROADCAST(&scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_BROADCAST(&scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_BROADCAST(&scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_BROADCAST(&scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_BROADCAST(&scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_BROADCAST(&scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_BROADCAST(&scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_BROADCAST(&scalarprods[14]);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == VSX_SSE || VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _XOR(tau1, sign);
#endif

#if VEC_SET == SPARC64_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
    h1 = _SIMD_NEG( ADDITIONAL_ARGUMENT tau1);
#endif

#if VEC_SET == AVX_512
#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi64((__SIMD_INTEGER) tau1, (__SIMD_INTEGER) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi32((__SIMD_INTEGER) tau1, (__SIMD_INTEGER) sign);
#endif
#endif /* HAVE_AVX512_XEON_PHI */

#ifdef HAVE_AVX512_XEON
    h1 = _XOR(tau1, sign);
#endif
#endif /* VEC_SET == AVX_512 */

#endif  /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau1;
#endif

   x1 = _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1);
   x2 = _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1);

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == VSX_SSE || VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _XOR(tau2, sign);
#endif

#if VEC_SET == SPARC64_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
   h1 = _SIMD_NEG( ADDITIONAL_ARGUMENT tau2);
#endif

#if VEC_SET == AVX_512
#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi64((__SIMD_INTEGER) tau2, (__SIMD_INTEGER) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi32((__SIMD_INTEGER) tau2, (__SIMD_INTEGER) sign);
#endif
#endif /* HAVE_AVX512_XEON_PHI */

#ifdef HAVE_AVX512_XEON
    h1 = _XOR(tau2, sign);
#endif
#endif /* VEC_SET == AVX_512 */
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs);
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau2;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_2); 
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK2

#ifdef __ELPA_USE_FMA__
   y1 = _SIMD_FMA(y1, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_FMA(y2, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
#else
   y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
#endif
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)

#ifdef __ELPA_USE_FMA__
   y1 = _SIMD_FMSUB(y1, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_FMSUB(y2, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
#else   
   y1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
   y2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2));
#endif /* __ELPA_USE_FMA__ */

   h1 = tau3;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_3);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_2_3);

#ifdef __ELPA_USE_FMA__
   z1 = _SIMD_FMSUB(z1, h1, _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)));
   z2 = _SIMD_FMSUB(z2, h1, _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)));
#else
   z1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)));
   z2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)));
#endif /* __ELPA_USE_FMA__ */

   h1 = tau4;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_4);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_2_4);
   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_3_4);

#ifdef __ELPA_USE_FMA__
   w1 = _SIMD_FMSUB(w1, h1, _SIMD_FMA(z1, h4, _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   w2 = _SIMD_FMSUB(w2, h1, _SIMD_FMA(z2, h4, _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
#else
   w1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)))); 
   w2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_1_5); 
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_2_5);
   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_3_5);
   h5 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_4_5);

#ifdef __ELPA_USE_FMA__
   v1 = _SIMD_FMSUB(v1, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w1, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   v2 = _SIMD_FMSUB(v2, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w2, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
#else
   v1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v1,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
   v2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v2,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2))));
#endif /* __ELPA_USE_FMA__ */

   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_1_6);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_2_6);
   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_3_6);
   h5 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_4_6);
   h6 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_5_6);

#ifdef __ELPA_USE_FMA__
   t1 = _SIMD_FMSUB(t1, tau6, _SIMD_FMA(v1, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w1, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)))));
   t2 = _SIMD_FMSUB(t2, tau6, _SIMD_FMA(v2, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w2, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_FMA(y2, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)))));
#else
   t1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t1,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v1,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)))));
   t2 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t2,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v2,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h2)))));
#endif /* __ELPA_USE_FMA__ */

   /////////////////////////////////////////////////////
   // Rank-1 update of Q [ROW_LENGTH x nb+3]
   /////////////////////////////////////////////////////
#endif /* BLOCK6 */

   q1 = _LOAD(&q[0]);
#ifdef BLOCK2
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, y1);
#endif
#ifdef BLOCK4
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, w1);
#endif
#ifdef BLOCK6
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, t1); 
#endif
   _STORE(&q[0],q1);
   q2 = _LOAD(&q[offset]);
#ifdef BLOCK2
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, y2);
#endif
#ifdef BLOCK4
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, w2);
#endif
#ifdef BLOCK6
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, t2);
#endif
   _STORE(&q[offset],q2);

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _LOAD(&q[ldq]);
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_FMA(y1, h2, x1));
   _STORE(&q[ldq],q1);
   q2 = _LOAD(&q[ldq+offset]);
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_FMA(y2, h2, x2));
   _STORE(&q[ldq+offset],q2);
#else
   q1 = _LOAD(&q[ldq]);
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2)));
   _STORE(&q[ldq],q1);
   q2 = _LOAD(&q[ldq+offset]);
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_ADD( ADDITIONAL_ARGUMENT x2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2)));
   _STORE(&q[ldq+offset],q2);
#endif /* __ELPA_USE_FMA__ */
#endif /* BLOCK2 */

#ifdef BLOCK4
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
#endif

   q1 = _LOAD(&q[ldq]);
   q2 = _LOAD(&q[ldq+offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_FMA(w1, h4, z1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_FMA(w2, h4, z2));
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4)));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_ADD( ADDITIONAL_ARGUMENT z2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4)));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq],q1);
   _STORE(&q[ldq+offset],q2);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+2]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
#endif

   q1 = _LOAD(&q[ldq*2]);
   q2 = _LOAD(&q[(ldq*2)+offset]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, y1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, y2);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*2],q1);
   _STORE(&q[(ldq*2)+offset],q2);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

   q1 = _LOAD(&q[ldq*3]);
   q2 = _LOAD(&q[(ldq*3)+offset]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, x1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, x2);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*3], q1);
   _STORE(&q[(ldq*3)+offset], q2);

#endif /* BLOCK4 */

#ifdef BLOCK6
   
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+1]);
#endif

   q1 = _LOAD(&q[ldq]);
   q2 = _LOAD(&q[(ldq+offset)]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, v1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, v2);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq],q1);
   _STORE(&q[(ldq+offset)],q2);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+1]);
#endif
   q1 = _LOAD(&q[ldq*2]);
   q2 = _LOAD(&q[(ldq*2)+offset]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, w1); 
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, w2);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5)); 
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*2],q1);
   _STORE(&q[(ldq*2)+offset],q2);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
#endif

   q1 = _LOAD(&q[ldq*3]);
   q2 = _LOAD(&q[(ldq*3)+offset]);

   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, z1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, z2);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
#endif

   _STORE(&q[ldq*3],q1);
   _STORE(&q[(ldq*3)+offset],q2);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
#endif
   q1 = _LOAD(&q[ldq*4]);
   q2 = _LOAD(&q[(ldq*4)+offset]);

   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, y1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, y2);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+4]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+4]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
#endif

   _STORE(&q[ldq*4],q1);
   _STORE(&q[(ldq*4)+offset],q2);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[(ldh)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[(ldh)+1], hh[(ldh)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[(ldh)+1]);
#endif
   q1 = _LOAD(&q[ldq*5]);
   q2 = _LOAD(&q[(ldq*5)+offset]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, x1);
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, x2);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+4]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+4]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+5]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+5]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
   q2 = _SIMD_NFMA(t2, h6, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
#endif
   _STORE(&q[ldq*5],q1);
   _STORE(&q[(ldq*5)+offset],q2);

#endif /* BLOCK6 */

   for (i = BLOCK; i < nb; i++)
   {
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h1 = _SIMD_SET1(hh[i-(BLOCK-1)]);
     h2 = _SIMD_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == SPARC64_SSE
     h1 = _SIMD_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
     h2 = _SIMD_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[i-(BLOCK-1)]);
    h2 = _SIMD_BROADCAST(&hh[ldh+i-(BLOCK-2)]);
#endif

     q1 = _LOAD(&q[i*ldq]);
     q2 = _LOAD(&q[(i*ldq)+offset]);

#ifdef BLOCK2
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_FMA(x1, h1, q1);
     q1 = _SIMD_FMA(y1, h2, q1);
     q2 = _SIMD_FMA(x2, h1, q2);
     q2 = _SIMD_FMA(y2, h2, q2);
#else
     q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2)));
     q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2)));
#endif /* __ELPA_USE_FMA__ */
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(x1, h1, q1);
     q2 = _SIMD_NFMA(x2, h1, q2);
#else  
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h1));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2,h1));
#endif /* __ELPA_USE_FMA__ */

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(y1, h2, q1);
     q2 = _SIMD_NFMA(y2, h2, q2);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h2));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h3 = _SIMD_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
     h3 = _SIMD_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h3 = _SIMD_BROADCAST(&hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(z1, h3, q1);
     q2 = _SIMD_NFMA(z2, h3, q2);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h3));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h4 = _SIMD_SET1(hh[(ldh*3)+i-(BLOCK-4)]); 
#endif

#if VEC_SET == SPARC64_SSE
     h4 = _SIMD_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h4 = _SIMD_BROADCAST(&hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(w1, h4, q1);
     q2 = _SIMD_NFMA(w2, h4, q2);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h4));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2,h4));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 || BLOCK6*/

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h5 = _SIMD_SET1(hh[(ldh*4)+i-1]);
#endif
#if VEC_SET == SPARC64_SSE
     h5 = _SIMD_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h5 = _SIMD_BROADCAST(&hh[(ldh*4)+i-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(v1, h5, q1);
     q2 = _SIMD_NFMA(v2, h5, q2);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h6 = _SIMD_SET1(hh[(ldh*5)+i]);
#endif
#if VEC_SET == SPARC64_SSE
     h6 = _SIMD_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h6 = _SIMD_BROADCAST(&hh[(ldh*5)+i-(BLOCK-6)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(t1, h6, q1);
     q2 = _SIMD_NFMA(t2, h6, q2);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
     q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT t2, h6));
#endif /* __ELPA_USE_FMA__ */
#endif /* BLOCK6 */

     _STORE(&q[i*ldq],q1);
     _STORE(&q[(i*ldq)+offset],q2);

   }
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-1)]);
#endif

   q1 = _LOAD(&q[nb*ldq]);
   q2 = _LOAD(&q[(nb*ldq)+offset]);

#ifdef BLOCK2

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_FMA(x1, h1, q1);
   q2 = _SIMD_FMA(x2, h1, q2);
#else
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_ADD( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
#endif
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
#else   
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
#endif

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
   q2 = _SIMD_NFMA(v2, h5, q2);
#else 
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT v2, h5));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

   _STORE(&q[nb*ldq],q1);
   _STORE(&q[(nb*ldq)+offset],q2);

#if defined(BLOCK4) || defined(BLOCK6)
   
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-2)], hh[nb-(BLOCK-2)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-2)]);
#endif

   q1 = _LOAD(&q[(nb+1)*ldq]);
   q2 = _LOAD(&q[((nb+1)*ldq)+offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-3)], hh[ldh+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
#endif

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
   q2 = _SIMD_NFMA(w2, h4, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT w2, h4));
#endif /* __ELPA_USE_FMA__ */
#endif /* BLOCK6 */

   _STORE(&q[(nb+1)*ldq],q1);
   _STORE(&q[((nb+1)*ldq)+offset],q2);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-3)], hh[nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-3)]);
#endif

   q1 = _LOAD(&q[(nb+2)*ldq]);
   q2 = _LOAD(&q[((nb+2)*ldq)+offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
#endif

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
   q2 = _SIMD_NFMA(z2, h3, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT z2, h3));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

   _STORE(&q[(nb+2)*ldq],q1);
   _STORE(&q[((nb+2)*ldq)+offset],q2);

#endif /* BLOCK4 || BLOCK6*/

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-2], hh[nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-2]);
#endif

   q1 = _LOAD(&q[(nb+3)*ldq]);
   q2 = _LOAD(&q[((nb+3)*ldq)+offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
   q2 = _SIMD_NFMA(y2, h2, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT y2, h2));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[(nb+3)*ldq],q1);
   _STORE(&q[((nb+3)*ldq)+offset],q2);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-1], hh[nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-1]);
#endif

   q1 = _LOAD(&q[(nb+4)*ldq]);
   q2 = _LOAD(&q[((nb+4)*ldq)+offset]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
   q2 = _SIMD_NFMA(x2, h1, q2);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
   q2 = _SIMD_SUB( ADDITIONAL_ARGUMENT q2, _SIMD_MUL( ADDITIONAL_ARGUMENT x2, h1));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[(nb+4)*ldq],q1);
   _STORE(&q[((nb+4)*ldq)+offset],q2);

#endif /* BLOCK6 */

}

#undef ROW_LENGTH
#if  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 2
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 4
#endif
#endif /*  VEC_SET == SSE_128 || VEC_SET == SPARC64_SSE || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_128 */

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 8
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 || VEC_SET == SVE_256 */

#if  VEC_SET == AVX_512 || VEC_SET == SVE_512
#ifdef DOUBLE_PRECISION_REAL
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_REAL
#define ROW_LENGTH 16
#endif
#endif /* VEC_SET == AVX_512 || VEC_SET == SVE_512 */


/*
 * Unrolled kernel that computes
 * ROW_LENGTH rows of Q simultaneously, a
 * matrix Vector product with two householder
 */
#ifdef BLOCK2
/*
 * vectors + a rank 2 update is performed
 */
#endif
#ifdef BLOCK4
/*
 * vectors + a rank 1 update is performed
 */
#endif

__forceinline void CONCAT_8ARGS(hh_trafo_kernel_,ROW_LENGTH,_,SIMD_SET,_,BLOCK,hv_,WORD_LENGTH) (DATA_TYPE_PTR q, DATA_TYPE_PTR hh, int nb, int ldq, int ldh,
#ifdef BLOCK2
               DATA_TYPE s)
#endif
#ifdef BLOCK4
               DATA_TYPE s_1_2, DATA_TYPE s_1_3, DATA_TYPE s_2_3, DATA_TYPE s_1_4, DATA_TYPE s_2_4, DATA_TYPE s_3_4)
#endif
#ifdef BLOCK6
               DATA_TYPE_PTR scalarprods)
#endif 
  {
#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Matrix Vector Multiplication, Q [ ROW_LENGTH x nb+1] * hh
    // hh contains two householder vectors, with offset 1
    /////////////////////////////////////////////////////
#endif
#if defined(BLOCK4) || defined(BLOCK6)
    /////////////////////////////////////////////////////
    // Matrix Vector Multiplication, Q [ ROW_LENGTH x nb+3] * hh
    // hh contains four householder vectors
    /////////////////////////////////////////////////////
#endif
    int i;
#ifdef BLOCK2
#if VEC_SET == SSE_128
    // Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif /* VEC_SET == SSE_128 */

#if VEC_SET == VSX_SSE
    __SIMD_DATATYPE sign = vec_splats(MONE);
#endif

#if VEC_SET == NEON_ARCH64_128
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = vdupq_n_f64(MONE);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = vdupq_n_f32(MONE);
#endif
#endif

#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256
#ifdef DOUBLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set1_epi64x(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm256_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#if  VEC_SET == AVX_512
#ifdef DOUBLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SIMD_DATATYPE sign = (__SIMD_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif
#endif /* VEC_SET == AVX_512 */

#if VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
#ifdef DOUBLE_PRECISION_REAL
    __SIMD_DATATYPE sign = svdup_f64(MONE);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SIMD_DATATYPE sign = svdup_f32(MONE);
#endif
#endif

    __SIMD_DATATYPE x1 = _LOAD(&q[ldq]);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h1 = _SIMD_SET1(hh[ldh+1]);
#endif
#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h1 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif
 
    __SIMD_DATATYPE h2;
#ifdef __ELPA_USE_FMA__
    __SIMD_DATATYPE q1 = _LOAD(q);
    __SIMD_DATATYPE y1 = _SIMD_FMA(x1, h1, q1);
#else
    __SIMD_DATATYPE q1 = _LOAD(q);
    __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
#endif
#endif /* BLOCK2 */

#ifdef BLOCK4
    __SIMD_DATATYPE a1_1 = _LOAD(&q[ldq*3]);
    __SIMD_DATATYPE a2_1 = _LOAD(&q[ldq*2]);
    __SIMD_DATATYPE a3_1 = _LOAD(&q[ldq]);  
    __SIMD_DATATYPE a4_1 = _LOAD(&q[0]);    

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_2_1 = _SIMD_SET1(hh[ldh+1]);    
    __SIMD_DATATYPE h_3_2 = _SIMD_SET1(hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET1(hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_SET1(hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET1(hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_2_1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_2_1 = _SIMD_BROADCAST(&hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
    __SIMD_DATATYPE h_4_3 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w1 = _SIMD_FMA(a3_1, h_4_3, a4_1);
    w1 = _SIMD_FMA(a2_1, h_4_2, w1);
    w1 = _SIMD_FMA(a1_1, h_4_1, w1);
    register __SIMD_DATATYPE z1 = _SIMD_FMA(a2_1, h_3_2, a3_1);
    z1 = _SIMD_FMA(a1_1, h_3_1, z1);
    register __SIMD_DATATYPE y1 = _SIMD_FMA(a1_1, h_2_1, a2_1);
    register __SIMD_DATATYPE x1 = a1_1;
#else
    register __SIMD_DATATYPE w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_4_3));
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_4_2));                          
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_4_1));                          
    register __SIMD_DATATYPE z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_3_2));
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_3_1));                          
    register __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_2_1));
    register __SIMD_DATATYPE x1 = a1_1;
#endif /* __ELPA_USE_FMA__ */

    __SIMD_DATATYPE q1;

    __SIMD_DATATYPE h1;
    __SIMD_DATATYPE h2;
    __SIMD_DATATYPE h3;
    __SIMD_DATATYPE h4;
#endif /* BLOCK4 */

#ifdef BLOCK6
    
    __SIMD_DATATYPE a1_1 = _LOAD(&q[ldq*5]);
    __SIMD_DATATYPE a2_1 = _LOAD(&q[ldq*4]);
    __SIMD_DATATYPE a3_1 = _LOAD(&q[ldq*3]);
    __SIMD_DATATYPE a4_1 = _LOAD(&q[ldq*2]);
    __SIMD_DATATYPE a5_1 = _LOAD(&q[ldq]);  
    __SIMD_DATATYPE a6_1 = _LOAD(&q[0]);    

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_6_5 = _SIMD_SET1(hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_SET1(hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_SET1(hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_SET1(hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_SET1(hh[(ldh*5)+5]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_6_5 = _SIMD_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_6_5 = _SIMD_BROADCAST(&hh[(ldh*5)+1]);
    __SIMD_DATATYPE h_6_4 = _SIMD_BROADCAST(&hh[(ldh*5)+2]);
    __SIMD_DATATYPE h_6_3 = _SIMD_BROADCAST(&hh[(ldh*5)+3]);
    __SIMD_DATATYPE h_6_2 = _SIMD_BROADCAST(&hh[(ldh*5)+4]);
    __SIMD_DATATYPE h_6_1 = _SIMD_BROADCAST(&hh[(ldh*5)+5]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE t1 = _SIMD_FMA(a5_1, h_6_5, a6_1);
    t1 = _SIMD_FMA(a4_1, h_6_4, t1);
    t1 = _SIMD_FMA(a3_1, h_6_3, t1);
    t1 = _SIMD_FMA(a2_1, h_6_2, t1);
    t1 = _SIMD_FMA(a1_1, h_6_1, t1);
#else
    register __SIMD_DATATYPE t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a6_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a5_1, h_6_5)); 
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_1, h_6_4));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_6_3));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_6_2));
    t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_6_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_5_4 = _SIMD_SET1(hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_SET1(hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_SET1(hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_SET1(hh[(ldh*4)+4]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_5_4 = _SIMD_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_5_4 = _SIMD_BROADCAST(&hh[(ldh*4)+1]);
    __SIMD_DATATYPE h_5_3 = _SIMD_BROADCAST(&hh[(ldh*4)+2]);
    __SIMD_DATATYPE h_5_2 = _SIMD_BROADCAST(&hh[(ldh*4)+3]);
    __SIMD_DATATYPE h_5_1 = _SIMD_BROADCAST(&hh[(ldh*4)+4]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE v1 = _SIMD_FMA(a4_1, h_5_4, a5_1);
    v1 = _SIMD_FMA(a3_1, h_5_3, v1);
    v1 = _SIMD_FMA(a2_1, h_5_2, v1);
    v1 = _SIMD_FMA(a1_1, h_5_1, v1);
#else
    register __SIMD_DATATYPE v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a5_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a4_1, h_5_4)); 
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_5_3));
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_5_2));
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_5_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_4_3 = _SIMD_SET1(hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET1(hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_4_3 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_4_3 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
    __SIMD_DATATYPE h_4_2 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
    __SIMD_DATATYPE h_4_1 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE w1 = _SIMD_FMA(a3_1, h_4_3, a4_1);
    w1 = _SIMD_FMA(a2_1, h_4_2, w1);
    w1 = _SIMD_FMA(a1_1, h_4_1, w1);
#else
    register __SIMD_DATATYPE w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a4_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a3_1, h_4_3)); 
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_4_2));
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_4_1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE h_2_1 = _SIMD_SET1(hh[ldh+1]);    
    __SIMD_DATATYPE h_3_2 = _SIMD_SET1(hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE h_2_1 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    __SIMD_DATATYPE h_2_1 = _SIMD_BROADCAST(&hh[ldh+1]);
    __SIMD_DATATYPE h_3_2 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
    __SIMD_DATATYPE h_3_1 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
    register __SIMD_DATATYPE z1 = _SIMD_FMA(a2_1, h_3_2, a3_1);
    z1 = _SIMD_FMA(a1_1, h_3_1, z1);
    register __SIMD_DATATYPE y1 = _SIMD_FMA(a1_1, h_2_1, a2_1);
#else
    register __SIMD_DATATYPE z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a3_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a2_1, h_3_2));
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_3_1));
    register __SIMD_DATATYPE y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT a2_1, _SIMD_MUL( ADDITIONAL_ARGUMENT a1_1, h_2_1)); 
#endif /* __ELPA_USE_FMA__ */

    register __SIMD_DATATYPE x1 = a1_1;

    __SIMD_DATATYPE q1;

    __SIMD_DATATYPE h1;
    __SIMD_DATATYPE h2;
    __SIMD_DATATYPE h3;
    __SIMD_DATATYPE h4;
    __SIMD_DATATYPE h5;
    __SIMD_DATATYPE h6;

#endif /* BLOCK6 */

    for(i = BLOCK; i < nb; i++)
      {
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h1 = _SIMD_SET1(hh[i-(BLOCK-1)]);
        h2 = _SIMD_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == SPARC64_SSE
        h1 = _SIMD_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
        h2 = _SIMD_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif
#if  VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h1 = _SIMD_BROADCAST(&hh[i-(BLOCK-1)]);
        h2 = _SIMD_BROADCAST(&hh[ldh+i-(BLOCK-2)]);
#endif /*   VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#ifdef __ELPA_USE_FMA__
        q1 = _LOAD(&q[i*ldq]);
        x1 = _SIMD_FMA(q1, h1, x1);
        y1 = _SIMD_FMA(q1, h2, y1);
#else
        q1 = _LOAD(&q[i*ldq]);
        x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
        y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
#endif /* __ELPA_USE_FMA__ */

#if defined(BLOCK4) || defined(BLOCK6)
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h3 = _SIMD_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
        h3 = _SIMD_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h3 = _SIMD_BROADCAST(&hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
        z1 = _SIMD_FMA(q1, h3, z1);
#else	
        z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h4 = _SIMD_SET1(hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == SPARC64_SSE
        h4 = _SIMD_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h4 = _SIMD_BROADCAST(&hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
        w1 = _SIMD_FMA(q1, h4, w1);
#else
        w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4));
#endif
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h5 = _SIMD_SET1(hh[(ldh*4)+i-(BLOCK-5)]);
#endif
#if VEC_SET == SPARC64_SSE
        h5 = _SIMD_SET(hh[(ldh*4)+i-(BLOCK-5)], hh[(ldh*4)+i-(BLOCK-5)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h5 = _SIMD_BROADCAST(&hh[(ldh*4)+i-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
        v1 = _SIMD_FMA(q1, h5, v1);
#else
        v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
        h6 = _SIMD_SET1(hh[(ldh*5)+i]);
#endif

#if VEC_SET == SPARC64_SSE
        h6 = _SIMD_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
        h6 = _SIMD_BROADCAST(&hh[(ldh*5)+i-(BLOCK-6)]);
#endif

#ifdef __ELPA_USE_FMA__
        t1 = _SIMD_FMA(q1, h6, t1);
#else
        t1 = _SIMD_ADD( ADDITIONAL_ARGUMENT t1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h6));
#endif /* __ELPA_USE_FMA__ */	

#endif /* BLOCK6 */
      }
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-1)]);
#endif

#ifdef __ELPA_USE_FMA__
    q1 = _LOAD(&q[nb*ldq]);
    x1 = _SIMD_FMA(q1, h1, x1);
#else
    q1 = _LOAD(&q[nb*ldq]);
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
#endif /* __ELPA_USE_FMA__ */

#if defined(BLOCK4) || defined(BLOCK6)
    
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
#endif

#ifdef BLOCK4

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-2]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-2], hh[nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-2]);
#endif

    q1 = _LOAD(&q[(nb+1)*ldq]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[(ldh*1)+nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[(ldh*1)+nb-1], hh[(ldh*1)+nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[(ldh*1)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-1], hh[nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-1]);
#endif

    q1 = _LOAD(&q[(nb+2)*ldq]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
#endif /* __ELPA_USE_FMA__ */
#endif /* BLOCK4 */
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h4 = _SIMD_SET1(hh[(ldh*3)+nb-2]);
#endif

#if VEC_SET == SPARC64_SSE
    h4 = _SIMD_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    w1 = _SIMD_FMA(q1, h4, w1);
#else
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4)); 
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h5 = _SIMD_SET1(hh[(ldh*4)+nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h5 = _SIMD_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h5 = _SIMD_BROADCAST(&hh[(ldh*4)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    v1 = _SIMD_FMA(q1, h5, v1);
#else
    v1 = _SIMD_ADD( ADDITIONAL_ARGUMENT v1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-4]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-4], hh[nb-4]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-2)]);
#endif

    q1 = _LOAD(&q[(nb+1)*ldq]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-3]);
#endif
#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-3], hh[ldh+nb-3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h4 = _SIMD_SET1(hh[(ldh*3)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
    h4 = _SIMD_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    w1 = _SIMD_FMA(q1, h4, w1);
#else
    w1 = _SIMD_ADD( ADDITIONAL_ARGUMENT w1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-3]);
#endif
#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-3], hh[nb-3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-3]);
#endif

    q1 = _LOAD(&q[(nb+2)*ldq]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-2]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h3 = _SIMD_SET1(hh[(ldh*2)+nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h3 = _SIMD_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    z1 = _SIMD_FMA(q1, h3, z1);
#else
    z1 = _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-2]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-2], hh[nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-4)]);
#endif

    q1 = _LOAD(&q[(nb+3)*ldq]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h2 = _SIMD_SET1(hh[ldh+nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h2 = _SIMD_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
    y1 = _SIMD_FMA(q1, h2, y1);
#else
    y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT y1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    h1 = _SIMD_SET1(hh[nb-1]);
#endif

#if VEC_SET == SPARC64_SSE
    h1 = _SIMD_SET(hh[nb-1], hh[nb-1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-5)]);
#endif

    q1 = _LOAD(&q[(nb+4)*ldq]);

#ifdef __ELPA_USE_FMA__
    x1 = _SIMD_FMA(q1, h1, x1);
#else
    x1 = _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT q1,h1));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Rank-2 update of Q [ ROW_LENGTH x nb+1]
    /////////////////////////////////////////////////////
#endif
#ifdef BLOCK4
    /////////////////////////////////////////////////////
    // Rank-1 update of Q [ ROW_LENGTH x nb+3]
    /////////////////////////////////////////////////////
#endif
#ifdef BLOCK6
    /////////////////////////////////////////////////////
    // Apply tau, correct wrong calculation using pre-calculated scalar products
    /////////////////////////////////////////////////////
#endif


#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
    __SIMD_DATATYPE tau1 = _SIMD_SET1(hh[0]);
    __SIMD_DATATYPE tau2 = _SIMD_SET1(hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_SET1(hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_SET1(hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_SET1(hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_SET1(hh[ldh*5]);       
#endif

#ifdef BLOCK2    
    __SIMD_DATATYPE vs = _SIMD_SET1(s);
#endif
#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET1(s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET1(s_1_3);  
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET1(s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET1(s_1_4);  
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET1(s_2_4);  
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET1(s_3_4);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET1(scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET1(scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET1(scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET1(scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET1(scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET1(scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_SET1(scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_SET1(scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_SET1(scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_SET1(scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_SET1(scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_SET1(scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_SET1(scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_SET1(scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_SET1(scalarprods[14]);
#endif
#endif /* VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE */

#if VEC_SET == SPARC64_SSE
    __SIMD_DATATYPE tau1 = _SIMD_SET(hh[0], hh[0]);
    __SIMD_DATATYPE tau2 = _SIMD_SET(hh[ldh], hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_SET(hh[ldh*2], hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_SET(hh[ldh*3], hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_SET(hh[ldh*4], hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_SET(hh[ldh*5], hh[ldh*5]);
#endif

#ifdef BLOCK2
    __SIMD_DATATYPE vs = _SIMD_SET(s, s);
#endif
#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET(s_1_2, s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET(s_1_3, s_1_3);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET(s_2_3, s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET(s_1_4, s_1_4);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET(s_2_4, s_2_4);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET(s_3_4, s_3_4);

#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_SET(scalarprods[0], scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_SET(scalarprods[1], scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_SET(scalarprods[2], scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_SET(scalarprods[3], scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_SET(scalarprods[4], scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_SET(scalarprods[5], scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_SET(scalarprods[6], scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_SET(scalarprods[7], scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_SET(scalarprods[8], scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_SET(scalarprods[9], scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_SET(scalarprods[10], scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_SET(scalarprods[11], scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_SET(scalarprods[12], scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_SET(scalarprods[13], scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_SET(scalarprods[14], scalarprods[14]);
#endif
#endif /* VEC_SET == SPARC64_SSE */

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   __SIMD_DATATYPE tau1 = _SIMD_BROADCAST(hh);
   __SIMD_DATATYPE tau2 = _SIMD_BROADCAST(&hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SIMD_DATATYPE tau3 = _SIMD_BROADCAST(&hh[ldh*2]);
   __SIMD_DATATYPE tau4 = _SIMD_BROADCAST(&hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE tau5 = _SIMD_BROADCAST(&hh[ldh*4]);
   __SIMD_DATATYPE tau6 = _SIMD_BROADCAST(&hh[ldh*5]);
#endif

#ifdef BLOCK2  
   __SIMD_DATATYPE vs = _SIMD_BROADCAST(&s);
#endif

#ifdef BLOCK4
   __SIMD_DATATYPE vs_1_2 = _SIMD_BROADCAST(&s_1_2);
   __SIMD_DATATYPE vs_1_3 = _SIMD_BROADCAST(&s_1_3);
   __SIMD_DATATYPE vs_2_3 = _SIMD_BROADCAST(&s_2_3);
   __SIMD_DATATYPE vs_1_4 = _SIMD_BROADCAST(&s_1_4);
   __SIMD_DATATYPE vs_2_4 = _SIMD_BROADCAST(&s_2_4);
   __SIMD_DATATYPE vs_3_4 = _SIMD_BROADCAST(&s_3_4);
#endif
#ifdef BLOCK6
   __SIMD_DATATYPE vs_1_2 = _SIMD_BROADCAST(&scalarprods[0]);
   __SIMD_DATATYPE vs_1_3 = _SIMD_BROADCAST(&scalarprods[1]);
   __SIMD_DATATYPE vs_2_3 = _SIMD_BROADCAST(&scalarprods[2]);
   __SIMD_DATATYPE vs_1_4 = _SIMD_BROADCAST(&scalarprods[3]);
   __SIMD_DATATYPE vs_2_4 = _SIMD_BROADCAST(&scalarprods[4]);
   __SIMD_DATATYPE vs_3_4 = _SIMD_BROADCAST(&scalarprods[5]);
   __SIMD_DATATYPE vs_1_5 = _SIMD_BROADCAST(&scalarprods[6]);
   __SIMD_DATATYPE vs_2_5 = _SIMD_BROADCAST(&scalarprods[7]);
   __SIMD_DATATYPE vs_3_5 = _SIMD_BROADCAST(&scalarprods[8]);
   __SIMD_DATATYPE vs_4_5 = _SIMD_BROADCAST(&scalarprods[9]);
   __SIMD_DATATYPE vs_1_6 = _SIMD_BROADCAST(&scalarprods[10]);
   __SIMD_DATATYPE vs_2_6 = _SIMD_BROADCAST(&scalarprods[11]);
   __SIMD_DATATYPE vs_3_6 = _SIMD_BROADCAST(&scalarprods[12]);
   __SIMD_DATATYPE vs_4_6 = _SIMD_BROADCAST(&scalarprods[13]);
   __SIMD_DATATYPE vs_5_6 = _SIMD_BROADCAST(&scalarprods[14]);
#endif
#endif /* VEC_SET == AVX_256 || VEC_SET == AVX2_256 */

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == VSX_SSE || VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _XOR(tau1, sign);
#endif

#if VEC_SET == SPARC64_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
    h1 = _SIMD_NEG( ADDITIONAL_ARGUMENT tau1);
#endif

#if VEC_SET == AVX_512
#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi64((__SIMD_INTEGER) tau1, (__SIMD_INTEGER) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi32((__SIMD_INTEGER) tau1, (__SIMD_INTEGER) sign);
#endif
#endif /* HAVE_AVX512_XEON_PHI */

#ifdef HAVE_AVX512_XEON
    h1 = _XOR(tau1, sign);
#endif
#endif /* VEC_SET == AVX_512 */
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau1;
#endif

   x1 = _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1);

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == VSX_SSE || VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _XOR(tau2, sign);
#endif

#if VEC_SET == SPARC64_SSE || VEC_SET == NEON_ARCH64_128 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128
   h1 = _SIMD_NEG( ADDITIONAL_ARGUMENT tau2);
#endif

#if VEC_SET == AVX_512
#ifdef HAVE_AVX512_XEON_PHI
#ifdef DOUBLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi64((__SIMD_INTEGER) tau2, (__SIMD_INTEGER) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
    h1 = (__SIMD_DATATYPE) _mm512_xor_epi32((__SIMD_INTEGER) tau2, (__SIMD_INTEGER) sign);
#endif
#endif /* HAVE_AVX512_XEON_PHI */

#ifdef HAVE_AVX512_XEON
    h1 = _XOR(tau2, sign);
#endif
#endif /* VEC_SET == AVX_512 */
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs);
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau2;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_2); 
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK2

#ifdef __ELPA_USE_FMA__
   y1 = _SIMD_FMA(y1, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
#else
   y1 = _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
#endif
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)

#ifdef __ELPA_USE_FMA__
   y1 = _SIMD_FMSUB(y1, h1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
#else
   y1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2));
#endif /* __ELPA_USE_FMA__ */

   h1 = tau3;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_3);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_2_3);

#ifdef __ELPA_USE_FMA__
   z1 = _SIMD_FMSUB(z1, h1, _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)));
#else
   z1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)));
#endif /* __ELPA_USE_FMA__ */

   h1 = tau4;
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_1_4);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_2_4);
   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT h1, vs_3_4);

#ifdef __ELPA_USE_FMA__
   w1 = _SIMD_FMSUB(w1, h1, _SIMD_FMA(z1, h4, _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
#else
   w1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h1), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)))); 
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_1_5); 
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_2_5);

   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_3_5);
   h5 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau5, vs_4_5);


#ifdef __ELPA_USE_FMA__
   v1 = _SIMD_FMSUB(v1, tau5, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w1, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
#else
   v1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT v1,tau5), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2))));
#endif /* __ELPA_USE_FMA__ */

   h2 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_1_6);
   h3 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_2_6);
   h4 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_3_6);
   h5 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_4_6);
   h6 = _SIMD_MUL( ADDITIONAL_ARGUMENT tau6, vs_5_6);

#ifdef __ELPA_USE_FMA__
   t1 = _SIMD_FMSUB(t1, tau6, _SIMD_FMA(v1, h6, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_FMA(w1, h5, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_FMA(y1, h3, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)))));
#else
   t1 = _SIMD_SUB( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT t1,tau6), _SIMD_ADD( ADDITIONAL_ARGUMENT  _SIMD_MUL( ADDITIONAL_ARGUMENT v1,h6), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h5), _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h4)), _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h3), _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h2)))));
#endif /* __ELPA_USE_FMA__ */

   /////////////////////////////////////////////////////
   // Rank-1 update of Q [ROW_LENGTH x nb+3]
   /////////////////////////////////////////////////////
#endif /* BLOCK6 */

   q1 = _LOAD(&q[0]);
#ifdef BLOCK2
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, y1);
#endif
#ifdef BLOCK4
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, w1);
#endif
#ifdef BLOCK6
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, t1); 
#endif
   _STORE(&q[0],q1);

#ifdef BLOCK2
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _LOAD(&q[ldq]);
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_FMA(y1, h2, x1));
   _STORE(&q[ldq],q1);
#else
   q1 = _LOAD(&q[ldq]);
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT x1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2)));
   _STORE(&q[ldq],q1);
#endif /* __ELPA_USE_FMA__ */
#endif /* BLOCK2 */

#ifdef BLOCK4
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
#endif

   q1 = _LOAD(&q[ldq]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_FMA(w1, h4, z1));
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT z1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4)));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq],q1);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+2]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
#endif

   q1 = _LOAD(&q[ldq*2]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, y1);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*2],q1);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

   q1 = _LOAD(&q[ldq*3]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, x1);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+1], hh[ldh+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*3], q1);

#endif /* BLOCK4 */

#ifdef BLOCK6
   
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+1]);
#endif

   q1 = _LOAD(&q[ldq]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, v1);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
#endif

   _STORE(&q[ldq],q1);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+1]);
#endif
   q1 = _LOAD(&q[ldq*2]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, w1);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5)); 
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[ldq*2],q1);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+1]);
#endif

#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+1]);
#endif
   q1 = _LOAD(&q[ldq*3]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, z1);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
#endif

   _STORE(&q[ldq*3],q1);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+1]);
#endif

   q1 = _LOAD(&q[ldq*4]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, y1);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+4]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+4]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
#endif

   _STORE(&q[ldq*4],q1);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[(ldh)+1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[(ldh)+1], hh[(ldh)+1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[(ldh)+1]);
#endif
   q1 = _LOAD(&q[ldq*5]);
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, x1);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+2]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+3]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+3]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+4]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+4]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h6 = _SIMD_SET1(hh[(ldh*5)+5]);
#endif
#if VEC_SET == SPARC64_SSE
   h6 = _SIMD_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h6 = _SIMD_BROADCAST(&hh[(ldh*5)+5]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(t1, h6, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
#endif

   _STORE(&q[ldq*5],q1);

#endif /* BLOCK6 */

   for (i = BLOCK; i < nb; i++)
   {
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h1 = _SIMD_SET1(hh[i-(BLOCK-1)]);
     h2 = _SIMD_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == SPARC64_SSE
     h1 = _SIMD_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
     h2 = _SIMD_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
    h1 = _SIMD_BROADCAST(&hh[i-(BLOCK-1)]);
    h2 = _SIMD_BROADCAST(&hh[ldh+i-(BLOCK-2)]);
#endif

     q1 = _LOAD(&q[i*ldq]);

#ifdef BLOCK2
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_FMA(x1, h1, q1);
     q1 = _SIMD_FMA(y1, h2, q1);
#else
     q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_ADD( ADDITIONAL_ARGUMENT _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h1), _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2)));
#endif /* __ELPA_USE_FMA__ */
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
     
#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(x1, h1, q1);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1,h1));
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(y1, h2, q1);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1,h2));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h3 = _SIMD_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
     h3 = _SIMD_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h3 = _SIMD_BROADCAST(&hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(z1, h3, q1);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1,h3));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h4 = _SIMD_SET1(hh[(ldh*3)+i-(BLOCK-4)]); 
#endif

#if VEC_SET == SPARC64_SSE
     h4 = _SIMD_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h4 = _SIMD_BROADCAST(&hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(w1, h4, q1);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1,h4));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK4 || BLOCK6*/

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h5 = _SIMD_SET1(hh[(ldh*4)+i-(BLOCK-5)]);
#endif
#if VEC_SET == SPARC64_SSE
     h5 = _SIMD_SET(hh[(ldh*4)+i-(BLOCK-5)], hh[(ldh*4)+i-(BLOCK-5)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h5 = _SIMD_BROADCAST(&hh[(ldh*4)+i-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(v1, h5, q1);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
     h6 = _SIMD_SET1(hh[(ldh*5)+i-(BLOCK-6)]);
#endif
#if VEC_SET == SPARC64_SSE
     h6 = _SIMD_SET(hh[(ldh*5)+i], hh[(ldh*5)+i-(BLOCK-6)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
     h6 = _SIMD_BROADCAST(&hh[(ldh*5)+i-(BLOCK-6)]);
#endif

#ifdef __ELPA_USE_FMA__
     q1 = _SIMD_NFMA(t1, h6, q1);
#else
     q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT t1, h6));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

     _STORE(&q[i*ldq],q1);

   }
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-1)]);
#endif

   q1 = _LOAD(&q[nb*ldq]);

#ifdef BLOCK2

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_FMA(x1, h1, q1);
#else
   q1 = _SIMD_ADD( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
#endif
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
#endif

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+nb-(BLOCK-4)]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+nb-(BLOCK-4)], hh[(ldh*3)+nb-(BLOCK-4)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h5 = _SIMD_SET1(hh[(ldh*4)+nb-(BLOCK-5)]);
#endif
#if VEC_SET == SPARC64_SSE
   h5 = _SIMD_SET(hh[(ldh*4)+nb-(BLOCK-5)], hh[(ldh*4)+nb-(BLOCK-5)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h5 = _SIMD_BROADCAST(&hh[(ldh*4)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(v1, h5, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT v1, h5));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

   _STORE(&q[nb*ldq],q1);

#if defined(BLOCK4) || defined(BLOCK6)
   
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-2)]);
#endif

#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-2)], hh[nb-(BLOCK-2)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-2)]);
#endif

   q1 = _LOAD(&q[(nb+1)*ldq]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-(BLOCK-3)], hh[ldh+nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
#endif

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-(BLOCK-4)]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-(BLOCK-4)], hh[(ldh*2)+nb-(BLOCK-4)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-(BLOCK-4)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
#endif

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h4 = _SIMD_SET1(hh[(ldh*3)+nb-(BLOCK-5)]);
#endif
#if VEC_SET == SPARC64_SSE
   h4 = _SIMD_SET(hh[(ldh*3)+nb-(BLOCK-5)], hh[(ldh*3)+nb-(BLOCK-5)]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h4 = _SIMD_BROADCAST(&hh[(ldh*3)+nb-(BLOCK-5)]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(w1, h4, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT w1, h4));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */
   _STORE(&q[(nb+1)*ldq],q1);

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-(BLOCK-3)]);
#endif

#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-(BLOCK-3)], hh[nb-(BLOCK-3)]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-(BLOCK-3)]);
#endif

   q1 = _LOAD(&q[(nb+2)*ldq]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
#endif

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-2]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h3 = _SIMD_SET1(hh[(ldh*2)+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h3 = _SIMD_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h3 = _SIMD_BROADCAST(&hh[(ldh*2)+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(z1, h3, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT z1, h3));
#endif /* __ELPA_USE_FMA__ */

#endif /* BLOCK6 */

   _STORE(&q[(nb+2)*ldq],q1);

#endif /* BLOCK4 || BLOCK6*/

#ifdef BLOCK6
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-2]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-2], hh[nb-2]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-2]);
#endif

   q1 = _LOAD(&q[(nb+3)*ldq]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
#endif /* __ELPA_USE_FMA__ */

#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h2 = _SIMD_SET1(hh[ldh+nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h2 = _SIMD_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h2 = _SIMD_BROADCAST(&hh[ldh+nb-1]);
#endif

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(y1, h2, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT y1, h2));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[(nb+3)*ldq],q1);
#if VEC_SET == SSE_128 || VEC_SET == AVX_512 || VEC_SET == SVE_512 || VEC_SET == SVE_256 || VEC_SET == SVE_128 || VEC_SET == VSX_SSE || VEC_SET == NEON_ARCH64_128
   h1 = _SIMD_SET1(hh[nb-1]);
#endif
#if VEC_SET == SPARC64_SSE
   h1 = _SIMD_SET(hh[nb-1], hh[nb-1]);
#endif
#if VEC_SET == AVX_256 || VEC_SET == AVX2_256
   h1 = _SIMD_BROADCAST(&hh[nb-1]);
#endif

   q1 = _LOAD(&q[(nb+4)*ldq]);

#ifdef __ELPA_USE_FMA__
   q1 = _SIMD_NFMA(x1, h1, q1);
#else
   q1 = _SIMD_SUB( ADDITIONAL_ARGUMENT q1, _SIMD_MUL( ADDITIONAL_ARGUMENT x1, h1));
#endif /* __ELPA_USE_FMA__ */

   _STORE(&q[(nb+4)*ldq],q1);

#endif /* BLOCK6 */

}

#undef SIMD_SET
#undef OFFSET
