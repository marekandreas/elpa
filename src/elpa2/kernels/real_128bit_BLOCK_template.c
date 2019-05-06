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



#ifdef HAVE_SSE_INTRINSICS
#include <x86intrin.h>
#endif
#ifdef HAVE_SPARC64_SSE
#include <fjmfunc.h>
#include <emmintrin.h>
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

#ifdef HAVE_SSE_INTRINSICS
#define SIMD_SET SSE
#endif

#ifdef HAVE_SPARC64_SSE
#define SIMD_SET SPARC64
#endif
#define __forceinline __attribute__((always_inline)) static

#ifdef DOUBLE_PRECISION_REAL
#define offset 2
#define WORD_LENGTH double
#define DATA_TYPE double
#define DATA_TYPE_PTR double*

#define __SSE_DATATYPE __m128d
#define _SSE_LOAD _mm_load_pd
#define _SSE_ADD _mm_add_pd
#define _SSE_SUB _mm_sub_pd
#define _SSE_MUL _mm_mul_pd
#define _SSE_XOR _mm_xor_pd
#define _SSE_STORE _mm_store_pd
#define _SSE_SET _mm_set_pd
#define _SSE_SET1 _mm_set1_pd
#define _SSE_SET _mm_set_pd
#endif

#ifdef SINGLE_PRECISION_REAL
#define offset 4
#define WORD_LENGTH single
#define DATA_TYPE float
#define DATA_TYPE_PTR float*

#define __SSE_DATATYPE __m128
#define _SSE_LOAD _mm_load_ps
#define _SSE_ADD _mm_add_ps
#define _SSE_SUB _mm_sub_ps
#define _SSE_MUL _mm_mul_ps
#define _SSE_XOR _mm_xor_ps
#define _SSE_STORE _mm_store_ps
#define _SSE_SET _mm_set_ps
#define _SSE_SET1 _mm_set1_ps
#define _SSE_SET _mm_set_ps

#endif

#ifdef HAVE_SSE_INTRINSICS
#undef __AVX__
#endif

//Forward declaration
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 2
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 4
#endif

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

#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 8
#endif
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

#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 6
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 12
#endif
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

#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 16
#endif
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

#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 10
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 20
#endif
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

#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 12
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 24
#endif

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
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine hexa_hh_trafo_real_sse_6hv_double(q, hh, pnb, pnq, pldq, pldh) &
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
!f>   subroutine hexa_hh_trafo_real_sparc64_6hv_double(q, hh, pnb, pnq, pldq, pldh) &
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
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine hexa_hh_trafo_real_sse_6hv_single(q, hh, pnb, pnq, pldq, pldh) &
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
!f>   subroutine hexa_hh_trafo_real_sparc64_6hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                                bind(C, name="hexa_hh_trafo_real_SPARC64_6hv_single")
!f>        use, intrinsic :: iso_c_binding
!f>        integer(kind=c_int)        :: pnb, pnq, pldq, pldh
!f>        type(c_ptr), value        :: q
!f>        real(kind=c_float)        :: hh(pnb,6)
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

#ifdef BLOCK2
  // calculating scalar product to compute
  // 2 householder vectors simultaneously
  DATA_TYPE s = hh[(ldh)+1]*1.0;
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

#ifdef HAVE_SSE_INTRINSICS
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
#ifdef DOUBLE_PRECISION_REAL
  for (i = 0; i < nq-10; i+=12)
    {
      CONCAT_4ARGS(hh_trafo_kernel_12_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
      worked_on += 12;
    }
#endif
#ifdef SINGLE_PRECISION_REAL
  for (i = 0; i < nq-20; i+=24)
    {
      CONCAT_4ARGS(hh_trafo_kernel_24_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
      worked_on += 24;
    }
#endif

  if (nq == i)
    {
      return;
    }

#ifdef DOUBLE_PRECISION_REAL
  if (nq-i == 10)
    {
      CONCAT_4ARGS(hh_trafo_kernel_10_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
      worked_on += 10;
    }
#endif

#ifdef SINGLE_PRECISION_REAL
  if (nq-i == 20)
    {
      CONCAT_4ARGS(hh_trafo_kernel_20_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
      worked_on += 20;
    }
#endif

#ifdef DOUBLE_PRECISION_REAL
  if (nq-i == 8)
    {
      CONCAT_4ARGS(hh_trafo_kernel_8_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
      worked_on += 8;
    }
#endif

#ifdef SINGLE_PRECISION_REAL
  if (nq-i == 16)
    {
      CONCAT_4ARGS(hh_trafo_kernel_16_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
      worked_on += 16;
    }
#endif

#ifdef DOUBLE_PRECISION_REAL
  if (nq-i == 6)
    {
      CONCAT_4ARGS(hh_trafo_kernel_6_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
      worked_on += 6;
    }
#endif

#ifdef SINGLE_PRECISION_REAL
  if (nq-i == 12)
    {
      CONCAT_4ARGS(hh_trafo_kernel_12_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
      worked_on += 12;
    }
#endif

#ifdef DOUBLE_PRECISION_REAL
  if (nq-i == 4)
    {
      CONCAT_4ARGS(hh_trafo_kernel_4_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
      worked_on += 4;
    }
#endif

#ifdef SINGLE_PRECISION_REAL
   if (nq-i == 8)
     {
       CONCAT_4ARGS(hh_trafo_kernel_8_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
       worked_on += 8;
     }
#endif

#ifdef DOUBLE_PRECISION_REAL
  if (nq-i == 2)
    {
      CONCAT_4ARGS(hh_trafo_kernel_2_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
      worked_on += 2;
    }
#endif

#ifdef SINGLE_PRECISION_REAL
  if (nq-i == 4)
    {
      CONCAT_4ARGS(hh_trafo_kernel_4_,SIMD_SET,_2hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s);
      worked_on += 4;
    }
#endif

#endif /* BLOCK2 */

#ifdef BLOCK4
#ifdef DOUBLE_PRECISION_REAL
  for (i = 0; i < nq-4; i+=6)
    {
      CONCAT_4ARGS(hh_trafo_kernel_6_,SIMD_SET,_4hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
      worked_on += 6;
    }
#endif

#ifdef SINGLE_PRECISION_REAL
  for (i = 0; i < nq-8; i+=12)
    {
      CONCAT_4ARGS(hh_trafo_kernel_12_,SIMD_SET,_4hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
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
      CONCAT_4ARGS(hh_trafo_kernel_4_,SIMD_SET,_4hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
      worked_on += 4;
    }
#endif

#ifdef SINGLE_PRECISION_REAL
  if (nq-i ==8)
    {
      CONCAT_4ARGS(hh_trafo_kernel_8_,SIMD_SET,_4hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
      worked_on += 8;
    }
#endif

#ifdef DOUBLE_PRECISION_REAL
   if (nq-i == 2)
     {
       CONCAT_4ARGS(hh_trafo_kernel_2_,SIMD_SET,_4hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
       worked_on += 2;
     }
#endif

#ifdef SINGLE_PRECISION_REAL
  if (nq-i == 4)
    {
      CONCAT_4ARGS(hh_trafo_kernel_4_,SIMD_SET,_4hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
      worked_on += 4;
    }
#endif

#endif /* BLOCK4 */

#ifdef BLOCK6
#ifdef DOUBLE_PRECISION_REAL
  
  for (i = 0; i < nq-2; i+=4)
    {
      CONCAT_4ARGS(hh_trafo_kernel_4_,SIMD_SET,_6hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, scalarprods);
      worked_on += 4;
    }
#endif
#ifdef SINGLE_PRECISION_REAL
    for (i = 0; i < nq-4; i+=8)
      {
        CONCAT_4ARGS(hh_trafo_kernel_8_,SIMD_SET,_6hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, scalarprods);
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
        CONCAT_4ARGS(hh_trafo_kernel_2_,SIMD_SET,_6hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, scalarprods);
        worked_on += 2;
      }
#endif
#ifdef SINGLE_PRECISION_REAL
    if (nq -i == 4)
      {
        CONCAT_4ARGS(hh_trafo_kernel_4_,SIMD_SET,_6hv_,WORD_LENGTH) (&q[i], hh, nb, ldq, ldh, scalarprods);
        worked_on += 4;
      }
#endif
  
#endif /* BLOCK6 */

#ifdef WITH_DEBUG
  if (worked_on != nq)
    {
      printf("Error in real SIMD_SET BLOCK BLOCK kernel %d %d\n", worked_on, nq);
      abort();
    }
#endif
}

/*
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 12 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 24 rows of Q simultaneously, a
#endif
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
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 12
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 24
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
#ifdef HAVE_SSE_INTRINSICS
    // Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
    __SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SSE_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif
    __SSE_DATATYPE x1 = _SSE_LOAD(&q[ldq]);
    __SSE_DATATYPE x2 = _SSE_LOAD(&q[ldq+offset]);
    __SSE_DATATYPE x3 = _SSE_LOAD(&q[ldq+2*offset]);
    __SSE_DATATYPE x4 = _SSE_LOAD(&q[ldq+3*offset]);
    __SSE_DATATYPE x5 = _SSE_LOAD(&q[ldq+4*offset]);
    __SSE_DATATYPE x6 = _SSE_LOAD(&q[ldq+5*offset]);

#ifdef HAVE_SSE_INTRINSICS
    __SSE_DATATYPE h1 = _SSE_SET1(hh[ldh+1]);
#endif
#ifdef HAVE_SPARC64_SSE
    __SSE_DATATYPE h1 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
#endif
    __SSE_DATATYPE h2;

    __SSE_DATATYPE q1 = _SSE_LOAD(q);
    __SSE_DATATYPE y1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
    __SSE_DATATYPE q2 = _SSE_LOAD(&q[offset]);
    __SSE_DATATYPE y2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
    __SSE_DATATYPE q3 = _SSE_LOAD(&q[2*offset]);
    __SSE_DATATYPE y3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
    __SSE_DATATYPE q4 = _SSE_LOAD(&q[3*offset]);
    __SSE_DATATYPE y4 = _SSE_ADD(q4, _SSE_MUL(x4, h1));
    __SSE_DATATYPE q5 = _SSE_LOAD(&q[4*offset]);
    __SSE_DATATYPE y5 = _SSE_ADD(q5, _SSE_MUL(x5, h1));
    __SSE_DATATYPE q6 = _SSE_LOAD(&q[5*offset]);
    __SSE_DATATYPE y6 = _SSE_ADD(q6, _SSE_MUL(x6, h1));
#endif /* BLOCK2 */

#ifdef BLOCK4
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

    register __SSE_DATATYPE w1 = _SSE_ADD(a4_1, _SSE_MUL(a3_1, h_4_3));
    w1 = _SSE_ADD(w1, _SSE_MUL(a2_1, h_4_2));                          
    w1 = _SSE_ADD(w1, _SSE_MUL(a1_1, h_4_1));                          
    register __SSE_DATATYPE z1 = _SSE_ADD(a3_1, _SSE_MUL(a2_1, h_3_2));
    z1 = _SSE_ADD(z1, _SSE_MUL(a1_1, h_3_1));                          
    register __SSE_DATATYPE y1 = _SSE_ADD(a2_1, _SSE_MUL(a1_1, h_2_1));
    register __SSE_DATATYPE x1 = a1_1;

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

    __SSE_DATATYPE a1_4 = _SSE_LOAD(&q[(ldq*3)+3*offset]);
    __SSE_DATATYPE a2_4 = _SSE_LOAD(&q[(ldq*2)+3*offset]);
    __SSE_DATATYPE a3_4 = _SSE_LOAD(&q[ldq+3*offset]);
    __SSE_DATATYPE a4_4 = _SSE_LOAD(&q[0+3*offset]);

    register __SSE_DATATYPE w4 = _SSE_ADD(a4_4, _SSE_MUL(a3_4, h_4_3));
    w4 = _SSE_ADD(w4, _SSE_MUL(a2_4, h_4_2));
    w4 = _SSE_ADD(w4, _SSE_MUL(a1_4, h_4_1));
    register __SSE_DATATYPE z4 = _SSE_ADD(a3_4, _SSE_MUL(a2_4, h_3_2));
    z4 = _SSE_ADD(z4, _SSE_MUL(a1_4, h_3_1));
    register __SSE_DATATYPE y4 = _SSE_ADD(a2_4, _SSE_MUL(a1_4, h_2_1));
    register __SSE_DATATYPE x4 = a1_4;

    __SSE_DATATYPE a1_5 = _SSE_LOAD(&q[(ldq*3)+4*offset]);
    __SSE_DATATYPE a2_5 = _SSE_LOAD(&q[(ldq*2)+4*offset]);
    __SSE_DATATYPE a3_5 = _SSE_LOAD(&q[ldq+4*offset]);
    __SSE_DATATYPE a4_5 = _SSE_LOAD(&q[0+4*offset]);

    register __SSE_DATATYPE w5 = _SSE_ADD(a4_5, _SSE_MUL(a3_5, h_4_3));
    w5 = _SSE_ADD(w5, _SSE_MUL(a2_5, h_4_2));
    w5 = _SSE_ADD(w5, _SSE_MUL(a1_5, h_4_1));
    register __SSE_DATATYPE z5 = _SSE_ADD(a3_5, _SSE_MUL(a2_5, h_3_2));
    z5 = _SSE_ADD(z5, _SSE_MUL(a1_5, h_3_1));
    register __SSE_DATATYPE y5 = _SSE_ADD(a2_5, _SSE_MUL(a1_5, h_2_1));
    register __SSE_DATATYPE x5 = a1_5;

    __SSE_DATATYPE a1_6 = _SSE_LOAD(&q[(ldq*3)+5*offset]);
    __SSE_DATATYPE a2_6 = _SSE_LOAD(&q[(ldq*2)+5*offset]);
    __SSE_DATATYPE a3_6 = _SSE_LOAD(&q[ldq+5*offset]);
    __SSE_DATATYPE a4_6 = _SSE_LOAD(&q[0+5*offset]);

    register __SSE_DATATYPE w6 = _SSE_ADD(a4_6, _SSE_MUL(a3_6, h_4_3));
    w6 = _SSE_ADD(w6, _SSE_MUL(a2_6, h_4_2));
    w6 = _SSE_ADD(w6, _SSE_MUL(a1_6, h_4_1));
    register __SSE_DATATYPE z6 = _SSE_ADD(a3_6, _SSE_MUL(a2_6, h_3_2));
    z6 = _SSE_ADD(z6, _SSE_MUL(a1_6, h_3_1));
    register __SSE_DATATYPE y6 = _SSE_ADD(a2_6, _SSE_MUL(a1_6, h_2_1));
    register __SSE_DATATYPE x6 = a1_6;

    __SSE_DATATYPE q1;
    __SSE_DATATYPE q2;
    __SSE_DATATYPE q3;
    __SSE_DATATYPE q4;
    __SSE_DATATYPE q5;
    __SSE_DATATYPE q6;

    __SSE_DATATYPE h1;
    __SSE_DATATYPE h2;
    __SSE_DATATYPE h3;
    __SSE_DATATYPE h4;
#endif /* BLOCK4 */

#ifdef BLOCK6
    
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

    __SSE_DATATYPE a1_3 = _SSE_LOAD(&q[(ldq*5)+2*offset]);
    __SSE_DATATYPE a2_3 = _SSE_LOAD(&q[(ldq*4)+2*offset]);
    __SSE_DATATYPE a3_3 = _SSE_LOAD(&q[(ldq*3)+2*offset]);
    __SSE_DATATYPE a4_3 = _SSE_LOAD(&q[(ldq*2)+2*offset]);
    __SSE_DATATYPE a5_3 = _SSE_LOAD(&q[(ldq)+2*offset]);
    __SSE_DATATYPE a6_3 = _SSE_LOAD(&q[2*offset]);

    register __SSE_DATATYPE t3 = _SSE_ADD(a6_3, _SSE_MUL(a5_3, h_6_5));
    t3 = _SSE_ADD(t3, _SSE_MUL(a4_3, h_6_4));
    t3 = _SSE_ADD(t3, _SSE_MUL(a3_3, h_6_3));
    t3 = _SSE_ADD(t3, _SSE_MUL(a2_3, h_6_2));
    t3 = _SSE_ADD(t3, _SSE_MUL(a1_3, h_6_1));
    register __SSE_DATATYPE v3 = _SSE_ADD(a5_3, _SSE_MUL(a4_3, h_5_4));
    v3 = _SSE_ADD(v3, _SSE_MUL(a3_3, h_5_3));
    v3 = _SSE_ADD(v3, _SSE_MUL(a2_3, h_5_2));
    v3 = _SSE_ADD(v3, _SSE_MUL(a1_3, h_5_1));
    register __SSE_DATATYPE w3 = _SSE_ADD(a4_3, _SSE_MUL(a3_3, h_4_3));
    w3 = _SSE_ADD(w3, _SSE_MUL(a2_3, h_4_2));
    w3 = _SSE_ADD(w3, _SSE_MUL(a1_3, h_4_1));
    register __SSE_DATATYPE z3 = _SSE_ADD(a3_3, _SSE_MUL(a2_3, h_3_2));
    z3 = _SSE_ADD(z3, _SSE_MUL(a1_3, h_3_1));
    register __SSE_DATATYPE y3 = _SSE_ADD(a2_3, _SSE_MUL(a1_3, h_2_1));

    register __SSE_DATATYPE x3 = a1_3;

    __SSE_DATATYPE a1_4 = _SSE_LOAD(&q[(ldq*5)+3*offset]);
    __SSE_DATATYPE a2_4 = _SSE_LOAD(&q[(ldq*4)+3*offset]);
    __SSE_DATATYPE a3_4 = _SSE_LOAD(&q[(ldq*3)+3*offset]);
    __SSE_DATATYPE a4_4 = _SSE_LOAD(&q[(ldq*2)+3*offset]);
    __SSE_DATATYPE a5_4 = _SSE_LOAD(&q[(ldq)+3*offset]);
    __SSE_DATATYPE a6_4 = _SSE_LOAD(&q[3*offset]);

    register __SSE_DATATYPE t4 = _SSE_ADD(a6_4, _SSE_MUL(a5_4, h_6_5));
    t4 = _SSE_ADD(t4, _SSE_MUL(a4_4, h_6_4));
    t4 = _SSE_ADD(t4, _SSE_MUL(a3_4, h_6_3));
    t4 = _SSE_ADD(t4, _SSE_MUL(a2_4, h_6_2));
    t4 = _SSE_ADD(t4, _SSE_MUL(a1_4, h_6_1));
    register __SSE_DATATYPE v4 = _SSE_ADD(a5_4, _SSE_MUL(a4_4, h_5_4));
    v4 = _SSE_ADD(v4, _SSE_MUL(a3_4, h_5_3));
    v4 = _SSE_ADD(v4, _SSE_MUL(a2_4, h_5_2));
    v4 = _SSE_ADD(v4, _SSE_MUL(a1_4, h_5_1));
    register __SSE_DATATYPE w4 = _SSE_ADD(a4_4, _SSE_MUL(a3_4, h_4_3));
    w4 = _SSE_ADD(w4, _SSE_MUL(a2_4, h_4_2));
    w4 = _SSE_ADD(w4, _SSE_MUL(a1_4, h_4_1));
    register __SSE_DATATYPE z4 = _SSE_ADD(a3_4, _SSE_MUL(a2_4, h_3_2));
    z4 = _SSE_ADD(z4, _SSE_MUL(a1_4, h_3_1));
    register __SSE_DATATYPE y4 = _SSE_ADD(a2_4, _SSE_MUL(a1_4, h_2_1));

    register __SSE_DATATYPE x4 = a1_4;

    __SSE_DATATYPE a1_5 = _SSE_LOAD(&q[(ldq*5)+4*offset]);
    __SSE_DATATYPE a2_5 = _SSE_LOAD(&q[(ldq*4)+4*offset]);
    __SSE_DATATYPE a3_5 = _SSE_LOAD(&q[(ldq*3)+4*offset]);
    __SSE_DATATYPE a4_5 = _SSE_LOAD(&q[(ldq*2)+4*offset]);
    __SSE_DATATYPE a5_5 = _SSE_LOAD(&q[(ldq)+4*offset]);
    __SSE_DATATYPE a6_5 = _SSE_LOAD(&q[4*offset]);

    register __SSE_DATATYPE t5 = _SSE_ADD(a6_5, _SSE_MUL(a5_5, h_6_5));
    t5 = _SSE_ADD(t5, _SSE_MUL(a4_5, h_6_4));
    t5 = _SSE_ADD(t5, _SSE_MUL(a3_5, h_6_3));
    t5 = _SSE_ADD(t5, _SSE_MUL(a2_5, h_6_2));
    t5 = _SSE_ADD(t5, _SSE_MUL(a1_5, h_6_1));
    register __SSE_DATATYPE v5 = _SSE_ADD(a5_5, _SSE_MUL(a4_5, h_5_4));
    v5 = _SSE_ADD(v5, _SSE_MUL(a3_5, h_5_3));
    v5 = _SSE_ADD(v5, _SSE_MUL(a2_5, h_5_2));
    v5 = _SSE_ADD(v5, _SSE_MUL(a1_5, h_5_1));
    register __SSE_DATATYPE w5 = _SSE_ADD(a4_5, _SSE_MUL(a3_5, h_4_3));
    w5 = _SSE_ADD(w5, _SSE_MUL(a2_5, h_4_2));
    w5 = _SSE_ADD(w5, _SSE_MUL(a1_5, h_4_1));
    register __SSE_DATATYPE z5 = _SSE_ADD(a3_5, _SSE_MUL(a2_5, h_3_2));
    z5 = _SSE_ADD(z5, _SSE_MUL(a1_5, h_3_1));
    register __SSE_DATATYPE y5 = _SSE_ADD(a2_5, _SSE_MUL(a1_5, h_2_1));

    register __SSE_DATATYPE x5 = a1_5;

    __SSE_DATATYPE a1_6 = _SSE_LOAD(&q[(ldq*5)+5*offset]);
    __SSE_DATATYPE a2_6 = _SSE_LOAD(&q[(ldq*4)+5*offset]);
    __SSE_DATATYPE a3_6 = _SSE_LOAD(&q[(ldq*3)+5*offset]);
    __SSE_DATATYPE a4_6 = _SSE_LOAD(&q[(ldq*2)+5*offset]);
    __SSE_DATATYPE a5_6 = _SSE_LOAD(&q[(ldq)+5*offset]);
    __SSE_DATATYPE a6_6 = _SSE_LOAD(&q[5*offset]);

    register __SSE_DATATYPE t6 = _SSE_ADD(a6_6, _SSE_MUL(a5_6, h_6_5));
    t6 = _SSE_ADD(t6, _SSE_MUL(a4_6, h_6_4));
    t6 = _SSE_ADD(t6, _SSE_MUL(a3_6, h_6_3));
    t6 = _SSE_ADD(t6, _SSE_MUL(a2_6, h_6_2));
    t6 = _SSE_ADD(t6, _SSE_MUL(a1_6, h_6_1));
    register __SSE_DATATYPE v6 = _SSE_ADD(a5_6, _SSE_MUL(a4_6, h_5_4));
    v6 = _SSE_ADD(v6, _SSE_MUL(a3_6, h_5_3));
    v6 = _SSE_ADD(v6, _SSE_MUL(a2_6, h_5_2));
    v6 = _SSE_ADD(v6, _SSE_MUL(a1_6, h_5_1));
    register __SSE_DATATYPE w6 = _SSE_ADD(a4_6, _SSE_MUL(a3_6, h_4_3));
    w6 = _SSE_ADD(w6, _SSE_MUL(a2_6, h_4_2));
    w6 = _SSE_ADD(w6, _SSE_MUL(a1_6, h_4_1));
    register __SSE_DATATYPE z6 = _SSE_ADD(a3_6, _SSE_MUL(a2_6, h_3_2));
    z6 = _SSE_ADD(z6, _SSE_MUL(a1_6, h_3_1));
    register __SSE_DATATYPE y6 = _SSE_ADD(a2_6, _SSE_MUL(a1_6, h_2_1));

    register __SSE_DATATYPE x6 = a1_6;

    __SSE_DATATYPE q1;
    __SSE_DATATYPE q2;
    __SSE_DATATYPE q3;
    __SSE_DATATYPE q4;
    __SSE_DATATYPE q5;
    __SSE_DATATYPE q6;

    __SSE_DATATYPE h1;
    __SSE_DATATYPE h2;
    __SSE_DATATYPE h3;
    __SSE_DATATYPE h4;
    __SSE_DATATYPE h5;
    __SSE_DATATYPE h6;

#endif /* BLOCK6 */


    for(i = BLOCK; i < nb; i++)
      {
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[i-(BLOCK-1)]);
        h2 = _SSE_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
        h2 = _SSE_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
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
        q4 = _SSE_LOAD(&q[(i*ldq)+3*offset]);
        x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
        y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
        q5 = _SSE_LOAD(&q[(i*ldq)+4*offset]);
        x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));
        y5 = _SSE_ADD(y5, _SSE_MUL(q5,h2));
        q6 = _SSE_LOAD(&q[(i*ldq)+5*offset]);
        x6 = _SSE_ADD(x6, _SSE_MUL(q6,h1));
        y6 = _SSE_ADD(y6, _SSE_MUL(q6,h2));

#if defined(BLOCK4) || defined(BLOCK6)
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

        z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
        z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
        z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));
        z4 = _SSE_ADD(z4, _SSE_MUL(q4,h3));
        z5 = _SSE_ADD(z5, _SSE_MUL(q5,h3));
        z6 = _SSE_ADD(z6, _SSE_MUL(q6,h3));
#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

        w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
        w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));
        w3 = _SSE_ADD(w3, _SSE_MUL(q3,h4));
        w4 = _SSE_ADD(w4, _SSE_MUL(q4,h4));
        w5 = _SSE_ADD(w5, _SSE_MUL(q5,h4));
        w6 = _SSE_ADD(w6, _SSE_MUL(q6,h4));
	
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+i-1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif
        v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));
        v2 = _SSE_ADD(v2, _SSE_MUL(q2,h5));
        v3 = _SSE_ADD(v3, _SSE_MUL(q3,h5));
        v4 = _SSE_ADD(v4, _SSE_MUL(q4,h5));
        v5 = _SSE_ADD(v5, _SSE_MUL(q5,h5));
        v6 = _SSE_ADD(v6, _SSE_MUL(q6,h5));

#ifdef HAVE_SSE_INTRINSICS
        h6 = _SSE_SET1(hh[(ldh*5)+i]);
#endif

#ifdef HAVE_SPARC64_SSE
        h6 = _SSE_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif

        t1 = _SSE_ADD(t1, _SSE_MUL(q1,h6));
        t2 = _SSE_ADD(t2, _SSE_MUL(q2,h6));
        t3 = _SSE_ADD(t3, _SSE_MUL(q3,h6));
        t4 = _SSE_ADD(t4, _SSE_MUL(q4,h6));
        t5 = _SSE_ADD(t5, _SSE_MUL(q5,h6));
        t6 = _SSE_ADD(t6, _SSE_MUL(q6,h6));
	
#endif /* BLOCK6 */
      }
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-(BLOCK-1)]);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif

    q1 = _SSE_LOAD(&q[nb*ldq]);
    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    q3 = _SSE_LOAD(&q[(nb*ldq)+2*offset]);
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    q4 = _SSE_LOAD(&q[(nb*ldq)+3*offset]);
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
    q5 = _SSE_LOAD(&q[(nb*ldq)+4*offset]);
    x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));
    q6 = _SSE_LOAD(&q[(nb*ldq)+5*offset]);
    x6 = _SSE_ADD(x6, _SSE_MUL(q6,h1));

#if defined(BLOCK4) || defined(BLOCK6)
    
#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
    y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
    y5 = _SSE_ADD(y5, _SSE_MUL(q5,h2));
    y6 = _SSE_ADD(y6, _SSE_MUL(q6,h2));

#ifdef HAVE_SSE_INTRINSICS
    h3 = _SSE_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
    h3 = _SSE_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

    z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
    z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
    z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));
    z4 = _SSE_ADD(z4, _SSE_MUL(q4,h3));
    z5 = _SSE_ADD(z5, _SSE_MUL(q5,h3));
    z6 = _SSE_ADD(z6, _SSE_MUL(q6,h3));

#ifdef BLOCK4
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

    q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+1)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+1)*ldq)+3*offset]);
    q5 = _SSE_LOAD(&q[((nb+1)*ldq)+4*offset]);
    q6 = _SSE_LOAD(&q[((nb+1)*ldq)+5*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
    x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));
    x6 = _SSE_ADD(x6, _SSE_MUL(q6,h1));

#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[(ldh*1)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[(ldh*1)+nb-1], hh[(ldh*1)+nb-1]);
#endif


    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
    y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
    y5 = _SSE_ADD(y5, _SSE_MUL(q5,h2));
    y6 = _SSE_ADD(y6, _SSE_MUL(q6,h2));

#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif


    q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+2)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+2)*ldq)+3*offset]);
    q5 = _SSE_LOAD(&q[((nb+2)*ldq)+4*offset]);
    q6 = _SSE_LOAD(&q[((nb+2)*ldq)+5*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
    x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));
    x6 = _SSE_ADD(x6, _SSE_MUL(q6,h1));

#endif /* BLOCK4 */
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#ifdef HAVE_SSE_INTRINSICS
    h4 = _SSE_SET1(hh[(ldh*3)+nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h4 = _SSE_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif

    w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4)); 
    w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));
    w3 = _SSE_ADD(w3, _SSE_MUL(q3,h4));
    w4 = _SSE_ADD(w4, _SSE_MUL(q4,h4));
    w5 = _SSE_ADD(w5, _SSE_MUL(q5,h4));
    w6 = _SSE_ADD(w6, _SSE_MUL(q6,h4));

#ifdef HAVE_SSE_INTRINSICS
    h5 = _SSE_SET1(hh[(ldh*4)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h5 = _SSE_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif

    v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));
    v2 = _SSE_ADD(v2, _SSE_MUL(q2,h5));
    v3 = _SSE_ADD(v3, _SSE_MUL(q3,h5));
    v4 = _SSE_ADD(v4, _SSE_MUL(q4,h5));
    v5 = _SSE_ADD(v5, _SSE_MUL(q5,h5));
    v6 = _SSE_ADD(v6, _SSE_MUL(q6,h5));
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-4]);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-4], hh[nb-4]);
#endif

    q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+1)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+1)*ldq)+3*offset]);
    q5 = _SSE_LOAD(&q[((nb+1)*ldq)+4*offset]);
    q6 = _SSE_LOAD(&q[((nb+1)*ldq)+5*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
    x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));
    x6 = _SSE_ADD(x6, _SSE_MUL(q6,h1));
#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-3]);
#endif
#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-3], hh[ldh+nb-3]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
    y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
    y5 = _SSE_ADD(y5, _SSE_MUL(q5,h2));
    y6 = _SSE_ADD(y6, _SSE_MUL(q6,h2));

#ifdef HAVE_SSE_INTRINSICS
    h3 = _SSE_SET1(hh[(ldh*2)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
    h3 = _SSE_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif

    z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
    z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
    z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));
    z4 = _SSE_ADD(z4, _SSE_MUL(q4,h3));
    z5 = _SSE_ADD(z5, _SSE_MUL(q5,h3));
    z6 = _SSE_ADD(z6, _SSE_MUL(q6,h3));

#ifdef HAVE_SSE_INTRINSICS
    h4 = _SSE_SET1(hh[(ldh*3)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
    h4 = _SSE_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif

    w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
    w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));
    w3 = _SSE_ADD(w3, _SSE_MUL(q3,h4));
    w4 = _SSE_ADD(w4, _SSE_MUL(q4,h4));
    w5 = _SSE_ADD(w5, _SSE_MUL(q5,h4));
    w6 = _SSE_ADD(w6, _SSE_MUL(q6,h4));

#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-3]);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-3], hh[nb-3]);
#endif

    q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+2)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+2)*ldq)+3*offset]);
    q5 = _SSE_LOAD(&q[((nb+2)*ldq)+4*offset]);
    q6 = _SSE_LOAD(&q[((nb+2)*ldq)+5*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
    x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));
    x6 = _SSE_ADD(x6, _SSE_MUL(q6,h1));

#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
    y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
    y5 = _SSE_ADD(y5, _SSE_MUL(q5,h2));
    y6 = _SSE_ADD(y6, _SSE_MUL(q6,h2));
#ifdef HAVE_SSE_INTRINSICS
    h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif

    z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
    z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
    z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));
    z4 = _SSE_ADD(z4, _SSE_MUL(q4,h3));
    z5 = _SSE_ADD(z5, _SSE_MUL(q5,h3));
    z6 = _SSE_ADD(z6, _SSE_MUL(q6,h3));
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

    q1 = _SSE_LOAD(&q[(nb+3)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+3)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+3)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+3)*ldq)+3*offset]);
    q5 = _SSE_LOAD(&q[((nb+3)*ldq)+4*offset]);
    q6 = _SSE_LOAD(&q[((nb+3)*ldq)+5*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
    x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));
    x6 = _SSE_ADD(x6, _SSE_MUL(q6,h1));

#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
    y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
    y5 = _SSE_ADD(y5, _SSE_MUL(q5,h2));
    y6 = _SSE_ADD(y6, _SSE_MUL(q6,h2));

#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif

    q1 = _SSE_LOAD(&q[(nb+4)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+4)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+4)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+4)*ldq)+3*offset]);
    q5 = _SSE_LOAD(&q[((nb+4)*ldq)+4*offset]);
    q6 = _SSE_LOAD(&q[((nb+4)*ldq)+5*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
    x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));
    x6 = _SSE_ADD(x6, _SSE_MUL(q6,h1));
#endif /* BLOCK6 */

#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Rank-2 update of Q [6 x nb+1]
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

#ifdef HAVE_SSE_INTRINSICS
    __SSE_DATATYPE tau1 = _SSE_SET1(hh[0]);

    __SSE_DATATYPE tau2 = _SSE_SET1(hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SSE_DATATYPE tau3 = _SSE_SET1(hh[ldh*2]);
   __SSE_DATATYPE tau4 = _SSE_SET1(hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE tau5 = _SSE_SET1(hh[ldh*4]);
   __SSE_DATATYPE tau6 = _SSE_SET1(hh[ldh*5]);       
#endif

#ifdef BLOCK2    
    __SSE_DATATYPE vs = _SSE_SET1(s);
#endif
#ifdef BLOCK4
   __SSE_DATATYPE vs_1_2 = _SSE_SET1(s_1_2);
   __SSE_DATATYPE vs_1_3 = _SSE_SET1(s_1_3);  
   __SSE_DATATYPE vs_2_3 = _SSE_SET1(s_2_3);
   __SSE_DATATYPE vs_1_4 = _SSE_SET1(s_1_4);  
   __SSE_DATATYPE vs_2_4 = _SSE_SET1(s_2_4);  
   __SSE_DATATYPE vs_3_4 = _SSE_SET1(s_3_4);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE vs_1_2 = _SSE_SET1(scalarprods[0]);
   __SSE_DATATYPE vs_1_3 = _SSE_SET1(scalarprods[1]);
   __SSE_DATATYPE vs_2_3 = _SSE_SET1(scalarprods[2]);
   __SSE_DATATYPE vs_1_4 = _SSE_SET1(scalarprods[3]);
   __SSE_DATATYPE vs_2_4 = _SSE_SET1(scalarprods[4]);
   __SSE_DATATYPE vs_3_4 = _SSE_SET1(scalarprods[5]);
   __SSE_DATATYPE vs_1_5 = _SSE_SET1(scalarprods[6]);
   __SSE_DATATYPE vs_2_5 = _SSE_SET1(scalarprods[7]);
   __SSE_DATATYPE vs_3_5 = _SSE_SET1(scalarprods[8]);
   __SSE_DATATYPE vs_4_5 = _SSE_SET1(scalarprods[9]);
   __SSE_DATATYPE vs_1_6 = _SSE_SET1(scalarprods[10]);
   __SSE_DATATYPE vs_2_6 = _SSE_SET1(scalarprods[11]);
   __SSE_DATATYPE vs_3_6 = _SSE_SET1(scalarprods[12]);
   __SSE_DATATYPE vs_4_6 = _SSE_SET1(scalarprods[13]);
   __SSE_DATATYPE vs_5_6 = _SSE_SET1(scalarprods[14]);
#endif
#endif /* HAVE_SSE_INTRINSICS */

#ifdef HAVE_SPARC64_SSE
    __SSE_DATATYPE tau1 = _SSE_SET(hh[0], hh[0]);

    __SSE_DATATYPE tau2 = _SSE_SET(hh[ldh], hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SSE_DATATYPE tau3 = _SSE_SET(hh[ldh*2], hh[ldh*2]);
   __SSE_DATATYPE tau4 = _SSE_SET(hh[ldh*3], hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE tau5 = _SSE_SET(hh[ldh*4], hh[ldh*4]);
   __SSE_DATATYPE tau6 = _SSE_SET(hh[ldh*5], hh[ldh*5]);
#endif

#ifdef BLOCK2
    __SSE_DATATYPE vs = _SSE_SET(s, s);
#endif
#ifdef BLOCK4
   __SSE_DATATYPE vs_1_2 = _SSE_SET(s_1_2, s_1_2);
   __SSE_DATATYPE vs_1_3 = _SSE_SET(s_1_3, s_1_3);
   __SSE_DATATYPE vs_2_3 = _SSE_SET(s_2_3, s_2_3);
   __SSE_DATATYPE vs_1_4 = _SSE_SET(s_1_4, s_1_4);
   __SSE_DATATYPE vs_2_4 = _SSE_SET(s_2_4, s_2_4);
   __SSE_DATATYPE vs_3_4 = _SSE_SET(s_3_4, s_3_4);

#endif
#ifdef BLOCK6
   __SSE_DATATYPE vs_1_2 = _SSE_SET(scalarprods[0], scalarprods[0]);
   __SSE_DATATYPE vs_1_3 = _SSE_SET(scalarprods[1], scalarprods[1]);
   __SSE_DATATYPE vs_2_3 = _SSE_SET(scalarprods[2], scalarprods[2]);
   __SSE_DATATYPE vs_1_4 = _SSE_SET(scalarprods[3], scalarprods[3]);
   __SSE_DATATYPE vs_2_4 = _SSE_SET(scalarprods[4], scalarprods[4]);
   __SSE_DATATYPE vs_3_4 = _SSE_SET(scalarprods[5], scalarprods[5]);
   __SSE_DATATYPE vs_1_5 = _SSE_SET(scalarprods[6], scalarprods[6]);
   __SSE_DATATYPE vs_2_5 = _SSE_SET(scalarprods[7], scalarprods[7]);
   __SSE_DATATYPE vs_3_5 = _SSE_SET(scalarprods[8], scalarprods[8]);
   __SSE_DATATYPE vs_4_5 = _SSE_SET(scalarprods[9], scalarprods[9]);
   __SSE_DATATYPE vs_1_6 = _SSE_SET(scalarprods[10], scalarprods[10]);
   __SSE_DATATYPE vs_2_6 = _SSE_SET(scalarprods[11], scalarprods[11]);
   __SSE_DATATYPE vs_3_6 = _SSE_SET(scalarprods[12], scalarprods[12]);
   __SSE_DATATYPE vs_4_6 = _SSE_SET(scalarprods[13], scalarprods[13]);
   __SSE_DATATYPE vs_5_6 = _SSE_SET(scalarprods[14], scalarprods[14]);
#endif
#endif /* HAVE_SPARC64_SSE */

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_XOR(tau1, sign);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _fjsp_neg_v2r8(tau1);
#endif
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau1;
#endif

   x1 = _SSE_MUL(x1, h1);
   x2 = _SSE_MUL(x2, h1);
   x3 = _SSE_MUL(x3, h1);
   x4 = _SSE_MUL(x4, h1);
   x5 = _SSE_MUL(x5, h1);
   x6 = _SSE_MUL(x6, h1);

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_XOR(tau2, sign);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _fjsp_neg_v2r8(tau2);
#endif
   h2 = _SSE_MUL(h1, vs);
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau2;
   h2 = _SSE_MUL(h1, vs_1_2);
#endif /* BLOCK4 || BLOCK6  */

#ifdef BLOCK2
   y1 = _SSE_ADD(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
   y2 = _SSE_ADD(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
   y3 = _SSE_ADD(_SSE_MUL(y3,h1), _SSE_MUL(x3,h2));
   y4 = _SSE_ADD(_SSE_MUL(y4,h1), _SSE_MUL(x4,h2));
   y5 = _SSE_ADD(_SSE_MUL(y5,h1), _SSE_MUL(x5,h2));
   y6 = _SSE_ADD(_SSE_MUL(y6,h1), _SSE_MUL(x6,h2));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   y1 = _SSE_SUB(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
   y2 = _SSE_SUB(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
   y3 = _SSE_SUB(_SSE_MUL(y3,h1), _SSE_MUL(x3,h2));
   y4 = _SSE_SUB(_SSE_MUL(y4,h1), _SSE_MUL(x4,h2));
   y5 = _SSE_SUB(_SSE_MUL(y5,h1), _SSE_MUL(x5,h2));
   y6 = _SSE_SUB(_SSE_MUL(y6,h1), _SSE_MUL(x6,h2));

   h1 = tau3;
   h2 = _SSE_MUL(h1, vs_1_3);
   h3 = _SSE_MUL(h1, vs_2_3);

   z1 = _SSE_SUB(_SSE_MUL(z1,h1), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)));
   z2 = _SSE_SUB(_SSE_MUL(z2,h1), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2)));
   z3 = _SSE_SUB(_SSE_MUL(z3,h1), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2)));
   z4 = _SSE_SUB(_SSE_MUL(z4,h1), _SSE_ADD(_SSE_MUL(y4,h3), _SSE_MUL(x4,h2)));
   z5 = _SSE_SUB(_SSE_MUL(z5,h1), _SSE_ADD(_SSE_MUL(y5,h3), _SSE_MUL(x5,h2)));
   z6 = _SSE_SUB(_SSE_MUL(z6,h1), _SSE_ADD(_SSE_MUL(y6,h3), _SSE_MUL(x6,h2)));

   h1 = tau4;
   h2 = _SSE_MUL(h1, vs_1_4);
   h3 = _SSE_MUL(h1, vs_2_4);
   h4 = _SSE_MUL(h1, vs_3_4);

   w1 = _SSE_SUB(_SSE_MUL(w1,h1), _SSE_ADD(_SSE_MUL(z1,h4), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));
   w2 = _SSE_SUB(_SSE_MUL(w2,h1), _SSE_ADD(_SSE_MUL(z2,h4), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2))));
   w3 = _SSE_SUB(_SSE_MUL(w3,h1), _SSE_ADD(_SSE_MUL(z3,h4), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2))));
   w4 = _SSE_SUB(_SSE_MUL(w4,h1), _SSE_ADD(_SSE_MUL(z4,h4), _SSE_ADD(_SSE_MUL(y4,h3), _SSE_MUL(x4,h2))));
   w5 = _SSE_SUB(_SSE_MUL(w5,h1), _SSE_ADD(_SSE_MUL(z5,h4), _SSE_ADD(_SSE_MUL(y5,h3), _SSE_MUL(x5,h2))));
   w6 = _SSE_SUB(_SSE_MUL(w6,h1), _SSE_ADD(_SSE_MUL(z6,h4), _SSE_ADD(_SSE_MUL(y6,h3), _SSE_MUL(x6,h2))));

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
   h2 = _SSE_MUL(tau5, vs_1_5); 
   h3 = _SSE_MUL(tau5, vs_2_5);
   h4 = _SSE_MUL(tau5, vs_3_5);
   h5 = _SSE_MUL(tau5, vs_4_5);

   v1 = _SSE_SUB(_SSE_MUL(v1,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));
   v2 = _SSE_SUB(_SSE_MUL(v2,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w2,h5), _SSE_MUL(z2,h4)), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2))));
   v3 = _SSE_SUB(_SSE_MUL(v3,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w3,h5), _SSE_MUL(z3,h4)), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2))));
   v4 = _SSE_SUB(_SSE_MUL(v4,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w4,h5), _SSE_MUL(z4,h4)), _SSE_ADD(_SSE_MUL(y4,h3), _SSE_MUL(x4,h2))));
   v5 = _SSE_SUB(_SSE_MUL(v5,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w5,h5), _SSE_MUL(z5,h4)), _SSE_ADD(_SSE_MUL(y5,h3), _SSE_MUL(x5,h2))));
   v6 = _SSE_SUB(_SSE_MUL(v6,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w6,h5), _SSE_MUL(z6,h4)), _SSE_ADD(_SSE_MUL(y6,h3), _SSE_MUL(x6,h2))));

   h2 = _SSE_MUL(tau6, vs_1_6);
   h3 = _SSE_MUL(tau6, vs_2_6);
   h4 = _SSE_MUL(tau6, vs_3_6);
   h5 = _SSE_MUL(tau6, vs_4_6);
   h6 = _SSE_MUL(tau6, vs_5_6);

   t1 = _SSE_SUB(_SSE_MUL(t1,tau6), _SSE_ADD( _SSE_MUL(v1,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)))));
   t2 = _SSE_SUB(_SSE_MUL(t2,tau6), _SSE_ADD( _SSE_MUL(v2,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w2,h5), _SSE_MUL(z2,h4)), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2)))));
   t3 = _SSE_SUB(_SSE_MUL(t3,tau6), _SSE_ADD( _SSE_MUL(v3,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w3,h5), _SSE_MUL(z3,h4)), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2)))));
   t4 = _SSE_SUB(_SSE_MUL(t4,tau6), _SSE_ADD( _SSE_MUL(v4,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w4,h5), _SSE_MUL(z4,h4)), _SSE_ADD(_SSE_MUL(y4,h3), _SSE_MUL(x4,h2)))));
   t5 = _SSE_SUB(_SSE_MUL(t5,tau6), _SSE_ADD( _SSE_MUL(v5,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w5,h5), _SSE_MUL(z5,h4)), _SSE_ADD(_SSE_MUL(y5,h3), _SSE_MUL(x5,h2)))));
   t6 = _SSE_SUB(_SSE_MUL(t6,tau6), _SSE_ADD( _SSE_MUL(v6,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w6,h5), _SSE_MUL(z6,h4)), _SSE_ADD(_SSE_MUL(y6,h3), _SSE_MUL(x6,h2)))));

   /////////////////////////////////////////////////////
   // Rank-1 update of Q [4 x nb+3]
   /////////////////////////////////////////////////////
#endif /* BLOCK6 */
   q1 = _SSE_LOAD(&q[0]);
#ifdef BLOCK2
   q1 = _SSE_ADD(q1, y1);
#endif
#ifdef BLOCK4
   q1 = _SSE_SUB(q1, w1);
#endif
#ifdef BLOCK6
   q1 = _SSE_SUB(q1, t1); 
#endif
   _SSE_STORE(&q[0],q1);
   q2 = _SSE_LOAD(&q[offset]);
#ifdef BLOCK2
   q2 = _SSE_ADD(q2, y2);
#endif
#ifdef BLOCK4
   q2 = _SSE_SUB(q2, w2);
#endif
#ifdef BLOCK6
   q2 = _SSE_SUB(q2, t2);
#endif
   _SSE_STORE(&q[offset],q2);
   q3 = _SSE_LOAD(&q[2*offset]);
#ifdef BLOCK2
   q3 = _SSE_ADD(q3, y3);
#endif
#ifdef BLOCK4
   q3 = _SSE_SUB(q3, w3);
#endif
#ifdef BLOCK6
   q3 = _SSE_SUB(q3, t3);
#endif
   _SSE_STORE(&q[2*offset],q3);
   q4 = _SSE_LOAD(&q[3*offset]);
#ifdef BLOCK2
   q4 = _SSE_ADD(q4, y4);
#endif
#ifdef BLOCK4
   q4 = _SSE_SUB(q4, w4);
#endif
#ifdef BLOCK6
   q4 = _SSE_SUB(q4, t4);
#endif
   _SSE_STORE(&q[3*offset],q4);
   q5 = _SSE_LOAD(&q[4*offset]);
#ifdef BLOCK2
   q5 = _SSE_ADD(q5, y5);
#endif
#ifdef BLOCK4
   q5 = _SSE_SUB(q5, w5);
#endif
#ifdef BLOCK6
   q5 = _SSE_SUB(q5, t5);
#endif
   _SSE_STORE(&q[4*offset],q5);
   q6 = _SSE_LOAD(&q[5*offset]);
#ifdef BLOCK2
   q6 = _SSE_ADD(q6, y6);
#endif
#ifdef BLOCK4
   q6 = _SSE_SUB(q6, w6);
#endif
#ifdef BLOCK6
   q6 = _SSE_SUB(q6, t6);
#endif
   _SSE_STORE(&q[5*offset],q6);

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
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
   q4 = _SSE_LOAD(&q[ldq+3*offset]);
   q4 = _SSE_ADD(q4, _SSE_ADD(x4, _SSE_MUL(y4, h2)));
   _SSE_STORE(&q[ldq+3*offset],q4);
   q5 = _SSE_LOAD(&q[ldq+4*offset]);
   q5 = _SSE_ADD(q5, _SSE_ADD(x5, _SSE_MUL(y5, h2)));
   _SSE_STORE(&q[ldq+4*offset],q5);
   q6 = _SSE_LOAD(&q[ldq+5*offset]);
   q6 = _SSE_ADD(q6, _SSE_ADD(x6, _SSE_MUL(y6, h2)));
   _SSE_STORE(&q[ldq+5*offset],q6);
#endif /* BLOCK2 */

#ifdef BLOCK4
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

   q1 = _SSE_LOAD(&q[ldq]);
   q2 = _SSE_LOAD(&q[ldq+offset]);
   q3 = _SSE_LOAD(&q[ldq+2*offset]);
   q4 = _SSE_LOAD(&q[ldq+3*offset]);
   q5 = _SSE_LOAD(&q[ldq+4*offset]);
   q6 = _SSE_LOAD(&q[ldq+5*offset]);

   q1 = _SSE_SUB(q1, _SSE_ADD(z1, _SSE_MUL(w1, h4)));
   q2 = _SSE_SUB(q2, _SSE_ADD(z2, _SSE_MUL(w2, h4)));
   q3 = _SSE_SUB(q3, _SSE_ADD(z3, _SSE_MUL(w3, h4)));
   q4 = _SSE_SUB(q4, _SSE_ADD(z4, _SSE_MUL(w4, h4)));
   q5 = _SSE_SUB(q5, _SSE_ADD(z5, _SSE_MUL(w5, h4)));
   q6 = _SSE_SUB(q6, _SSE_ADD(z6, _SSE_MUL(w6, h4)));

   _SSE_STORE(&q[ldq],q1);
   _SSE_STORE(&q[ldq+offset],q2);
   _SSE_STORE(&q[ldq+2*offset],q3);
   _SSE_STORE(&q[ldq+3*offset],q4);
   _SSE_STORE(&q[ldq+4*offset],q5);
   _SSE_STORE(&q[ldq+5*offset],q6);

#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+2]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

   q1 = _SSE_LOAD(&q[ldq*2]);
   q2 = _SSE_LOAD(&q[(ldq*2)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*2)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*2)+3*offset]);
   q5 = _SSE_LOAD(&q[(ldq*2)+4*offset]);
   q6 = _SSE_LOAD(&q[(ldq*2)+5*offset]);
   q1 = _SSE_SUB(q1, y1);
   q2 = _SSE_SUB(q2, y2);
   q3 = _SSE_SUB(q3, y3);
   q4 = _SSE_SUB(q4, y4);
   q5 = _SSE_SUB(q5, y5);
   q6 = _SSE_SUB(q6, y6);

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
   q5 = _SSE_SUB(q5, _SSE_MUL(w5, h4));
   q6 = _SSE_SUB(q6, _SSE_MUL(w6, h4));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
   q5 = _SSE_SUB(q5, _SSE_MUL(z5, h3));
   q6 = _SSE_SUB(q6, _SSE_MUL(z6, h3));

   _SSE_STORE(&q[ldq*2],q1);
   _SSE_STORE(&q[(ldq*2)+offset],q2);
   _SSE_STORE(&q[(ldq*2)+2*offset],q3);
   _SSE_STORE(&q[(ldq*2)+3*offset],q4);
   _SSE_STORE(&q[(ldq*2)+4*offset],q5);
   _SSE_STORE(&q[(ldq*2)+5*offset],q6);

#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+3]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

   q1 = _SSE_LOAD(&q[ldq*3]);
   q2 = _SSE_LOAD(&q[(ldq*3)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*3)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*3)+3*offset]);
   q5 = _SSE_LOAD(&q[(ldq*3)+4*offset]);
   q6 = _SSE_LOAD(&q[(ldq*3)+5*offset]);

   q1 = _SSE_SUB(q1, x1);
   q2 = _SSE_SUB(q2, x2);
   q3 = _SSE_SUB(q3, x3);
   q4 = _SSE_SUB(q4, x4);
   q5 = _SSE_SUB(q5, x5);
   q6 = _SSE_SUB(q6, x6);

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
   q5 = _SSE_SUB(q5, _SSE_MUL(w5, h4));
   q6 = _SSE_SUB(q6, _SSE_MUL(w6, h4));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));
   q5 = _SSE_SUB(q5, _SSE_MUL(y5, h2));
   q6 = _SSE_SUB(q6, _SSE_MUL(y6, h2));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+2]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
   q5 = _SSE_SUB(q5, _SSE_MUL(z5, h3));
   q6 = _SSE_SUB(q6, _SSE_MUL(z6, h3));
   _SSE_STORE(&q[ldq*3], q1);
   _SSE_STORE(&q[(ldq*3)+offset], q2);
   _SSE_STORE(&q[(ldq*3)+2*offset], q3);
   _SSE_STORE(&q[(ldq*3)+3*offset], q4);
   _SSE_STORE(&q[(ldq*3)+4*offset], q5);
   _SSE_STORE(&q[(ldq*3)+5*offset], q6);

#endif /* BLOCK4 */

#ifdef BLOCK6
   
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
#endif

   q1 = _SSE_LOAD(&q[ldq]);
   q2 = _SSE_LOAD(&q[(ldq+offset)]);
   q3 = _SSE_LOAD(&q[(ldq+2*offset)]);
   q4 = _SSE_LOAD(&q[(ldq+3*offset)]);
   q5 = _SSE_LOAD(&q[(ldq+4*offset)]);
   q6 = _SSE_LOAD(&q[(ldq+5*offset)]);
   q1 = _SSE_SUB(q1, v1);
   q2 = _SSE_SUB(q2, v2);
   q3 = _SSE_SUB(q3, v3);
   q4 = _SSE_SUB(q4, v4);
   q5 = _SSE_SUB(q5, v5);
   q6 = _SSE_SUB(q6, v6);

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
   q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));
   q5 = _SSE_SUB(q5, _SSE_MUL(t5, h6));
   q6 = _SSE_SUB(q6, _SSE_MUL(t6, h6));

   _SSE_STORE(&q[ldq],q1);
   _SSE_STORE(&q[(ldq+offset)],q2);
   _SSE_STORE(&q[(ldq+2*offset)],q3);
   _SSE_STORE(&q[(ldq+3*offset)],q4);
   _SSE_STORE(&q[(ldq+4*offset)],q5);
   _SSE_STORE(&q[(ldq+5*offset)],q6);
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
#endif
   q1 = _SSE_LOAD(&q[ldq*2]);
   q2 = _SSE_LOAD(&q[(ldq*2)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*2)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*2)+3*offset]);
   q5 = _SSE_LOAD(&q[(ldq*2)+4*offset]);
   q6 = _SSE_LOAD(&q[(ldq*2)+5*offset]);
   q1 = _SSE_SUB(q1, w1); 
   q2 = _SSE_SUB(q2, w2);
   q3 = _SSE_SUB(q3, w3);
   q4 = _SSE_SUB(q4, w4);
   q5 = _SSE_SUB(q5, w5);
   q6 = _SSE_SUB(q6, w6);
   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5)); 
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));  
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));  
   q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));  
   q5 = _SSE_SUB(q5, _SSE_MUL(v5, h5));  
   q6 = _SSE_SUB(q6, _SSE_MUL(v6, h5));  
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
   q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));
   q5 = _SSE_SUB(q5, _SSE_MUL(t5, h6));
   q6 = _SSE_SUB(q6, _SSE_MUL(t6, h6));

   _SSE_STORE(&q[ldq*2],q1);
   _SSE_STORE(&q[(ldq*2)+offset],q2);
   _SSE_STORE(&q[(ldq*2)+2*offset],q3);
   _SSE_STORE(&q[(ldq*2)+3*offset],q4);
   _SSE_STORE(&q[(ldq*2)+4*offset],q5);
   _SSE_STORE(&q[(ldq*2)+5*offset],q6);

#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

   q1 = _SSE_LOAD(&q[ldq*3]);
   q2 = _SSE_LOAD(&q[(ldq*3)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*3)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*3)+3*offset]);
   q5 = _SSE_LOAD(&q[(ldq*3)+4*offset]);
   q6 = _SSE_LOAD(&q[(ldq*3)+5*offset]);
   q1 = _SSE_SUB(q1, z1);
   q2 = _SSE_SUB(q2, z2);
   q3 = _SSE_SUB(q3, z3);
   q4 = _SSE_SUB(q4, z4);
   q5 = _SSE_SUB(q5, z5);
   q6 = _SSE_SUB(q6, z6);

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
   q5 = _SSE_SUB(q5, _SSE_MUL(w5, h4));
   q6 = _SSE_SUB(q6, _SSE_MUL(w6, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
   q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));
   q5 = _SSE_SUB(q5, _SSE_MUL(v5, h5));
   q6 = _SSE_SUB(q6, _SSE_MUL(v6, h5));
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
   q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));
   q5 = _SSE_SUB(q5, _SSE_MUL(t5, h6));
   q6 = _SSE_SUB(q6, _SSE_MUL(t6, h6));

   _SSE_STORE(&q[ldq*3],q1);
   _SSE_STORE(&q[(ldq*3)+offset],q2);
   _SSE_STORE(&q[(ldq*3)+2*offset],q3);
   _SSE_STORE(&q[(ldq*3)+3*offset],q4);
   _SSE_STORE(&q[(ldq*3)+4*offset],q5);
   _SSE_STORE(&q[(ldq*3)+5*offset],q6);
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif
   q1 = _SSE_LOAD(&q[ldq*4]);
   q2 = _SSE_LOAD(&q[(ldq*4)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*4)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*4)+3*offset]);
   q5 = _SSE_LOAD(&q[(ldq*4)+4*offset]);
   q6 = _SSE_LOAD(&q[(ldq*4)+5*offset]);
   q1 = _SSE_SUB(q1, y1);
   q2 = _SSE_SUB(q2, y2);
   q3 = _SSE_SUB(q3, y3);
   q4 = _SSE_SUB(q4, y4);
   q5 = _SSE_SUB(q5, y5);
   q6 = _SSE_SUB(q6, y6);

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
   q5 = _SSE_SUB(q5, _SSE_MUL(z5, h3));
   q6 = _SSE_SUB(q6, _SSE_MUL(z6, h3));
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
   q5 = _SSE_SUB(q5, _SSE_MUL(w5, h4));
   q6 = _SSE_SUB(q6, _SSE_MUL(w6, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
   q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));
   q5 = _SSE_SUB(q5, _SSE_MUL(v5, h5));
   q6 = _SSE_SUB(q6, _SSE_MUL(v6, h5));
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+4]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
   q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));
   q5 = _SSE_SUB(q5, _SSE_MUL(t5, h6));
   q6 = _SSE_SUB(q6, _SSE_MUL(t6, h6));

   _SSE_STORE(&q[ldq*4],q1);
   _SSE_STORE(&q[(ldq*4)+offset],q2);
   _SSE_STORE(&q[(ldq*4)+2*offset],q3);
   _SSE_STORE(&q[(ldq*4)+3*offset],q4);
   _SSE_STORE(&q[(ldq*4)+4*offset],q5);
   _SSE_STORE(&q[(ldq*4)+5*offset],q6);
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[(ldh)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[(ldh)+1], hh[(ldh)+1]);
#endif
   q1 = _SSE_LOAD(&q[ldq*5]);
   q2 = _SSE_LOAD(&q[(ldq*5)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*5)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*5)+3*offset]);
   q5 = _SSE_LOAD(&q[(ldq*5)+4*offset]);
   q6 = _SSE_LOAD(&q[(ldq*5)+5*offset]);
   q1 = _SSE_SUB(q1, x1);
   q2 = _SSE_SUB(q2, x2);
   q3 = _SSE_SUB(q3, x3);
   q4 = _SSE_SUB(q4, x4);
   q5 = _SSE_SUB(q5, x5);
   q6 = _SSE_SUB(q6, x6);

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));
   q5 = _SSE_SUB(q5, _SSE_MUL(y5, h2));
   q6 = _SSE_SUB(q6, _SSE_MUL(y6, h2));
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
   q5 = _SSE_SUB(q5, _SSE_MUL(z5, h3));
   q6 = _SSE_SUB(q6, _SSE_MUL(z6, h3));
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
   q5 = _SSE_SUB(q5, _SSE_MUL(w5, h4));
   q6 = _SSE_SUB(q6, _SSE_MUL(w6, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+4]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
   q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));
   q5 = _SSE_SUB(q5, _SSE_MUL(v5, h5));
   q6 = _SSE_SUB(q6, _SSE_MUL(v6, h5));
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+5]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
   q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));
   q5 = _SSE_SUB(q5, _SSE_MUL(t5, h6));
   q6 = _SSE_SUB(q6, _SSE_MUL(t6, h6));

   _SSE_STORE(&q[ldq*5],q1);
   _SSE_STORE(&q[(ldq*5)+offset],q2);
   _SSE_STORE(&q[(ldq*5)+2*offset],q3);
   _SSE_STORE(&q[(ldq*5)+3*offset],q4);
   _SSE_STORE(&q[(ldq*5)+4*offset],q5);
   _SSE_STORE(&q[(ldq*5)+5*offset],q6);

#endif /* BLOCK6 */

   for (i = BLOCK; i < nb; i++)
   {
#ifdef HAVE_SSE_INTRINSICS
     h1 = _SSE_SET1(hh[i-(BLOCK-1)]);
     h2 = _SSE_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#ifdef HAVE_SPARC64_SSE
     h1 = _SSE_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
     h2 = _SSE_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif

     q1 = _SSE_LOAD(&q[i*ldq]);
     q2 = _SSE_LOAD(&q[(i*ldq)+offset]);
     q3 = _SSE_LOAD(&q[(i*ldq)+2*offset]);
     q4 = _SSE_LOAD(&q[(i*ldq)+3*offset]);
     q5 = _SSE_LOAD(&q[(i*ldq)+4*offset]);
     q6 = _SSE_LOAD(&q[(i*ldq)+5*offset]);

#ifdef BLOCK2
     q1 = _SSE_ADD(q1, _SSE_ADD(_SSE_MUL(x1,h1), _SSE_MUL(y1, h2)));
     q2 = _SSE_ADD(q2, _SSE_ADD(_SSE_MUL(x2,h1), _SSE_MUL(y2, h2)));
     q3 = _SSE_ADD(q3, _SSE_ADD(_SSE_MUL(x3,h1), _SSE_MUL(y3, h2)));
     q4 = _SSE_ADD(q4, _SSE_ADD(_SSE_MUL(x4,h1), _SSE_MUL(y4, h2)));
     q5 = _SSE_ADD(q5, _SSE_ADD(_SSE_MUL(x5,h1), _SSE_MUL(y5, h2)));
     q6 = _SSE_ADD(q6, _SSE_ADD(_SSE_MUL(x6,h1), _SSE_MUL(y6, h2)));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
     
     q1 = _SSE_SUB(q1, _SSE_MUL(x1,h1));
     q2 = _SSE_SUB(q2, _SSE_MUL(x2,h1));
     q3 = _SSE_SUB(q3, _SSE_MUL(x3,h1));
     q4 = _SSE_SUB(q4, _SSE_MUL(x4,h1));
     q5 = _SSE_SUB(q5, _SSE_MUL(x5,h1));
     q6 = _SSE_SUB(q6, _SSE_MUL(x6,h1));

     q1 = _SSE_SUB(q1, _SSE_MUL(y1,h2));
     q2 = _SSE_SUB(q2, _SSE_MUL(y2,h2));
     q3 = _SSE_SUB(q3, _SSE_MUL(y3,h2));
     q4 = _SSE_SUB(q4, _SSE_MUL(y4,h2));
     q5 = _SSE_SUB(q5, _SSE_MUL(y5,h2));
     q6 = _SSE_SUB(q6, _SSE_MUL(y6,h2));

#ifdef HAVE_SSE_INTRINSICS
     h3 = _SSE_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
     h3 = _SSE_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(z1,h3));
     q2 = _SSE_SUB(q2, _SSE_MUL(z2,h3));
     q3 = _SSE_SUB(q3, _SSE_MUL(z3,h3));
     q4 = _SSE_SUB(q4, _SSE_MUL(z4,h3));
     q5 = _SSE_SUB(q5, _SSE_MUL(z5,h3));
     q6 = _SSE_SUB(q6, _SSE_MUL(z6,h3));

#ifdef HAVE_SSE_INTRINSICS
     h4 = _SSE_SET1(hh[(ldh*3)+i-(BLOCK-4)]); 
#endif

#ifdef HAVE_SPRC64_SSE
     h4 = _SSE_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(w1,h4));
     q2 = _SSE_SUB(q2, _SSE_MUL(w2,h4));
     q3 = _SSE_SUB(q3, _SSE_MUL(w3,h4));
     q4 = _SSE_SUB(q4, _SSE_MUL(w4,h4));
     q5 = _SSE_SUB(q5, _SSE_MUL(w5,h4));
     q6 = _SSE_SUB(q6, _SSE_MUL(w6,h4));

#endif /* BLOCK4 || BLOCK6  */


#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
     h5 = _SSE_SET1(hh[(ldh*4)+i-1]);
#endif
#ifdef HAVE_SPARC64_SSE
     h5 = _SSE_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
     q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
     q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
     q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));
     q5 = _SSE_SUB(q5, _SSE_MUL(v5, h5));
     q6 = _SSE_SUB(q6, _SSE_MUL(v6, h5));
#ifdef HAVE_SSE_INTRINSICS
     h6 = _SSE_SET1(hh[(ldh*5)+i]);
#endif
#ifdef HAVE_SPARC64_SSE
     h6 = _SSE_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
     q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
     q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
     q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));
     q5 = _SSE_SUB(q5, _SSE_MUL(t5, h6));
     q6 = _SSE_SUB(q6, _SSE_MUL(t6, h6));

#endif /* BLOCK6 */
     _SSE_STORE(&q[i*ldq],q1);
     _SSE_STORE(&q[(i*ldq)+offset],q2);
     _SSE_STORE(&q[(i*ldq)+2*offset],q3);
     _SSE_STORE(&q[(i*ldq)+3*offset],q4);
     _SSE_STORE(&q[(i*ldq)+4*offset],q5);
     _SSE_STORE(&q[(i*ldq)+5*offset],q6);

   }
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-1)]);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif

   q1 = _SSE_LOAD(&q[nb*ldq]);
   q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);
   q3 = _SSE_LOAD(&q[(nb*ldq)+2*offset]);
   q4 = _SSE_LOAD(&q[(nb*ldq)+3*offset]);
   q5 = _SSE_LOAD(&q[(nb*ldq)+4*offset]);
   q6 = _SSE_LOAD(&q[(nb*ldq)+5*offset]);

#ifdef BLOCK2
   q1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_ADD(q4, _SSE_MUL(x4, h1));
   q5 = _SSE_ADD(q5, _SSE_MUL(x5, h1));
   q6 = _SSE_ADD(q6, _SSE_MUL(x6, h1));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_SUB(q4, _SSE_MUL(x4, h1));
   q5 = _SSE_SUB(q5, _SSE_MUL(x5, h1));
   q6 = _SSE_SUB(q6, _SSE_MUL(x6, h1));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));
   q5 = _SSE_SUB(q5, _SSE_MUL(y5, h2));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
   q5 = _SSE_SUB(q5, _SSE_MUL(z5, h3));
   q6 = _SSE_SUB(q6, _SSE_MUL(z6, h3));

#endif /* BLOCK4 || BLOCK6  */

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
   q5 = _SSE_SUB(q5, _SSE_MUL(w5, h4));
   q6 = _SSE_SUB(q6, _SSE_MUL(w6, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
   q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));
   q5 = _SSE_SUB(q5, _SSE_MUL(v5, h5));
   q6 = _SSE_SUB(q6, _SSE_MUL(v6, h5));
#endif /* BLOCK6 */

   _SSE_STORE(&q[nb*ldq],q1);
   _SSE_STORE(&q[(nb*ldq)+offset],q2);
   _SSE_STORE(&q[(nb*ldq)+2*offset],q3);
   _SSE_STORE(&q[(nb*ldq)+3*offset],q4);
   _SSE_STORE(&q[(nb*ldq)+4*offset],q5);
   _SSE_STORE(&q[(nb*ldq)+5*offset],q6);

#if defined(BLOCK4) || defined(BLOCK6)
   
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-2)], hh[nb-(BLOCK-2)]);
#endif

   q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);
   q3 = _SSE_LOAD(&q[((nb+1)*ldq)+2*offset]);
   q4 = _SSE_LOAD(&q[((nb+1)*ldq)+3*offset]);
   q5 = _SSE_LOAD(&q[((nb+1)*ldq)+4*offset]);
   q6 = _SSE_LOAD(&q[((nb+1)*ldq)+5*offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_SUB(q4, _SSE_MUL(x4, h1));
   q5 = _SSE_SUB(q5, _SSE_MUL(x5, h1));
   q6 = _SSE_SUB(q6, _SSE_MUL(x6, h1));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-(BLOCK-3)], hh[ldh+nb-(BLOCK-3)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));
   q5 = _SSE_SUB(q5, _SSE_MUL(y5, h2));
   q6 = _SSE_SUB(q6, _SSE_MUL(y6, h2));

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
   q5 = _SSE_SUB(q5, _SSE_MUL(z5, h3));
   q6 = _SSE_SUB(q6, _SSE_MUL(z6, h3));
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
   q5 = _SSE_SUB(q5, _SSE_MUL(w5, h4));
   q6 = _SSE_SUB(q6, _SSE_MUL(w6, h4));

#endif /* BLOCK6 */

   _SSE_STORE(&q[(nb+1)*ldq],q1);
   _SSE_STORE(&q[((nb+1)*ldq)+offset],q2);
   _SSE_STORE(&q[((nb+1)*ldq)+2*offset],q3);
   _SSE_STORE(&q[((nb+1)*ldq)+3*offset],q4);
   _SSE_STORE(&q[((nb+1)*ldq)+4*offset],q5);
   _SSE_STORE(&q[((nb+1)*ldq)+5*offset],q6);

#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-3)], hh[nb-(BLOCK-3)]);
#endif

   q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);
   q3 = _SSE_LOAD(&q[((nb+2)*ldq)+2*offset]);
   q4 = _SSE_LOAD(&q[((nb+2)*ldq)+3*offset]);
   q5 = _SSE_LOAD(&q[((nb+2)*ldq)+4*offset]);
   q6 = _SSE_LOAD(&q[((nb+2)*ldq)+5*offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_SUB(q4, _SSE_MUL(x4, h1));
   q5 = _SSE_SUB(q5, _SSE_MUL(x5, h1));
   q6 = _SSE_SUB(q6, _SSE_MUL(x6, h1));

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));
   q5 = _SSE_SUB(q5, _SSE_MUL(y5, h2));
   q6 = _SSE_SUB(q6, _SSE_MUL(y6, h2));
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
   q5 = _SSE_SUB(q5, _SSE_MUL(z5, h3));
   q6 = _SSE_SUB(q6, _SSE_MUL(z6, h3));
#endif /* BLOCK6 */

   _SSE_STORE(&q[(nb+2)*ldq],q1);
   _SSE_STORE(&q[((nb+2)*ldq)+offset],q2);
   _SSE_STORE(&q[((nb+2)*ldq)+2*offset],q3);
   _SSE_STORE(&q[((nb+2)*ldq)+3*offset],q4);
   _SSE_STORE(&q[((nb+2)*ldq)+4*offset],q5);
   _SSE_STORE(&q[((nb+2)*ldq)+5*offset],q6);

#endif /* BLOCK4 || BLOCK6  */

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

   q1 = _SSE_LOAD(&q[(nb+3)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+3)*ldq)+offset]);
   q3 = _SSE_LOAD(&q[((nb+3)*ldq)+2*offset]);
   q4 = _SSE_LOAD(&q[((nb+3)*ldq)+3*offset]);
   q5 = _SSE_LOAD(&q[((nb+3)*ldq)+4*offset]);
   q6 = _SSE_LOAD(&q[((nb+3)*ldq)+5*offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_SUB(q4, _SSE_MUL(x4, h1));
   q5 = _SSE_SUB(q5, _SSE_MUL(x5, h1));
   q6 = _SSE_SUB(q6, _SSE_MUL(x6, h1));
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));
   q5 = _SSE_SUB(q5, _SSE_MUL(y5, h2));
   q6 = _SSE_SUB(q6, _SSE_MUL(y6, h2));

   _SSE_STORE(&q[(nb+3)*ldq],q1);
   _SSE_STORE(&q[((nb+3)*ldq)+offset],q2);
   _SSE_STORE(&q[((nb+3)*ldq)+2*offset],q3);
   _SSE_STORE(&q[((nb+3)*ldq)+3*offset],q4);
   _SSE_STORE(&q[((nb+3)*ldq)+4*offset],q5);
   _SSE_STORE(&q[((nb+3)*ldq)+5*offset],q6);
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif

   q1 = _SSE_LOAD(&q[(nb+4)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+4)*ldq)+offset]);
   q3 = _SSE_LOAD(&q[((nb+4)*ldq)+2*offset]);
   q4 = _SSE_LOAD(&q[((nb+4)*ldq)+3*offset]);
   q5 = _SSE_LOAD(&q[((nb+4)*ldq)+4*offset]);
   q6 = _SSE_LOAD(&q[((nb+4)*ldq)+5*offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_SUB(q4, _SSE_MUL(x4, h1));
   q5 = _SSE_SUB(q5, _SSE_MUL(x5, h1));
   q6 = _SSE_SUB(q6, _SSE_MUL(x6, h1));

   _SSE_STORE(&q[(nb+4)*ldq],q1);
   _SSE_STORE(&q[((nb+4)*ldq)+offset],q2);
   _SSE_STORE(&q[((nb+4)*ldq)+2*offset],q3);
   _SSE_STORE(&q[((nb+4)*ldq)+3*offset],q4);
   _SSE_STORE(&q[((nb+4)*ldq)+4*offset],q5);
   _SSE_STORE(&q[((nb+4)*ldq)+5*offset],q6);

#endif /* BLOCK6 */
}


/*
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 10 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 20 rows of Q simultaneously, a
#endif
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
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 10
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 20
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
#ifdef HAVE_SSE_INTRINSICS
    // Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
    __SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SSE_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif
    __SSE_DATATYPE x1 = _SSE_LOAD(&q[ldq]);
    __SSE_DATATYPE x2 = _SSE_LOAD(&q[ldq+offset]);
    __SSE_DATATYPE x3 = _SSE_LOAD(&q[ldq+2*offset]);
    __SSE_DATATYPE x4 = _SSE_LOAD(&q[ldq+3*offset]);
    __SSE_DATATYPE x5 = _SSE_LOAD(&q[ldq+4*offset]);

#ifdef HAVE_SSE_INTRINSICS
    __SSE_DATATYPE h1 = _SSE_SET1(hh[ldh+1]);
#endif
#ifdef HAVE_SPARC64_SSE
    __SSE_DATATYPE h1 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
#endif
    __SSE_DATATYPE h2;

    __SSE_DATATYPE q1 = _SSE_LOAD(q);
    __SSE_DATATYPE y1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
    __SSE_DATATYPE q2 = _SSE_LOAD(&q[offset]);
    __SSE_DATATYPE y2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
    __SSE_DATATYPE q3 = _SSE_LOAD(&q[2*offset]);
    __SSE_DATATYPE y3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
    __SSE_DATATYPE q4 = _SSE_LOAD(&q[3*offset]);
    __SSE_DATATYPE y4 = _SSE_ADD(q4, _SSE_MUL(x4, h1));
    __SSE_DATATYPE q5 = _SSE_LOAD(&q[4*offset]);
    __SSE_DATATYPE y5 = _SSE_ADD(q5, _SSE_MUL(x5, h1));
#endif /* BLOCK2 */

#ifdef BLOCK4
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

    register __SSE_DATATYPE w1 = _SSE_ADD(a4_1, _SSE_MUL(a3_1, h_4_3));
    w1 = _SSE_ADD(w1, _SSE_MUL(a2_1, h_4_2));                          
    w1 = _SSE_ADD(w1, _SSE_MUL(a1_1, h_4_1));                          
    register __SSE_DATATYPE z1 = _SSE_ADD(a3_1, _SSE_MUL(a2_1, h_3_2));
    z1 = _SSE_ADD(z1, _SSE_MUL(a1_1, h_3_1));                          
    register __SSE_DATATYPE y1 = _SSE_ADD(a2_1, _SSE_MUL(a1_1, h_2_1));
    register __SSE_DATATYPE x1 = a1_1; 

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

    __SSE_DATATYPE a1_4 = _SSE_LOAD(&q[(ldq*3)+3*offset]);
    __SSE_DATATYPE a2_4 = _SSE_LOAD(&q[(ldq*2)+3*offset]);
    __SSE_DATATYPE a3_4 = _SSE_LOAD(&q[ldq+3*offset]);
    __SSE_DATATYPE a4_4 = _SSE_LOAD(&q[0+3*offset]);

    register __SSE_DATATYPE w4 = _SSE_ADD(a4_4, _SSE_MUL(a3_4, h_4_3));
    w4 = _SSE_ADD(w4, _SSE_MUL(a2_4, h_4_2));
    w4 = _SSE_ADD(w4, _SSE_MUL(a1_4, h_4_1));
    register __SSE_DATATYPE z4 = _SSE_ADD(a3_4, _SSE_MUL(a2_4, h_3_2));
    z4 = _SSE_ADD(z4, _SSE_MUL(a1_4, h_3_1));
    register __SSE_DATATYPE y4 = _SSE_ADD(a2_4, _SSE_MUL(a1_4, h_2_1));
    register __SSE_DATATYPE x4 = a1_4;

    __SSE_DATATYPE a1_5 = _SSE_LOAD(&q[(ldq*3)+4*offset]);
    __SSE_DATATYPE a2_5 = _SSE_LOAD(&q[(ldq*2)+4*offset]);
    __SSE_DATATYPE a3_5 = _SSE_LOAD(&q[ldq+4*offset]);
    __SSE_DATATYPE a4_5 = _SSE_LOAD(&q[0+4*offset]);

    register __SSE_DATATYPE w5 = _SSE_ADD(a4_5, _SSE_MUL(a3_5, h_4_3));
    w5 = _SSE_ADD(w5, _SSE_MUL(a2_5, h_4_2));
    w5 = _SSE_ADD(w5, _SSE_MUL(a1_5, h_4_1));
    register __SSE_DATATYPE z5 = _SSE_ADD(a3_5, _SSE_MUL(a2_5, h_3_2));
    z5 = _SSE_ADD(z5, _SSE_MUL(a1_5, h_3_1));
    register __SSE_DATATYPE y5 = _SSE_ADD(a2_5, _SSE_MUL(a1_5, h_2_1));
    register __SSE_DATATYPE x5 = a1_5;

    __SSE_DATATYPE q1;
    __SSE_DATATYPE q2;
    __SSE_DATATYPE q3;
    __SSE_DATATYPE q4;
    __SSE_DATATYPE q5;

    __SSE_DATATYPE h1;
    __SSE_DATATYPE h2;
    __SSE_DATATYPE h3;
    __SSE_DATATYPE h4;
#endif /* BLOCK4 */

#ifdef BLOCK6
    
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

    __SSE_DATATYPE a1_3 = _SSE_LOAD(&q[(ldq*5)+2*offset]);
    __SSE_DATATYPE a2_3 = _SSE_LOAD(&q[(ldq*4)+2*offset]);
    __SSE_DATATYPE a3_3 = _SSE_LOAD(&q[(ldq*3)+2*offset]);
    __SSE_DATATYPE a4_3 = _SSE_LOAD(&q[(ldq*2)+2*offset]);
    __SSE_DATATYPE a5_3 = _SSE_LOAD(&q[(ldq)+2*offset]);
    __SSE_DATATYPE a6_3 = _SSE_LOAD(&q[2*offset]);

    register __SSE_DATATYPE t3 = _SSE_ADD(a6_3, _SSE_MUL(a5_3, h_6_5));
    t3 = _SSE_ADD(t3, _SSE_MUL(a4_3, h_6_4));
    t3 = _SSE_ADD(t3, _SSE_MUL(a3_3, h_6_3));
    t3 = _SSE_ADD(t3, _SSE_MUL(a2_3, h_6_2));
    t3 = _SSE_ADD(t3, _SSE_MUL(a1_3, h_6_1));
    register __SSE_DATATYPE v3 = _SSE_ADD(a5_3, _SSE_MUL(a4_3, h_5_4));
    v3 = _SSE_ADD(v3, _SSE_MUL(a3_3, h_5_3));
    v3 = _SSE_ADD(v3, _SSE_MUL(a2_3, h_5_2));
    v3 = _SSE_ADD(v3, _SSE_MUL(a1_3, h_5_1));
    register __SSE_DATATYPE w3 = _SSE_ADD(a4_3, _SSE_MUL(a3_3, h_4_3));
    w3 = _SSE_ADD(w3, _SSE_MUL(a2_3, h_4_2));
    w3 = _SSE_ADD(w3, _SSE_MUL(a1_3, h_4_1));
    register __SSE_DATATYPE z3 = _SSE_ADD(a3_3, _SSE_MUL(a2_3, h_3_2));
    z3 = _SSE_ADD(z3, _SSE_MUL(a1_3, h_3_1));
    register __SSE_DATATYPE y3 = _SSE_ADD(a2_3, _SSE_MUL(a1_3, h_2_1));

    register __SSE_DATATYPE x3 = a1_3;

    __SSE_DATATYPE a1_4 = _SSE_LOAD(&q[(ldq*5)+3*offset]);
    __SSE_DATATYPE a2_4 = _SSE_LOAD(&q[(ldq*4)+3*offset]);
    __SSE_DATATYPE a3_4 = _SSE_LOAD(&q[(ldq*3)+3*offset]);
    __SSE_DATATYPE a4_4 = _SSE_LOAD(&q[(ldq*2)+3*offset]);
    __SSE_DATATYPE a5_4 = _SSE_LOAD(&q[(ldq)+3*offset]);
    __SSE_DATATYPE a6_4 = _SSE_LOAD(&q[3*offset]);

    register __SSE_DATATYPE t4 = _SSE_ADD(a6_4, _SSE_MUL(a5_4, h_6_5));
    t4 = _SSE_ADD(t4, _SSE_MUL(a4_4, h_6_4));
    t4 = _SSE_ADD(t4, _SSE_MUL(a3_4, h_6_3));
    t4 = _SSE_ADD(t4, _SSE_MUL(a2_4, h_6_2));
    t4 = _SSE_ADD(t4, _SSE_MUL(a1_4, h_6_1));
    register __SSE_DATATYPE v4 = _SSE_ADD(a5_4, _SSE_MUL(a4_4, h_5_4));
    v4 = _SSE_ADD(v4, _SSE_MUL(a3_4, h_5_3));
    v4 = _SSE_ADD(v4, _SSE_MUL(a2_4, h_5_2));
    v4 = _SSE_ADD(v4, _SSE_MUL(a1_4, h_5_1));
    register __SSE_DATATYPE w4 = _SSE_ADD(a4_4, _SSE_MUL(a3_4, h_4_3));
    w4 = _SSE_ADD(w4, _SSE_MUL(a2_4, h_4_2));
    w4 = _SSE_ADD(w4, _SSE_MUL(a1_4, h_4_1));
    register __SSE_DATATYPE z4 = _SSE_ADD(a3_4, _SSE_MUL(a2_4, h_3_2));
    z4 = _SSE_ADD(z4, _SSE_MUL(a1_4, h_3_1));
    register __SSE_DATATYPE y4 = _SSE_ADD(a2_4, _SSE_MUL(a1_4, h_2_1));

    register __SSE_DATATYPE x4 = a1_4;

    __SSE_DATATYPE a1_5 = _SSE_LOAD(&q[(ldq*5)+4*offset]);
    __SSE_DATATYPE a2_5 = _SSE_LOAD(&q[(ldq*4)+4*offset]);
    __SSE_DATATYPE a3_5 = _SSE_LOAD(&q[(ldq*3)+4*offset]);
    __SSE_DATATYPE a4_5 = _SSE_LOAD(&q[(ldq*2)+4*offset]);
    __SSE_DATATYPE a5_5 = _SSE_LOAD(&q[(ldq)+4*offset]);
    __SSE_DATATYPE a6_5 = _SSE_LOAD(&q[4*offset]);

    register __SSE_DATATYPE t5 = _SSE_ADD(a6_5, _SSE_MUL(a5_5, h_6_5));
    t5 = _SSE_ADD(t5, _SSE_MUL(a4_5, h_6_4));
    t5 = _SSE_ADD(t5, _SSE_MUL(a3_5, h_6_3));
    t5 = _SSE_ADD(t5, _SSE_MUL(a2_5, h_6_2));
    t5 = _SSE_ADD(t5, _SSE_MUL(a1_5, h_6_1));
    register __SSE_DATATYPE v5 = _SSE_ADD(a5_5, _SSE_MUL(a4_5, h_5_4));
    v5 = _SSE_ADD(v5, _SSE_MUL(a3_5, h_5_3));
    v5 = _SSE_ADD(v5, _SSE_MUL(a2_5, h_5_2));
    v5 = _SSE_ADD(v5, _SSE_MUL(a1_5, h_5_1));
    register __SSE_DATATYPE w5 = _SSE_ADD(a4_5, _SSE_MUL(a3_5, h_4_3));
    w5 = _SSE_ADD(w5, _SSE_MUL(a2_5, h_4_2));
    w5 = _SSE_ADD(w5, _SSE_MUL(a1_5, h_4_1));
    register __SSE_DATATYPE z5 = _SSE_ADD(a3_5, _SSE_MUL(a2_5, h_3_2));
    z5 = _SSE_ADD(z5, _SSE_MUL(a1_5, h_3_1));
    register __SSE_DATATYPE y5 = _SSE_ADD(a2_5, _SSE_MUL(a1_5, h_2_1));

    register __SSE_DATATYPE x5 = a1_5;


    __SSE_DATATYPE q1;
    __SSE_DATATYPE q2;
    __SSE_DATATYPE q3;
    __SSE_DATATYPE q4;
    __SSE_DATATYPE q5;

    __SSE_DATATYPE h1;
    __SSE_DATATYPE h2;
    __SSE_DATATYPE h3;
    __SSE_DATATYPE h4;
    __SSE_DATATYPE h5;
    __SSE_DATATYPE h6;

#endif /* BLOCK6 */


    for(i = BLOCK; i < nb; i++)
      {
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[i-(BLOCK-1)]);
        h2 = _SSE_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
        h2 = _SSE_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
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
        q4 = _SSE_LOAD(&q[(i*ldq)+3*offset]);
        x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
        y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
        q5 = _SSE_LOAD(&q[(i*ldq)+4*offset]);
        x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));
        y5 = _SSE_ADD(y5, _SSE_MUL(q5,h2));

#if defined(BLOCK4) || defined(BLOCK6)
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

        z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
        z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
        z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));
        z4 = _SSE_ADD(z4, _SSE_MUL(q4,h3));
        z5 = _SSE_ADD(z5, _SSE_MUL(q5,h3));
#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

        w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
        w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));
        w3 = _SSE_ADD(w3, _SSE_MUL(q3,h4));
        w4 = _SSE_ADD(w4, _SSE_MUL(q4,h4));
        w5 = _SSE_ADD(w5, _SSE_MUL(q5,h4));
	
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+i-1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif
        v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));
        v2 = _SSE_ADD(v2, _SSE_MUL(q2,h5));
        v3 = _SSE_ADD(v3, _SSE_MUL(q3,h5));
        v4 = _SSE_ADD(v4, _SSE_MUL(q4,h5));
        v5 = _SSE_ADD(v5, _SSE_MUL(q5,h5));

#ifdef HAVE_SSE_INTRINSICS
        h6 = _SSE_SET1(hh[(ldh*5)+i]);
#endif

#ifdef HAVE_SPARC64_SSE
        h6 = _SSE_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif

        t1 = _SSE_ADD(t1, _SSE_MUL(q1,h6));
        t2 = _SSE_ADD(t2, _SSE_MUL(q2,h6));
        t3 = _SSE_ADD(t3, _SSE_MUL(q3,h6));
        t4 = _SSE_ADD(t4, _SSE_MUL(q4,h6));
        t5 = _SSE_ADD(t5, _SSE_MUL(q5,h6));
	
#endif /* BLOCK6 */
      }
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-(BLOCK-1)]);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif 

    q1 = _SSE_LOAD(&q[nb*ldq]);
    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    q3 = _SSE_LOAD(&q[(nb*ldq)+2*offset]);
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    q4 = _SSE_LOAD(&q[(nb*ldq)+3*offset]);
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
    q5 = _SSE_LOAD(&q[(nb*ldq)+4*offset]);
    x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));

#if defined(BLOCK4) || defined(BLOCK6)
    
#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
    y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
    y5 = _SSE_ADD(y5, _SSE_MUL(q5,h2));

#ifdef HAVE_SSE_INTRINSICS
    h3 = _SSE_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
    h3 = _SSE_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

    z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
    z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
    z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));
    z4 = _SSE_ADD(z4, _SSE_MUL(q4,h3));
    z5 = _SSE_ADD(z5, _SSE_MUL(q5,h3));

#ifdef BLOCK4
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

    q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+1)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+1)*ldq)+3*offset]);
    q5 = _SSE_LOAD(&q[((nb+1)*ldq)+4*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
    x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));

#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[(ldh*1)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[(ldh*1)+nb-1], hh[(ldh*1)+nb-1]);
#endif


    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
    y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
    y5 = _SSE_ADD(y5, _SSE_MUL(q5,h2));

#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif


    q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+2)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+2)*ldq)+3*offset]);
    q5 = _SSE_LOAD(&q[((nb+2)*ldq)+4*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
    x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));

#endif /* BLOCK4 */
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#ifdef HAVE_SSE_INTRINSICS
    h4 = _SSE_SET1(hh[(ldh*3)+nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h4 = _SSE_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif

    w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4)); 
    w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));
    w3 = _SSE_ADD(w3, _SSE_MUL(q3,h4));
    w4 = _SSE_ADD(w4, _SSE_MUL(q4,h4));
    w5 = _SSE_ADD(w5, _SSE_MUL(q5,h4));

#ifdef HAVE_SSE_INTRINSICS
    h5 = _SSE_SET1(hh[(ldh*4)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h5 = _SSE_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif

    v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));
    v2 = _SSE_ADD(v2, _SSE_MUL(q2,h5));
    v3 = _SSE_ADD(v3, _SSE_MUL(q3,h5));
    v4 = _SSE_ADD(v4, _SSE_MUL(q4,h5));
    v5 = _SSE_ADD(v5, _SSE_MUL(q5,h5));
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-4]);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-4], hh[nb-4]);
#endif

    q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+1)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+1)*ldq)+3*offset]);
    q5 = _SSE_LOAD(&q[((nb+1)*ldq)+4*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
    x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));
#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-3]);
#endif
#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-3], hh[ldh+nb-3]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
    y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
    y5 = _SSE_ADD(y5, _SSE_MUL(q5,h2));

#ifdef HAVE_SSE_INTRINSICS
    h3 = _SSE_SET1(hh[(ldh*2)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
    h3 = _SSE_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif

    z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
    z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
    z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));
    z4 = _SSE_ADD(z4, _SSE_MUL(q4,h3));
    z5 = _SSE_ADD(z5, _SSE_MUL(q5,h3));

#ifdef HAVE_SSE_INTRINSICS
    h4 = _SSE_SET1(hh[(ldh*3)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
    h4 = _SSE_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif

    w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
    w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));
    w3 = _SSE_ADD(w3, _SSE_MUL(q3,h4));
    w4 = _SSE_ADD(w4, _SSE_MUL(q4,h4));
    w5 = _SSE_ADD(w5, _SSE_MUL(q5,h4));

#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-3]);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-3], hh[nb-3]);
#endif

    q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+2)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+2)*ldq)+3*offset]);
    q5 = _SSE_LOAD(&q[((nb+2)*ldq)+4*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
    x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));

#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
    y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
    y5 = _SSE_ADD(y5, _SSE_MUL(q5,h2));
#ifdef HAVE_SSE_INTRINSICS
    h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif

    z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
    z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
    z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));
    z4 = _SSE_ADD(z4, _SSE_MUL(q4,h3));
    z5 = _SSE_ADD(z5, _SSE_MUL(q5,h3));
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

    q1 = _SSE_LOAD(&q[(nb+3)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+3)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+3)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+3)*ldq)+3*offset]);
    q5 = _SSE_LOAD(&q[((nb+3)*ldq)+4*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
    x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));

#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
    y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
    y5 = _SSE_ADD(y5, _SSE_MUL(q5,h2));

#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif

    q1 = _SSE_LOAD(&q[(nb+4)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+4)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+4)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+4)*ldq)+3*offset]);
    q5 = _SSE_LOAD(&q[((nb+4)*ldq)+4*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
    x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));
#endif /* BLOCK6 */

#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Rank-2 update of Q [6 x nb+1]
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

#ifdef HAVE_SSE_INTRINSICS
    __SSE_DATATYPE tau1 = _SSE_SET1(hh[0]);

    __SSE_DATATYPE tau2 = _SSE_SET1(hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SSE_DATATYPE tau3 = _SSE_SET1(hh[ldh*2]);
   __SSE_DATATYPE tau4 = _SSE_SET1(hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE tau5 = _SSE_SET1(hh[ldh*4]);
   __SSE_DATATYPE tau6 = _SSE_SET1(hh[ldh*5]);       
#endif

#ifdef BLOCK2    
    __SSE_DATATYPE vs = _SSE_SET1(s);
#endif
#ifdef BLOCK4
   __SSE_DATATYPE vs_1_2 = _SSE_SET1(s_1_2);
   __SSE_DATATYPE vs_1_3 = _SSE_SET1(s_1_3);  
   __SSE_DATATYPE vs_2_3 = _SSE_SET1(s_2_3);
   __SSE_DATATYPE vs_1_4 = _SSE_SET1(s_1_4);  
   __SSE_DATATYPE vs_2_4 = _SSE_SET1(s_2_4);  
   __SSE_DATATYPE vs_3_4 = _SSE_SET1(s_3_4);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE vs_1_2 = _SSE_SET1(scalarprods[0]);
   __SSE_DATATYPE vs_1_3 = _SSE_SET1(scalarprods[1]);
   __SSE_DATATYPE vs_2_3 = _SSE_SET1(scalarprods[2]);
   __SSE_DATATYPE vs_1_4 = _SSE_SET1(scalarprods[3]);
   __SSE_DATATYPE vs_2_4 = _SSE_SET1(scalarprods[4]);
   __SSE_DATATYPE vs_3_4 = _SSE_SET1(scalarprods[5]);
   __SSE_DATATYPE vs_1_5 = _SSE_SET1(scalarprods[6]);
   __SSE_DATATYPE vs_2_5 = _SSE_SET1(scalarprods[7]);
   __SSE_DATATYPE vs_3_5 = _SSE_SET1(scalarprods[8]);
   __SSE_DATATYPE vs_4_5 = _SSE_SET1(scalarprods[9]);
   __SSE_DATATYPE vs_1_6 = _SSE_SET1(scalarprods[10]);
   __SSE_DATATYPE vs_2_6 = _SSE_SET1(scalarprods[11]);
   __SSE_DATATYPE vs_3_6 = _SSE_SET1(scalarprods[12]);
   __SSE_DATATYPE vs_4_6 = _SSE_SET1(scalarprods[13]);
   __SSE_DATATYPE vs_5_6 = _SSE_SET1(scalarprods[14]);
#endif
#endif /* HAVE_SSE_INTRINSICS */

#ifdef HAVE_SPARC64_SSE
    __SSE_DATATYPE tau1 = _SSE_SET(hh[0], hh[0]);

    __SSE_DATATYPE tau2 = _SSE_SET(hh[ldh], hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SSE_DATATYPE tau3 = _SSE_SET(hh[ldh*2], hh[ldh*2]);
   __SSE_DATATYPE tau4 = _SSE_SET(hh[ldh*3], hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE tau5 = _SSE_SET(hh[ldh*4], hh[ldh*4]);
   __SSE_DATATYPE tau6 = _SSE_SET(hh[ldh*5], hh[ldh*5]);
#endif

#ifdef BLOCK2
    __SSE_DATATYPE vs = _SSE_SET(s, s);
#endif
#ifdef BLOCK4
   __SSE_DATATYPE vs_1_2 = _SSE_SET(s_1_2, s_1_2);
   __SSE_DATATYPE vs_1_3 = _SSE_SET(s_1_3, s_1_3);
   __SSE_DATATYPE vs_2_3 = _SSE_SET(s_2_3, s_2_3);
   __SSE_DATATYPE vs_1_4 = _SSE_SET(s_1_4, s_1_4);
   __SSE_DATATYPE vs_2_4 = _SSE_SET(s_2_4, s_2_4);
   __SSE_DATATYPE vs_3_4 = _SSE_SET(s_3_4, s_3_4);

#endif
#ifdef BLOCK6
   __SSE_DATATYPE vs_1_2 = _SSE_SET(scalarprods[0], scalarprods[0]);
   __SSE_DATATYPE vs_1_3 = _SSE_SET(scalarprods[1], scalarprods[1]);
   __SSE_DATATYPE vs_2_3 = _SSE_SET(scalarprods[2], scalarprods[2]);
   __SSE_DATATYPE vs_1_4 = _SSE_SET(scalarprods[3], scalarprods[3]);
   __SSE_DATATYPE vs_2_4 = _SSE_SET(scalarprods[4], scalarprods[4]);
   __SSE_DATATYPE vs_3_4 = _SSE_SET(scalarprods[5], scalarprods[5]);
   __SSE_DATATYPE vs_1_5 = _SSE_SET(scalarprods[6], scalarprods[6]);
   __SSE_DATATYPE vs_2_5 = _SSE_SET(scalarprods[7], scalarprods[7]);
   __SSE_DATATYPE vs_3_5 = _SSE_SET(scalarprods[8], scalarprods[8]);
   __SSE_DATATYPE vs_4_5 = _SSE_SET(scalarprods[9], scalarprods[9]);
   __SSE_DATATYPE vs_1_6 = _SSE_SET(scalarprods[10], scalarprods[10]);
   __SSE_DATATYPE vs_2_6 = _SSE_SET(scalarprods[11], scalarprods[11]);
   __SSE_DATATYPE vs_3_6 = _SSE_SET(scalarprods[12], scalarprods[12]);
   __SSE_DATATYPE vs_4_6 = _SSE_SET(scalarprods[13], scalarprods[13]);
   __SSE_DATATYPE vs_5_6 = _SSE_SET(scalarprods[14], scalarprods[14]);
#endif
#endif /* HAVE_SPARC64_SSE */

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_XOR(tau1, sign);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _fjsp_neg_v2r8(tau1);
#endif
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau1;
#endif

   x1 = _SSE_MUL(x1, h1);
   x2 = _SSE_MUL(x2, h1);
   x3 = _SSE_MUL(x3, h1);
   x4 = _SSE_MUL(x4, h1);
   x5 = _SSE_MUL(x5, h1);

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_XOR(tau2, sign);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _fjsp_neg_v2r8(tau2);
#endif
   h2 = _SSE_MUL(h1, vs);
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau2;
   h2 = _SSE_MUL(h1, vs_1_2);
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK2
   y1 = _SSE_ADD(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
   y2 = _SSE_ADD(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
   y3 = _SSE_ADD(_SSE_MUL(y3,h1), _SSE_MUL(x3,h2));
   y4 = _SSE_ADD(_SSE_MUL(y4,h1), _SSE_MUL(x4,h2));
   y5 = _SSE_ADD(_SSE_MUL(y5,h1), _SSE_MUL(x5,h2));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   y1 = _SSE_SUB(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
   y2 = _SSE_SUB(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
   y3 = _SSE_SUB(_SSE_MUL(y3,h1), _SSE_MUL(x3,h2));
   y4 = _SSE_SUB(_SSE_MUL(y4,h1), _SSE_MUL(x4,h2));
   y5 = _SSE_SUB(_SSE_MUL(y5,h1), _SSE_MUL(x5,h2));

   h1 = tau3;
   h2 = _SSE_MUL(h1, vs_1_3);
   h3 = _SSE_MUL(h1, vs_2_3);

   z1 = _SSE_SUB(_SSE_MUL(z1,h1), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)));
   z2 = _SSE_SUB(_SSE_MUL(z2,h1), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2)));
   z3 = _SSE_SUB(_SSE_MUL(z3,h1), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2)));
   z4 = _SSE_SUB(_SSE_MUL(z4,h1), _SSE_ADD(_SSE_MUL(y4,h3), _SSE_MUL(x4,h2)));
   z5 = _SSE_SUB(_SSE_MUL(z5,h1), _SSE_ADD(_SSE_MUL(y5,h3), _SSE_MUL(x5,h2)));

   h1 = tau4;
   h2 = _SSE_MUL(h1, vs_1_4);
   h3 = _SSE_MUL(h1, vs_2_4);
   h4 = _SSE_MUL(h1, vs_3_4);

   w1 = _SSE_SUB(_SSE_MUL(w1,h1), _SSE_ADD(_SSE_MUL(z1,h4), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));
   w2 = _SSE_SUB(_SSE_MUL(w2,h1), _SSE_ADD(_SSE_MUL(z2,h4), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2))));
   w3 = _SSE_SUB(_SSE_MUL(w3,h1), _SSE_ADD(_SSE_MUL(z3,h4), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2))));
   w4 = _SSE_SUB(_SSE_MUL(w4,h1), _SSE_ADD(_SSE_MUL(z4,h4), _SSE_ADD(_SSE_MUL(y4,h3), _SSE_MUL(x4,h2))));
   w5 = _SSE_SUB(_SSE_MUL(w5,h1), _SSE_ADD(_SSE_MUL(z5,h4), _SSE_ADD(_SSE_MUL(y5,h3), _SSE_MUL(x5,h2))));

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
   h2 = _SSE_MUL(tau5, vs_1_5); 
   h3 = _SSE_MUL(tau5, vs_2_5);
   h4 = _SSE_MUL(tau5, vs_3_5);
   h5 = _SSE_MUL(tau5, vs_4_5);

   v1 = _SSE_SUB(_SSE_MUL(v1,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));
   v2 = _SSE_SUB(_SSE_MUL(v2,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w2,h5), _SSE_MUL(z2,h4)), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2))));
   v3 = _SSE_SUB(_SSE_MUL(v3,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w3,h5), _SSE_MUL(z3,h4)), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2))));
   v4 = _SSE_SUB(_SSE_MUL(v4,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w4,h5), _SSE_MUL(z4,h4)), _SSE_ADD(_SSE_MUL(y4,h3), _SSE_MUL(x4,h2))));
   v5 = _SSE_SUB(_SSE_MUL(v5,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w5,h5), _SSE_MUL(z5,h4)), _SSE_ADD(_SSE_MUL(y5,h3), _SSE_MUL(x5,h2))));

   h2 = _SSE_MUL(tau6, vs_1_6);
   h3 = _SSE_MUL(tau6, vs_2_6);
   h4 = _SSE_MUL(tau6, vs_3_6);
   h5 = _SSE_MUL(tau6, vs_4_6);
   h6 = _SSE_MUL(tau6, vs_5_6);

   t1 = _SSE_SUB(_SSE_MUL(t1,tau6), _SSE_ADD( _SSE_MUL(v1,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)))));
   t2 = _SSE_SUB(_SSE_MUL(t2,tau6), _SSE_ADD( _SSE_MUL(v2,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w2,h5), _SSE_MUL(z2,h4)), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2)))));
   t3 = _SSE_SUB(_SSE_MUL(t3,tau6), _SSE_ADD( _SSE_MUL(v3,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w3,h5), _SSE_MUL(z3,h4)), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2)))));
   t4 = _SSE_SUB(_SSE_MUL(t4,tau6), _SSE_ADD( _SSE_MUL(v4,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w4,h5), _SSE_MUL(z4,h4)), _SSE_ADD(_SSE_MUL(y4,h3), _SSE_MUL(x4,h2)))));
   t5 = _SSE_SUB(_SSE_MUL(t5,tau6), _SSE_ADD( _SSE_MUL(v5,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w5,h5), _SSE_MUL(z5,h4)), _SSE_ADD(_SSE_MUL(y5,h3), _SSE_MUL(x5,h2)))));

   /////////////////////////////////////////////////////
   // Rank-1 update of Q [4 x nb+3]
   /////////////////////////////////////////////////////
#endif /* BLOCK6 */

   q1 = _SSE_LOAD(&q[0]);
#ifdef BLOCK2
   q1 = _SSE_ADD(q1, y1);
#endif
#ifdef BLOCK4
   q1 = _SSE_SUB(q1, w1);
#endif
#ifdef BLOCK6
   q1 = _SSE_SUB(q1, t1); 
#endif
   _SSE_STORE(&q[0],q1);
   q2 = _SSE_LOAD(&q[offset]);
#ifdef BLOCK2
   q2 = _SSE_ADD(q2, y2);
#endif
#ifdef BLOCK4
   q2 = _SSE_SUB(q2, w2);
#endif
#ifdef BLOCK6
   q2 = _SSE_SUB(q2, t2);
#endif
   _SSE_STORE(&q[offset],q2);
   q3 = _SSE_LOAD(&q[2*offset]);
#ifdef BLOCK2
   q3 = _SSE_ADD(q3, y3);
#endif
#ifdef BLOCK4
   q3 = _SSE_SUB(q3, w3);
#endif
#ifdef BLOCK6
   q3 = _SSE_SUB(q3, t3);
#endif
   _SSE_STORE(&q[2*offset],q3);
   q4 = _SSE_LOAD(&q[3*offset]);
#ifdef BLOCK2
   q4 = _SSE_ADD(q4, y4);
#endif
#ifdef BLOCK4
   q4 = _SSE_SUB(q4, w4);
#endif
#ifdef BLOCK6
   q4 = _SSE_SUB(q4, t4);
#endif
   _SSE_STORE(&q[3*offset],q4);
   q5 = _SSE_LOAD(&q[4*offset]);
#ifdef BLOCK2
   q5 = _SSE_ADD(q5, y5);
#endif
#ifdef BLOCK4
   q5 = _SSE_SUB(q5, w5);
#endif
#ifdef BLOCK6
   q5 = _SSE_SUB(q5, t5);
#endif
   _SSE_STORE(&q[4*offset],q5);

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
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
   q4 = _SSE_LOAD(&q[ldq+3*offset]);
   q4 = _SSE_ADD(q4, _SSE_ADD(x4, _SSE_MUL(y4, h2)));
   _SSE_STORE(&q[ldq+3*offset],q4);
   q5 = _SSE_LOAD(&q[ldq+4*offset]);
   q5 = _SSE_ADD(q5, _SSE_ADD(x5, _SSE_MUL(y5, h2)));
   _SSE_STORE(&q[ldq+4*offset],q5);
#endif /* BLOCK2 */

#ifdef BLOCK4
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

   q1 = _SSE_LOAD(&q[ldq]);
   q2 = _SSE_LOAD(&q[ldq+offset]);
   q3 = _SSE_LOAD(&q[ldq+2*offset]);
   q4 = _SSE_LOAD(&q[ldq+3*offset]);
   q5 = _SSE_LOAD(&q[ldq+4*offset]);

   q1 = _SSE_SUB(q1, _SSE_ADD(z1, _SSE_MUL(w1, h4)));
   q2 = _SSE_SUB(q2, _SSE_ADD(z2, _SSE_MUL(w2, h4)));
   q3 = _SSE_SUB(q3, _SSE_ADD(z3, _SSE_MUL(w3, h4)));
   q4 = _SSE_SUB(q4, _SSE_ADD(z4, _SSE_MUL(w4, h4)));
   q5 = _SSE_SUB(q5, _SSE_ADD(z5, _SSE_MUL(w5, h4)));

   _SSE_STORE(&q[ldq],q1);
   _SSE_STORE(&q[ldq+offset],q2);
   _SSE_STORE(&q[ldq+2*offset],q3);
   _SSE_STORE(&q[ldq+3*offset],q4);
   _SSE_STORE(&q[ldq+4*offset],q5);

#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+2]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

   q1 = _SSE_LOAD(&q[ldq*2]);
   q2 = _SSE_LOAD(&q[(ldq*2)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*2)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*2)+3*offset]);
   q5 = _SSE_LOAD(&q[(ldq*2)+4*offset]);
   q1 = _SSE_SUB(q1, y1);
   q2 = _SSE_SUB(q2, y2);
   q3 = _SSE_SUB(q3, y3);
   q4 = _SSE_SUB(q4, y4);
   q5 = _SSE_SUB(q5, y5);

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
   q5 = _SSE_SUB(q5, _SSE_MUL(w5, h4));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
   q5 = _SSE_SUB(q5, _SSE_MUL(z5, h3));

   _SSE_STORE(&q[ldq*2],q1);
   _SSE_STORE(&q[(ldq*2)+offset],q2);
   _SSE_STORE(&q[(ldq*2)+2*offset],q3);
   _SSE_STORE(&q[(ldq*2)+3*offset],q4);
   _SSE_STORE(&q[(ldq*2)+4*offset],q5);

#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+3]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

   q1 = _SSE_LOAD(&q[ldq*3]);
   q2 = _SSE_LOAD(&q[(ldq*3)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*3)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*3)+3*offset]);
   q5 = _SSE_LOAD(&q[(ldq*3)+4*offset]);

   q1 = _SSE_SUB(q1, x1);
   q2 = _SSE_SUB(q2, x2);
   q3 = _SSE_SUB(q3, x3);
   q4 = _SSE_SUB(q4, x4);
   q5 = _SSE_SUB(q5, x5);

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
   q5 = _SSE_SUB(q5, _SSE_MUL(w5, h4));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));
   q5 = _SSE_SUB(q5, _SSE_MUL(y5, h2));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+2]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
   q5 = _SSE_SUB(q5, _SSE_MUL(z5, h3));
   _SSE_STORE(&q[ldq*3], q1);
   _SSE_STORE(&q[(ldq*3)+offset], q2);
   _SSE_STORE(&q[(ldq*3)+2*offset], q3);
   _SSE_STORE(&q[(ldq*3)+3*offset], q4);
   _SSE_STORE(&q[(ldq*3)+4*offset], q5);

#endif /* BLOCK4 */

#ifdef BLOCK6
   
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
#endif

   q1 = _SSE_LOAD(&q[ldq]);
   q2 = _SSE_LOAD(&q[(ldq+offset)]);
   q3 = _SSE_LOAD(&q[(ldq+2*offset)]);
   q4 = _SSE_LOAD(&q[(ldq+3*offset)]);
   q5 = _SSE_LOAD(&q[(ldq+4*offset)]);
   q1 = _SSE_SUB(q1, v1);
   q2 = _SSE_SUB(q2, v2);
   q3 = _SSE_SUB(q3, v3);
   q4 = _SSE_SUB(q4, v4);
   q5 = _SSE_SUB(q5, v5);

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
   q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));
   q5 = _SSE_SUB(q5, _SSE_MUL(t5, h6));

   _SSE_STORE(&q[ldq],q1);
   _SSE_STORE(&q[(ldq+offset)],q2);
   _SSE_STORE(&q[(ldq+2*offset)],q3);
   _SSE_STORE(&q[(ldq+3*offset)],q4);
   _SSE_STORE(&q[(ldq+4*offset)],q5);
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
#endif
   q1 = _SSE_LOAD(&q[ldq*2]);
   q2 = _SSE_LOAD(&q[(ldq*2)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*2)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*2)+3*offset]);
   q5 = _SSE_LOAD(&q[(ldq*2)+4*offset]);
   q1 = _SSE_SUB(q1, w1); 
   q2 = _SSE_SUB(q2, w2);
   q3 = _SSE_SUB(q3, w3);
   q4 = _SSE_SUB(q4, w4);
   q5 = _SSE_SUB(q5, w5);
   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5)); 
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));  
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));  
   q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));  
   q5 = _SSE_SUB(q5, _SSE_MUL(v5, h5));  
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
   q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));
   q5 = _SSE_SUB(q5, _SSE_MUL(t5, h6));

   _SSE_STORE(&q[ldq*2],q1);
   _SSE_STORE(&q[(ldq*2)+offset],q2);
   _SSE_STORE(&q[(ldq*2)+2*offset],q3);
   _SSE_STORE(&q[(ldq*2)+3*offset],q4);
   _SSE_STORE(&q[(ldq*2)+4*offset],q5);

#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

   q1 = _SSE_LOAD(&q[ldq*3]);
   q2 = _SSE_LOAD(&q[(ldq*3)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*3)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*3)+3*offset]);
   q5 = _SSE_LOAD(&q[(ldq*3)+4*offset]);
   q1 = _SSE_SUB(q1, z1);
   q2 = _SSE_SUB(q2, z2);
   q3 = _SSE_SUB(q3, z3);
   q4 = _SSE_SUB(q4, z4);
   q5 = _SSE_SUB(q5, z5);

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
   q5 = _SSE_SUB(q5, _SSE_MUL(w5, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
   q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));
   q5 = _SSE_SUB(q5, _SSE_MUL(v5, h5));
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
   q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));
   q5 = _SSE_SUB(q5, _SSE_MUL(t5, h6));

   _SSE_STORE(&q[ldq*3],q1);
   _SSE_STORE(&q[(ldq*3)+offset],q2);
   _SSE_STORE(&q[(ldq*3)+2*offset],q3);
   _SSE_STORE(&q[(ldq*3)+3*offset],q4);
   _SSE_STORE(&q[(ldq*3)+4*offset],q5);
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif
   q1 = _SSE_LOAD(&q[ldq*4]);
   q2 = _SSE_LOAD(&q[(ldq*4)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*4)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*4)+3*offset]);
   q5 = _SSE_LOAD(&q[(ldq*4)+4*offset]);
   q1 = _SSE_SUB(q1, y1);
   q2 = _SSE_SUB(q2, y2);
   q3 = _SSE_SUB(q3, y3);
   q4 = _SSE_SUB(q4, y4);
   q5 = _SSE_SUB(q5, y5);

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
   q5 = _SSE_SUB(q5, _SSE_MUL(z5, h3));
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
   q5 = _SSE_SUB(q5, _SSE_MUL(w5, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
   q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));
   q5 = _SSE_SUB(q5, _SSE_MUL(v5, h5));
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+4]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
   q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));
   q5 = _SSE_SUB(q5, _SSE_MUL(t5, h6));

   _SSE_STORE(&q[ldq*4],q1);
   _SSE_STORE(&q[(ldq*4)+offset],q2);
   _SSE_STORE(&q[(ldq*4)+2*offset],q3);
   _SSE_STORE(&q[(ldq*4)+3*offset],q4);
   _SSE_STORE(&q[(ldq*4)+4*offset],q5);
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[(ldh)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[(ldh)+1], hh[(ldh)+1]);
#endif
   q1 = _SSE_LOAD(&q[ldq*5]);
   q2 = _SSE_LOAD(&q[(ldq*5)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*5)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*5)+3*offset]);
   q5 = _SSE_LOAD(&q[(ldq*5)+4*offset]);
   q1 = _SSE_SUB(q1, x1);
   q2 = _SSE_SUB(q2, x2);
   q3 = _SSE_SUB(q3, x3);
   q4 = _SSE_SUB(q4, x4);
   q5 = _SSE_SUB(q5, x5);

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));
   q5 = _SSE_SUB(q5, _SSE_MUL(y5, h2));
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
   q5 = _SSE_SUB(q5, _SSE_MUL(z5, h3));
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
   q5 = _SSE_SUB(q5, _SSE_MUL(w5, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+4]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
   q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));
   q5 = _SSE_SUB(q5, _SSE_MUL(v5, h5));
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+5]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
   q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));
   q5 = _SSE_SUB(q5, _SSE_MUL(t5, h6));

   _SSE_STORE(&q[ldq*5],q1);
   _SSE_STORE(&q[(ldq*5)+offset],q2);
   _SSE_STORE(&q[(ldq*5)+2*offset],q3);
   _SSE_STORE(&q[(ldq*5)+3*offset],q4);
   _SSE_STORE(&q[(ldq*5)+4*offset],q5);

#endif /* BLOCK6 */

   for (i = BLOCK; i < nb; i++)
   {
#ifdef HAVE_SSE_INTRINSICS
     h1 = _SSE_SET1(hh[i-(BLOCK-1)]);
     h2 = _SSE_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#ifdef HAVE_SPARC64_SSE
     h1 = _SSE_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
     h2 = _SSE_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif

     q1 = _SSE_LOAD(&q[i*ldq]);
     q2 = _SSE_LOAD(&q[(i*ldq)+offset]);
     q3 = _SSE_LOAD(&q[(i*ldq)+2*offset]);
     q4 = _SSE_LOAD(&q[(i*ldq)+3*offset]);
     q5 = _SSE_LOAD(&q[(i*ldq)+4*offset]);

#ifdef BLOCK2
     q1 = _SSE_ADD(q1, _SSE_ADD(_SSE_MUL(x1,h1), _SSE_MUL(y1, h2)));
     q2 = _SSE_ADD(q2, _SSE_ADD(_SSE_MUL(x2,h1), _SSE_MUL(y2, h2)));
     q3 = _SSE_ADD(q3, _SSE_ADD(_SSE_MUL(x3,h1), _SSE_MUL(y3, h2)));
     q4 = _SSE_ADD(q4, _SSE_ADD(_SSE_MUL(x4,h1), _SSE_MUL(y4, h2)));
     q5 = _SSE_ADD(q5, _SSE_ADD(_SSE_MUL(x5,h1), _SSE_MUL(y5, h2)));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
     
     q1 = _SSE_SUB(q1, _SSE_MUL(x1,h1));
     q2 = _SSE_SUB(q2, _SSE_MUL(x2,h1));
     q3 = _SSE_SUB(q3, _SSE_MUL(x3,h1));
     q4 = _SSE_SUB(q4, _SSE_MUL(x4,h1));
     q5 = _SSE_SUB(q5, _SSE_MUL(x5,h1));

     q1 = _SSE_SUB(q1, _SSE_MUL(y1,h2));
     q2 = _SSE_SUB(q2, _SSE_MUL(y2,h2));
     q3 = _SSE_SUB(q3, _SSE_MUL(y3,h2));
     q4 = _SSE_SUB(q4, _SSE_MUL(y4,h2));
     q5 = _SSE_SUB(q5, _SSE_MUL(y5,h2));

#ifdef HAVE_SSE_INTRINSICS
     h3 = _SSE_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
     h3 = _SSE_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(z1,h3));
     q2 = _SSE_SUB(q2, _SSE_MUL(z2,h3));
     q3 = _SSE_SUB(q3, _SSE_MUL(z3,h3));
     q4 = _SSE_SUB(q4, _SSE_MUL(z4,h3));
     q5 = _SSE_SUB(q5, _SSE_MUL(z5,h3));

#ifdef HAVE_SSE_INTRINSICS
     h4 = _SSE_SET1(hh[(ldh*3)+i-(BLOCK-4)]); 
#endif

#ifdef HAVE_SPRC64_SSE
     h4 = _SSE_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(w1,h4));
     q2 = _SSE_SUB(q2, _SSE_MUL(w2,h4));
     q3 = _SSE_SUB(q3, _SSE_MUL(w3,h4));
     q4 = _SSE_SUB(q4, _SSE_MUL(w4,h4));
     q5 = _SSE_SUB(q5, _SSE_MUL(w5,h4));

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
     h5 = _SSE_SET1(hh[(ldh*4)+i-1]);
#endif
#ifdef HAVE_SPARC64_SSE
     h5 = _SSE_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
     q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
     q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
     q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));
     q5 = _SSE_SUB(q5, _SSE_MUL(v5, h5));
#ifdef HAVE_SSE_INTRINSICS
     h6 = _SSE_SET1(hh[(ldh*5)+i]);
#endif
#ifdef HAVE_SPARC64_SSE
     h6 = _SSE_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
     q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
     q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
     q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));
     q5 = _SSE_SUB(q5, _SSE_MUL(t5, h6));
#endif /* BLOCK6 */

     _SSE_STORE(&q[i*ldq],q1);
     _SSE_STORE(&q[(i*ldq)+offset],q2);
     _SSE_STORE(&q[(i*ldq)+2*offset],q3);
     _SSE_STORE(&q[(i*ldq)+3*offset],q4);
     _SSE_STORE(&q[(i*ldq)+4*offset],q5);

   }
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-1)]);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif

   q1 = _SSE_LOAD(&q[nb*ldq]);
   q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);
   q3 = _SSE_LOAD(&q[(nb*ldq)+2*offset]);
   q4 = _SSE_LOAD(&q[(nb*ldq)+3*offset]);
   q5 = _SSE_LOAD(&q[(nb*ldq)+4*offset]);

#ifdef BLOCK2
   q1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_ADD(q4, _SSE_MUL(x4, h1));
   q5 = _SSE_ADD(q5, _SSE_MUL(x5, h1));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_SUB(q4, _SSE_MUL(x4, h1));
   q5 = _SSE_SUB(q5, _SSE_MUL(x5, h1));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));
   q5 = _SSE_SUB(q5, _SSE_MUL(y5, h2));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
   q5 = _SSE_SUB(q5, _SSE_MUL(z5, h3));

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
   q5 = _SSE_SUB(q5, _SSE_MUL(w5, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
   q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));
   q5 = _SSE_SUB(q5, _SSE_MUL(v5, h5));
#endif /* BLOCK6 */

   _SSE_STORE(&q[nb*ldq],q1);
   _SSE_STORE(&q[(nb*ldq)+offset],q2);
   _SSE_STORE(&q[(nb*ldq)+2*offset],q3);
   _SSE_STORE(&q[(nb*ldq)+3*offset],q4);
   _SSE_STORE(&q[(nb*ldq)+4*offset],q5);

#if defined(BLOCK4) || defined(BLOCK6)
   
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-2)], hh[nb-(BLOCK-2)]);
#endif

   q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);
   q3 = _SSE_LOAD(&q[((nb+1)*ldq)+2*offset]);
   q4 = _SSE_LOAD(&q[((nb+1)*ldq)+3*offset]);
   q5 = _SSE_LOAD(&q[((nb+1)*ldq)+4*offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_SUB(q4, _SSE_MUL(x4, h1));
   q5 = _SSE_SUB(q5, _SSE_MUL(x5, h1));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-(BLOCK-3)], hh[ldh+nb-(BLOCK-3)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));
   q5 = _SSE_SUB(q5, _SSE_MUL(y5, h2));

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
   q5 = _SSE_SUB(q5, _SSE_MUL(z5, h3));
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
   q5 = _SSE_SUB(q5, _SSE_MUL(w5, h4));

#endif /* BLOCK6 */

   _SSE_STORE(&q[(nb+1)*ldq],q1);
   _SSE_STORE(&q[((nb+1)*ldq)+offset],q2);
   _SSE_STORE(&q[((nb+1)*ldq)+2*offset],q3);
   _SSE_STORE(&q[((nb+1)*ldq)+3*offset],q4);
   _SSE_STORE(&q[((nb+1)*ldq)+4*offset],q5);

#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-3)], hh[nb-(BLOCK-3)]);
#endif

   q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);
   q3 = _SSE_LOAD(&q[((nb+2)*ldq)+2*offset]);
   q4 = _SSE_LOAD(&q[((nb+2)*ldq)+3*offset]);
   q5 = _SSE_LOAD(&q[((nb+2)*ldq)+4*offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_SUB(q4, _SSE_MUL(x4, h1));
   q5 = _SSE_SUB(q5, _SSE_MUL(x5, h1));

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));
   q5 = _SSE_SUB(q5, _SSE_MUL(y5, h2));
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
   q5 = _SSE_SUB(q5, _SSE_MUL(z5, h3));
#endif /* BLOCK6 */

   _SSE_STORE(&q[(nb+2)*ldq],q1);
   _SSE_STORE(&q[((nb+2)*ldq)+offset],q2);
   _SSE_STORE(&q[((nb+2)*ldq)+2*offset],q3);
   _SSE_STORE(&q[((nb+2)*ldq)+3*offset],q4);
   _SSE_STORE(&q[((nb+2)*ldq)+4*offset],q5);

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

   q1 = _SSE_LOAD(&q[(nb+3)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+3)*ldq)+offset]);
   q3 = _SSE_LOAD(&q[((nb+3)*ldq)+2*offset]);
   q4 = _SSE_LOAD(&q[((nb+3)*ldq)+3*offset]);
   q5 = _SSE_LOAD(&q[((nb+3)*ldq)+4*offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_SUB(q4, _SSE_MUL(x4, h1));
   q5 = _SSE_SUB(q5, _SSE_MUL(x5, h1));
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));
   q5 = _SSE_SUB(q5, _SSE_MUL(y5, h2));

   _SSE_STORE(&q[(nb+3)*ldq],q1);
   _SSE_STORE(&q[((nb+3)*ldq)+offset],q2);
   _SSE_STORE(&q[((nb+3)*ldq)+2*offset],q3);
   _SSE_STORE(&q[((nb+3)*ldq)+3*offset],q4);
   _SSE_STORE(&q[((nb+3)*ldq)+4*offset],q5);
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif

   q1 = _SSE_LOAD(&q[(nb+4)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+4)*ldq)+offset]);
   q3 = _SSE_LOAD(&q[((nb+4)*ldq)+2*offset]);
   q4 = _SSE_LOAD(&q[((nb+4)*ldq)+3*offset]);
   q5 = _SSE_LOAD(&q[((nb+4)*ldq)+4*offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_SUB(q4, _SSE_MUL(x4, h1));
   q5 = _SSE_SUB(q5, _SSE_MUL(x5, h1));

   _SSE_STORE(&q[(nb+4)*ldq],q1);
   _SSE_STORE(&q[((nb+4)*ldq)+offset],q2);
   _SSE_STORE(&q[((nb+4)*ldq)+2*offset],q3);
   _SSE_STORE(&q[((nb+4)*ldq)+3*offset],q4);
   _SSE_STORE(&q[((nb+4)*ldq)+4*offset],q5);

#endif /* BLOCK6 */
}


/*
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 8 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 16 rows of Q simultaneously, a
#endif
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
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 8
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 16
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
    // Matrix Vector Multiplication, Q [6 x nb+1] * hh
    // hh contains two householder vectors, with offset 1
    /////////////////////////////////////////////////////
#endif
#if defined(BLOCK4) || defined(BLOCK6)
    /////////////////////////////////////////////////////
    // Matrix Vector Multiplication, Q [6 x nb+3] * hh
    // hh contains four householder vectors
    /////////////////////////////////////////////////////
#endif

    int i;
#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
    // Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
    __SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SSE_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif
    __SSE_DATATYPE x1 = _SSE_LOAD(&q[ldq]);
    __SSE_DATATYPE x2 = _SSE_LOAD(&q[ldq+offset]);
    __SSE_DATATYPE x3 = _SSE_LOAD(&q[ldq+2*offset]);
    __SSE_DATATYPE x4 = _SSE_LOAD(&q[ldq+3*offset]);

#ifdef HAVE_SSE_INTRINSICS
    __SSE_DATATYPE h1 = _SSE_SET1(hh[ldh+1]);
#endif
#ifdef HAVE_SPARC64_SSE
    __SSE_DATATYPE h1 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
#endif
    __SSE_DATATYPE h2;

    __SSE_DATATYPE q1 = _SSE_LOAD(q);
    __SSE_DATATYPE y1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
    __SSE_DATATYPE q2 = _SSE_LOAD(&q[offset]);
    __SSE_DATATYPE y2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
    __SSE_DATATYPE q3 = _SSE_LOAD(&q[2*offset]);
    __SSE_DATATYPE y3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
    __SSE_DATATYPE q4 = _SSE_LOAD(&q[3*offset]);
    __SSE_DATATYPE y4 = _SSE_ADD(q4, _SSE_MUL(x4, h1));
#endif /* BLOCK2 */

#ifdef BLOCK4
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

    register __SSE_DATATYPE w1 = _SSE_ADD(a4_1, _SSE_MUL(a3_1, h_4_3));
    w1 = _SSE_ADD(w1, _SSE_MUL(a2_1, h_4_2));                          
    w1 = _SSE_ADD(w1, _SSE_MUL(a1_1, h_4_1));                          
    register __SSE_DATATYPE z1 = _SSE_ADD(a3_1, _SSE_MUL(a2_1, h_3_2));
    z1 = _SSE_ADD(z1, _SSE_MUL(a1_1, h_3_1));                          
    register __SSE_DATATYPE y1 = _SSE_ADD(a2_1, _SSE_MUL(a1_1, h_2_1));
    register __SSE_DATATYPE x1 = a1_1; 

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

    __SSE_DATATYPE a1_4 = _SSE_LOAD(&q[(ldq*3)+3*offset]);
    __SSE_DATATYPE a2_4 = _SSE_LOAD(&q[(ldq*2)+3*offset]);
    __SSE_DATATYPE a3_4 = _SSE_LOAD(&q[ldq+3*offset]);
    __SSE_DATATYPE a4_4 = _SSE_LOAD(&q[0+3*offset]);

    register __SSE_DATATYPE w4 = _SSE_ADD(a4_4, _SSE_MUL(a3_4, h_4_3));
    w4 = _SSE_ADD(w4, _SSE_MUL(a2_4, h_4_2));
    w4 = _SSE_ADD(w4, _SSE_MUL(a1_4, h_4_1));
    register __SSE_DATATYPE z4 = _SSE_ADD(a3_4, _SSE_MUL(a2_4, h_3_2));
    z4 = _SSE_ADD(z4, _SSE_MUL(a1_4, h_3_1));
    register __SSE_DATATYPE y4 = _SSE_ADD(a2_4, _SSE_MUL(a1_4, h_2_1));
    register __SSE_DATATYPE x4 = a1_4;

    __SSE_DATATYPE q1;
    __SSE_DATATYPE q2;
    __SSE_DATATYPE q3;
    __SSE_DATATYPE q4;

    __SSE_DATATYPE h1;
    __SSE_DATATYPE h2;
    __SSE_DATATYPE h3;
    __SSE_DATATYPE h4;
#endif /* BLOCK4 */

#ifdef BLOCK6
    
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

    __SSE_DATATYPE a1_3 = _SSE_LOAD(&q[(ldq*5)+2*offset]);
    __SSE_DATATYPE a2_3 = _SSE_LOAD(&q[(ldq*4)+2*offset]);
    __SSE_DATATYPE a3_3 = _SSE_LOAD(&q[(ldq*3)+2*offset]);
    __SSE_DATATYPE a4_3 = _SSE_LOAD(&q[(ldq*2)+2*offset]);
    __SSE_DATATYPE a5_3 = _SSE_LOAD(&q[(ldq)+2*offset]);
    __SSE_DATATYPE a6_3 = _SSE_LOAD(&q[2*offset]);

    register __SSE_DATATYPE t3 = _SSE_ADD(a6_3, _SSE_MUL(a5_3, h_6_5));
    t3 = _SSE_ADD(t3, _SSE_MUL(a4_3, h_6_4));
    t3 = _SSE_ADD(t3, _SSE_MUL(a3_3, h_6_3));
    t3 = _SSE_ADD(t3, _SSE_MUL(a2_3, h_6_2));
    t3 = _SSE_ADD(t3, _SSE_MUL(a1_3, h_6_1));
    register __SSE_DATATYPE v3 = _SSE_ADD(a5_3, _SSE_MUL(a4_3, h_5_4));
    v3 = _SSE_ADD(v3, _SSE_MUL(a3_3, h_5_3));
    v3 = _SSE_ADD(v3, _SSE_MUL(a2_3, h_5_2));
    v3 = _SSE_ADD(v3, _SSE_MUL(a1_3, h_5_1));
    register __SSE_DATATYPE w3 = _SSE_ADD(a4_3, _SSE_MUL(a3_3, h_4_3));
    w3 = _SSE_ADD(w3, _SSE_MUL(a2_3, h_4_2));
    w3 = _SSE_ADD(w3, _SSE_MUL(a1_3, h_4_1));
    register __SSE_DATATYPE z3 = _SSE_ADD(a3_3, _SSE_MUL(a2_3, h_3_2));
    z3 = _SSE_ADD(z3, _SSE_MUL(a1_3, h_3_1));
    register __SSE_DATATYPE y3 = _SSE_ADD(a2_3, _SSE_MUL(a1_3, h_2_1));

    register __SSE_DATATYPE x3 = a1_3;

    __SSE_DATATYPE a1_4 = _SSE_LOAD(&q[(ldq*5)+3*offset]);
    __SSE_DATATYPE a2_4 = _SSE_LOAD(&q[(ldq*4)+3*offset]);
    __SSE_DATATYPE a3_4 = _SSE_LOAD(&q[(ldq*3)+3*offset]);
    __SSE_DATATYPE a4_4 = _SSE_LOAD(&q[(ldq*2)+3*offset]);
    __SSE_DATATYPE a5_4 = _SSE_LOAD(&q[(ldq)+3*offset]);
    __SSE_DATATYPE a6_4 = _SSE_LOAD(&q[3*offset]);

    register __SSE_DATATYPE t4 = _SSE_ADD(a6_4, _SSE_MUL(a5_4, h_6_5));
    t4 = _SSE_ADD(t4, _SSE_MUL(a4_4, h_6_4));
    t4 = _SSE_ADD(t4, _SSE_MUL(a3_4, h_6_3));
    t4 = _SSE_ADD(t4, _SSE_MUL(a2_4, h_6_2));
    t4 = _SSE_ADD(t4, _SSE_MUL(a1_4, h_6_1));
    register __SSE_DATATYPE v4 = _SSE_ADD(a5_4, _SSE_MUL(a4_4, h_5_4));
    v4 = _SSE_ADD(v4, _SSE_MUL(a3_4, h_5_3));
    v4 = _SSE_ADD(v4, _SSE_MUL(a2_4, h_5_2));
    v4 = _SSE_ADD(v4, _SSE_MUL(a1_4, h_5_1));
    register __SSE_DATATYPE w4 = _SSE_ADD(a4_4, _SSE_MUL(a3_4, h_4_3));
    w4 = _SSE_ADD(w4, _SSE_MUL(a2_4, h_4_2));
    w4 = _SSE_ADD(w4, _SSE_MUL(a1_4, h_4_1));
    register __SSE_DATATYPE z4 = _SSE_ADD(a3_4, _SSE_MUL(a2_4, h_3_2));
    z4 = _SSE_ADD(z4, _SSE_MUL(a1_4, h_3_1));
    register __SSE_DATATYPE y4 = _SSE_ADD(a2_4, _SSE_MUL(a1_4, h_2_1));

    register __SSE_DATATYPE x4 = a1_4;


    __SSE_DATATYPE q1;
    __SSE_DATATYPE q2;
    __SSE_DATATYPE q3;
    __SSE_DATATYPE q4;

    __SSE_DATATYPE h1;
    __SSE_DATATYPE h2;
    __SSE_DATATYPE h3;
    __SSE_DATATYPE h4;
    __SSE_DATATYPE h5;
    __SSE_DATATYPE h6;

#endif /* BLOCK6 */

    for(i = BLOCK; i < nb; i++)
      {
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[i-(BLOCK-1)]);
        h2 = _SSE_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
        h2 = _SSE_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
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
        q4 = _SSE_LOAD(&q[(i*ldq)+3*offset]);
        x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
        y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
#if defined(BLOCK4) || defined(BLOCK6)
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

        z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
        z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
        z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));
        z4 = _SSE_ADD(z4, _SSE_MUL(q4,h3));
#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

        w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
        w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));
        w3 = _SSE_ADD(w3, _SSE_MUL(q3,h4));
        w4 = _SSE_ADD(w4, _SSE_MUL(q4,h4));
	
#endif /* BLOCK4 || BLOCK6 */
#ifdef BLOCK6

#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+i-1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif
        v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));
        v2 = _SSE_ADD(v2, _SSE_MUL(q2,h5));
        v3 = _SSE_ADD(v3, _SSE_MUL(q3,h5));
        v4 = _SSE_ADD(v4, _SSE_MUL(q4,h5));

#ifdef HAVE_SSE_INTRINSICS
        h6 = _SSE_SET1(hh[(ldh*5)+i]);
#endif

#ifdef HAVE_SPARC64_SSE
        h6 = _SSE_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif

        t1 = _SSE_ADD(t1, _SSE_MUL(q1,h6));
        t2 = _SSE_ADD(t2, _SSE_MUL(q2,h6));
        t3 = _SSE_ADD(t3, _SSE_MUL(q3,h6));
        t4 = _SSE_ADD(t4, _SSE_MUL(q4,h6));
	
#endif /* BLOCK6 */
      }
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-(BLOCK-1)]);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif

    q1 = _SSE_LOAD(&q[nb*ldq]);
    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    q3 = _SSE_LOAD(&q[(nb*ldq)+2*offset]);
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    q4 = _SSE_LOAD(&q[(nb*ldq)+3*offset]);
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));

#if defined(BLOCK4) || defined(BLOCK6)
    
#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
    y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));

#ifdef HAVE_SSE_INTRINSICS
    h3 = _SSE_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
    h3 = _SSE_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

    z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
    z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
    z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));
    z4 = _SSE_ADD(z4, _SSE_MUL(q4,h3));

#ifdef BLOCK4
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

    q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+1)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+1)*ldq)+3*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));

#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[(ldh*1)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[(ldh*1)+nb-1], hh[(ldh*1)+nb-1]);
#endif


    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
    y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));

#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif


    q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+2)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+2)*ldq)+3*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));

#endif /* BLOCK4 */
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#ifdef HAVE_SSE_INTRINSICS
    h4 = _SSE_SET1(hh[(ldh*3)+nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h4 = _SSE_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif

    w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4)); 
    w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));
    w3 = _SSE_ADD(w3, _SSE_MUL(q3,h4));
    w4 = _SSE_ADD(w4, _SSE_MUL(q4,h4));

#ifdef HAVE_SSE_INTRINSICS
    h5 = _SSE_SET1(hh[(ldh*4)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h5 = _SSE_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif

    v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));
    v2 = _SSE_ADD(v2, _SSE_MUL(q2,h5));
    v3 = _SSE_ADD(v3, _SSE_MUL(q3,h5));
    v4 = _SSE_ADD(v4, _SSE_MUL(q4,h5));
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-4]);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-4], hh[nb-4]);
#endif

    q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+1)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+1)*ldq)+3*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-3]);
#endif
#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-3], hh[ldh+nb-3]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
    y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));

#ifdef HAVE_SSE_INTRINSICS
    h3 = _SSE_SET1(hh[(ldh*2)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
    h3 = _SSE_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif

    z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
    z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
    z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));
    z4 = _SSE_ADD(z4, _SSE_MUL(q4,h3));

#ifdef HAVE_SSE_INTRINSICS
    h4 = _SSE_SET1(hh[(ldh*3)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
    h4 = _SSE_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif

    w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
    w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));
    w3 = _SSE_ADD(w3, _SSE_MUL(q3,h4));
    w4 = _SSE_ADD(w4, _SSE_MUL(q4,h4));

#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-3]);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-3], hh[nb-3]);
#endif

    q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+2)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+2)*ldq)+3*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));

#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
    y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
#ifdef HAVE_SSE_INTRINSICS
    h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif

    z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
    z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
    z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));
    z4 = _SSE_ADD(z4, _SSE_MUL(q4,h3));
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

    q1 = _SSE_LOAD(&q[(nb+3)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+3)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+3)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+3)*ldq)+3*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));

#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
    y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));

#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif

    q1 = _SSE_LOAD(&q[(nb+4)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+4)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+4)*ldq)+2*offset]);
    q4 = _SSE_LOAD(&q[((nb+4)*ldq)+3*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
    x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
#endif /* BLOCK6 */

#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Rank-2 update of Q [6 x nb+1]
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

#ifdef HAVE_SSE_INTRINSICS
    __SSE_DATATYPE tau1 = _SSE_SET1(hh[0]);

    __SSE_DATATYPE tau2 = _SSE_SET1(hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SSE_DATATYPE tau3 = _SSE_SET1(hh[ldh*2]);
   __SSE_DATATYPE tau4 = _SSE_SET1(hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE tau5 = _SSE_SET1(hh[ldh*4]);
   __SSE_DATATYPE tau6 = _SSE_SET1(hh[ldh*5]);       
#endif

#ifdef BLOCK2    
    __SSE_DATATYPE vs = _SSE_SET1(s);
#endif
#ifdef BLOCK4
   __SSE_DATATYPE vs_1_2 = _SSE_SET1(s_1_2);
   __SSE_DATATYPE vs_1_3 = _SSE_SET1(s_1_3);  
   __SSE_DATATYPE vs_2_3 = _SSE_SET1(s_2_3);
   __SSE_DATATYPE vs_1_4 = _SSE_SET1(s_1_4);  
   __SSE_DATATYPE vs_2_4 = _SSE_SET1(s_2_4);  
   __SSE_DATATYPE vs_3_4 = _SSE_SET1(s_3_4);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE vs_1_2 = _SSE_SET1(scalarprods[0]);
   __SSE_DATATYPE vs_1_3 = _SSE_SET1(scalarprods[1]);
   __SSE_DATATYPE vs_2_3 = _SSE_SET1(scalarprods[2]);
   __SSE_DATATYPE vs_1_4 = _SSE_SET1(scalarprods[3]);
   __SSE_DATATYPE vs_2_4 = _SSE_SET1(scalarprods[4]);
   __SSE_DATATYPE vs_3_4 = _SSE_SET1(scalarprods[5]);
   __SSE_DATATYPE vs_1_5 = _SSE_SET1(scalarprods[6]);
   __SSE_DATATYPE vs_2_5 = _SSE_SET1(scalarprods[7]);
   __SSE_DATATYPE vs_3_5 = _SSE_SET1(scalarprods[8]);
   __SSE_DATATYPE vs_4_5 = _SSE_SET1(scalarprods[9]);
   __SSE_DATATYPE vs_1_6 = _SSE_SET1(scalarprods[10]);
   __SSE_DATATYPE vs_2_6 = _SSE_SET1(scalarprods[11]);
   __SSE_DATATYPE vs_3_6 = _SSE_SET1(scalarprods[12]);
   __SSE_DATATYPE vs_4_6 = _SSE_SET1(scalarprods[13]);
   __SSE_DATATYPE vs_5_6 = _SSE_SET1(scalarprods[14]);
#endif
#endif /* HAVE_SSE_INTRINSICS */

#ifdef HAVE_SPARC64_SSE
    __SSE_DATATYPE tau1 = _SSE_SET(hh[0], hh[0]);

    __SSE_DATATYPE tau2 = _SSE_SET(hh[ldh], hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SSE_DATATYPE tau3 = _SSE_SET(hh[ldh*2], hh[ldh*2]);
   __SSE_DATATYPE tau4 = _SSE_SET(hh[ldh*3], hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE tau5 = _SSE_SET(hh[ldh*4], hh[ldh*4]);
   __SSE_DATATYPE tau6 = _SSE_SET(hh[ldh*5], hh[ldh*5]);
#endif

#ifdef BLOCK2
    __SSE_DATATYPE vs = _SSE_SET(s, s);
#endif
#ifdef BLOCK4
   __SSE_DATATYPE vs_1_2 = _SSE_SET(s_1_2, s_1_2);
   __SSE_DATATYPE vs_1_3 = _SSE_SET(s_1_3, s_1_3);
   __SSE_DATATYPE vs_2_3 = _SSE_SET(s_2_3, s_2_3);
   __SSE_DATATYPE vs_1_4 = _SSE_SET(s_1_4, s_1_4);
   __SSE_DATATYPE vs_2_4 = _SSE_SET(s_2_4, s_2_4);
   __SSE_DATATYPE vs_3_4 = _SSE_SET(s_3_4, s_3_4);

#endif
#ifdef BLOCK6
   __SSE_DATATYPE vs_1_2 = _SSE_SET(scalarprods[0], scalarprods[0]);
   __SSE_DATATYPE vs_1_3 = _SSE_SET(scalarprods[1], scalarprods[1]);
   __SSE_DATATYPE vs_2_3 = _SSE_SET(scalarprods[2], scalarprods[2]);
   __SSE_DATATYPE vs_1_4 = _SSE_SET(scalarprods[3], scalarprods[3]);
   __SSE_DATATYPE vs_2_4 = _SSE_SET(scalarprods[4], scalarprods[4]);
   __SSE_DATATYPE vs_3_4 = _SSE_SET(scalarprods[5], scalarprods[5]);
   __SSE_DATATYPE vs_1_5 = _SSE_SET(scalarprods[6], scalarprods[6]);
   __SSE_DATATYPE vs_2_5 = _SSE_SET(scalarprods[7], scalarprods[7]);
   __SSE_DATATYPE vs_3_5 = _SSE_SET(scalarprods[8], scalarprods[8]);
   __SSE_DATATYPE vs_4_5 = _SSE_SET(scalarprods[9], scalarprods[9]);
   __SSE_DATATYPE vs_1_6 = _SSE_SET(scalarprods[10], scalarprods[10]);
   __SSE_DATATYPE vs_2_6 = _SSE_SET(scalarprods[11], scalarprods[11]);
   __SSE_DATATYPE vs_3_6 = _SSE_SET(scalarprods[12], scalarprods[12]);
   __SSE_DATATYPE vs_4_6 = _SSE_SET(scalarprods[13], scalarprods[13]);
   __SSE_DATATYPE vs_5_6 = _SSE_SET(scalarprods[14], scalarprods[14]);
#endif
#endif /* HAVE_SPARC64_SSE */

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_XOR(tau1, sign);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _fjsp_neg_v2r8(tau1);
#endif
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau1;
#endif

   x1 = _SSE_MUL(x1, h1);
   x2 = _SSE_MUL(x2, h1);
   x3 = _SSE_MUL(x3, h1);
   x4 = _SSE_MUL(x4, h1);

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_XOR(tau2, sign);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _fjsp_neg_v2r8(tau2);
#endif
   h2 = _SSE_MUL(h1, vs);
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau2;
   h2 = _SSE_MUL(h1, vs_1_2);
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK2
   y1 = _SSE_ADD(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
   y2 = _SSE_ADD(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
   y3 = _SSE_ADD(_SSE_MUL(y3,h1), _SSE_MUL(x3,h2));
   y4 = _SSE_ADD(_SSE_MUL(y4,h1), _SSE_MUL(x4,h2));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   y1 = _SSE_SUB(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
   y2 = _SSE_SUB(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
   y3 = _SSE_SUB(_SSE_MUL(y3,h1), _SSE_MUL(x3,h2));
   y4 = _SSE_SUB(_SSE_MUL(y4,h1), _SSE_MUL(x4,h2));

   h1 = tau3;
   h2 = _SSE_MUL(h1, vs_1_3);
   h3 = _SSE_MUL(h1, vs_2_3);

   z1 = _SSE_SUB(_SSE_MUL(z1,h1), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)));
   z2 = _SSE_SUB(_SSE_MUL(z2,h1), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2)));
   z3 = _SSE_SUB(_SSE_MUL(z3,h1), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2)));
   z4 = _SSE_SUB(_SSE_MUL(z4,h1), _SSE_ADD(_SSE_MUL(y4,h3), _SSE_MUL(x4,h2)));

   h1 = tau4;
   h2 = _SSE_MUL(h1, vs_1_4);
   h3 = _SSE_MUL(h1, vs_2_4);
   h4 = _SSE_MUL(h1, vs_3_4);

   w1 = _SSE_SUB(_SSE_MUL(w1,h1), _SSE_ADD(_SSE_MUL(z1,h4), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));
   w2 = _SSE_SUB(_SSE_MUL(w2,h1), _SSE_ADD(_SSE_MUL(z2,h4), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2))));
   w3 = _SSE_SUB(_SSE_MUL(w3,h1), _SSE_ADD(_SSE_MUL(z3,h4), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2))));
   w4 = _SSE_SUB(_SSE_MUL(w4,h1), _SSE_ADD(_SSE_MUL(z4,h4), _SSE_ADD(_SSE_MUL(y4,h3), _SSE_MUL(x4,h2))));

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
   h2 = _SSE_MUL(tau5, vs_1_5); 
   h3 = _SSE_MUL(tau5, vs_2_5);
   h4 = _SSE_MUL(tau5, vs_3_5);
   h5 = _SSE_MUL(tau5, vs_4_5);

   v1 = _SSE_SUB(_SSE_MUL(v1,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));
   v2 = _SSE_SUB(_SSE_MUL(v2,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w2,h5), _SSE_MUL(z2,h4)), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2))));
   v3 = _SSE_SUB(_SSE_MUL(v3,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w3,h5), _SSE_MUL(z3,h4)), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2))));
   v4 = _SSE_SUB(_SSE_MUL(v4,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w4,h5), _SSE_MUL(z4,h4)), _SSE_ADD(_SSE_MUL(y4,h3), _SSE_MUL(x4,h2))));

   h2 = _SSE_MUL(tau6, vs_1_6);
   h3 = _SSE_MUL(tau6, vs_2_6);
   h4 = _SSE_MUL(tau6, vs_3_6);
   h5 = _SSE_MUL(tau6, vs_4_6);
   h6 = _SSE_MUL(tau6, vs_5_6);

   t1 = _SSE_SUB(_SSE_MUL(t1,tau6), _SSE_ADD( _SSE_MUL(v1,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)))));
   t2 = _SSE_SUB(_SSE_MUL(t2,tau6), _SSE_ADD( _SSE_MUL(v2,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w2,h5), _SSE_MUL(z2,h4)), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2)))));
   t3 = _SSE_SUB(_SSE_MUL(t3,tau6), _SSE_ADD( _SSE_MUL(v3,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w3,h5), _SSE_MUL(z3,h4)), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2)))));
   t4 = _SSE_SUB(_SSE_MUL(t4,tau6), _SSE_ADD( _SSE_MUL(v4,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w4,h5), _SSE_MUL(z4,h4)), _SSE_ADD(_SSE_MUL(y4,h3), _SSE_MUL(x4,h2)))));

   /////////////////////////////////////////////////////
   // Rank-1 update of Q [4 x nb+3]
   /////////////////////////////////////////////////////
#endif /* BLOCK6 */

   q1 = _SSE_LOAD(&q[0]);
#ifdef BLOCK2
   q1 = _SSE_ADD(q1, y1);
#endif
#ifdef BLOCK4
   q1 = _SSE_SUB(q1, w1);
#endif
#ifdef BLOCK6
   q1 = _SSE_SUB(q1, t1); 
#endif
   _SSE_STORE(&q[0],q1);
   q2 = _SSE_LOAD(&q[offset]);
#ifdef BLOCK2
   q2 = _SSE_ADD(q2, y2);
#endif
#ifdef BLOCK4
   q2 = _SSE_SUB(q2, w2);
#endif
#ifdef BLOCK6
   q2 = _SSE_SUB(q2, t2);
#endif
   _SSE_STORE(&q[offset],q2);
   q3 = _SSE_LOAD(&q[2*offset]);
#ifdef BLOCK2
   q3 = _SSE_ADD(q3, y3);
#endif
#ifdef BLOCK4
   q3 = _SSE_SUB(q3, w3);
#endif
#ifdef BLOCK6
   q3 = _SSE_SUB(q3, t3);
#endif
   _SSE_STORE(&q[2*offset],q3);
   q4 = _SSE_LOAD(&q[3*offset]);
#ifdef BLOCK2
   q4 = _SSE_ADD(q4, y4);
#endif
#ifdef BLOCK4
   q4 = _SSE_SUB(q4, w4);
#endif
#ifdef BLOCK6
   q4 = _SSE_SUB(q4, t4);
#endif

   _SSE_STORE(&q[3*offset],q4);

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
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
   q4 = _SSE_LOAD(&q[ldq+3*offset]);
   q4 = _SSE_ADD(q4, _SSE_ADD(x4, _SSE_MUL(y4, h2)));
   _SSE_STORE(&q[ldq+3*offset],q4);
#endif /* BLOCK2 */

#ifdef BLOCK4
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

   q1 = _SSE_LOAD(&q[ldq]);
   q2 = _SSE_LOAD(&q[ldq+offset]);
   q3 = _SSE_LOAD(&q[ldq+2*offset]);
   q4 = _SSE_LOAD(&q[ldq+3*offset]);

   q1 = _SSE_SUB(q1, _SSE_ADD(z1, _SSE_MUL(w1, h4)));
   q2 = _SSE_SUB(q2, _SSE_ADD(z2, _SSE_MUL(w2, h4)));
   q3 = _SSE_SUB(q3, _SSE_ADD(z3, _SSE_MUL(w3, h4)));
   q4 = _SSE_SUB(q4, _SSE_ADD(z4, _SSE_MUL(w4, h4)));

   _SSE_STORE(&q[ldq],q1);
   _SSE_STORE(&q[ldq+offset],q2);
   _SSE_STORE(&q[ldq+2*offset],q3);
   _SSE_STORE(&q[ldq+3*offset],q4);

#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+2]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

   q1 = _SSE_LOAD(&q[ldq*2]);
   q2 = _SSE_LOAD(&q[(ldq*2)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*2)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*2)+3*offset]);
   q1 = _SSE_SUB(q1, y1);
   q2 = _SSE_SUB(q2, y2);
   q3 = _SSE_SUB(q3, y3);
   q4 = _SSE_SUB(q4, y4);

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));

   _SSE_STORE(&q[ldq*2],q1);
   _SSE_STORE(&q[(ldq*2)+offset],q2);
   _SSE_STORE(&q[(ldq*2)+2*offset],q3);
   _SSE_STORE(&q[(ldq*2)+3*offset],q4);

#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+3]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

   q1 = _SSE_LOAD(&q[ldq*3]);
   q2 = _SSE_LOAD(&q[(ldq*3)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*3)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*3)+3*offset]);

   q1 = _SSE_SUB(q1, x1);
   q2 = _SSE_SUB(q2, x2);
   q3 = _SSE_SUB(q3, x3);
   q4 = _SSE_SUB(q4, x4);

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+2]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
   _SSE_STORE(&q[ldq*3], q1);
   _SSE_STORE(&q[(ldq*3)+offset], q2);
   _SSE_STORE(&q[(ldq*3)+2*offset], q3);
   _SSE_STORE(&q[(ldq*3)+3*offset], q4);

#endif /* BLOCK4 */

#ifdef BLOCK6
   
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
#endif

   q1 = _SSE_LOAD(&q[ldq]);
   q2 = _SSE_LOAD(&q[(ldq+offset)]);
   q3 = _SSE_LOAD(&q[(ldq+2*offset)]);
   q4 = _SSE_LOAD(&q[(ldq+3*offset)]);
   q1 = _SSE_SUB(q1, v1);
   q2 = _SSE_SUB(q2, v2);
   q3 = _SSE_SUB(q3, v3);
   q4 = _SSE_SUB(q4, v4);

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
   q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));

   _SSE_STORE(&q[ldq],q1);
   _SSE_STORE(&q[(ldq+offset)],q2);
   _SSE_STORE(&q[(ldq+2*offset)],q3);
   _SSE_STORE(&q[(ldq+3*offset)],q4);
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
#endif
   q1 = _SSE_LOAD(&q[ldq*2]);
   q2 = _SSE_LOAD(&q[(ldq*2)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*2)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*2)+3*offset]);
   q1 = _SSE_SUB(q1, w1); 
   q2 = _SSE_SUB(q2, w2);
   q3 = _SSE_SUB(q3, w3);
   q4 = _SSE_SUB(q4, w4);
   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5)); 
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));  
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));  
   q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));  
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
   q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));

   _SSE_STORE(&q[ldq*2],q1);
   _SSE_STORE(&q[(ldq*2)+offset],q2);
   _SSE_STORE(&q[(ldq*2)+2*offset],q3);
   _SSE_STORE(&q[(ldq*2)+3*offset],q4);

#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

   q1 = _SSE_LOAD(&q[ldq*3]);
   q2 = _SSE_LOAD(&q[(ldq*3)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*3)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*3)+3*offset]);
   q1 = _SSE_SUB(q1, z1);
   q2 = _SSE_SUB(q2, z2);
   q3 = _SSE_SUB(q3, z3);
   q4 = _SSE_SUB(q4, z4);

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
   q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
   q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));

   _SSE_STORE(&q[ldq*3],q1);
   _SSE_STORE(&q[(ldq*3)+offset],q2);
   _SSE_STORE(&q[(ldq*3)+2*offset],q3);
   _SSE_STORE(&q[(ldq*3)+3*offset],q4);
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif
   q1 = _SSE_LOAD(&q[ldq*4]);
   q2 = _SSE_LOAD(&q[(ldq*4)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*4)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*4)+3*offset]);
   q1 = _SSE_SUB(q1, y1);
   q2 = _SSE_SUB(q2, y2);
   q3 = _SSE_SUB(q3, y3);
   q4 = _SSE_SUB(q4, y4);

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
   q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+4]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
   q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));

   _SSE_STORE(&q[ldq*4],q1);
   _SSE_STORE(&q[(ldq*4)+offset],q2);
   _SSE_STORE(&q[(ldq*4)+2*offset],q3);
   _SSE_STORE(&q[(ldq*4)+3*offset],q4);
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[(ldh)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[(ldh)+1], hh[(ldh)+1]);
#endif
   q1 = _SSE_LOAD(&q[ldq*5]);
   q2 = _SSE_LOAD(&q[(ldq*5)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*5)+2*offset]);
   q4 = _SSE_LOAD(&q[(ldq*5)+3*offset]);
   q1 = _SSE_SUB(q1, x1);
   q2 = _SSE_SUB(q2, x2);
   q3 = _SSE_SUB(q3, x3);
   q4 = _SSE_SUB(q4, x4);

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+4]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
   q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+5]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
   q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));

   _SSE_STORE(&q[ldq*5],q1);
   _SSE_STORE(&q[(ldq*5)+offset],q2);
   _SSE_STORE(&q[(ldq*5)+2*offset],q3);
   _SSE_STORE(&q[(ldq*5)+3*offset],q4);

#endif /* BLOCK6 */

   for (i = BLOCK; i < nb; i++)
   {
#ifdef HAVE_SSE_INTRINSICS
     h1 = _SSE_SET1(hh[i-(BLOCK-1)]);
     h2 = _SSE_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#ifdef HAVE_SPARC64_SSE
     h1 = _SSE_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
     h2 = _SSE_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif

     q1 = _SSE_LOAD(&q[i*ldq]);
     q2 = _SSE_LOAD(&q[(i*ldq)+offset]);
     q3 = _SSE_LOAD(&q[(i*ldq)+2*offset]);
     q4 = _SSE_LOAD(&q[(i*ldq)+3*offset]);

#ifdef BLOCK2
     q1 = _SSE_ADD(q1, _SSE_ADD(_SSE_MUL(x1,h1), _SSE_MUL(y1, h2)));
     q2 = _SSE_ADD(q2, _SSE_ADD(_SSE_MUL(x2,h1), _SSE_MUL(y2, h2)));
     q3 = _SSE_ADD(q3, _SSE_ADD(_SSE_MUL(x3,h1), _SSE_MUL(y3, h2)));
     q4 = _SSE_ADD(q4, _SSE_ADD(_SSE_MUL(x4,h1), _SSE_MUL(y4, h2)));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
     
     q1 = _SSE_SUB(q1, _SSE_MUL(x1,h1));
     q2 = _SSE_SUB(q2, _SSE_MUL(x2,h1));
     q3 = _SSE_SUB(q3, _SSE_MUL(x3,h1));
     q4 = _SSE_SUB(q4, _SSE_MUL(x4,h1));

     q1 = _SSE_SUB(q1, _SSE_MUL(y1,h2));
     q2 = _SSE_SUB(q2, _SSE_MUL(y2,h2));
     q3 = _SSE_SUB(q3, _SSE_MUL(y3,h2));
     q4 = _SSE_SUB(q4, _SSE_MUL(y4,h2));

#ifdef HAVE_SSE_INTRINSICS
     h3 = _SSE_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
     h3 = _SSE_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(z1,h3));
     q2 = _SSE_SUB(q2, _SSE_MUL(z2,h3));
     q3 = _SSE_SUB(q3, _SSE_MUL(z3,h3));
     q4 = _SSE_SUB(q4, _SSE_MUL(z4,h3));

#ifdef HAVE_SSE_INTRINSICS
     h4 = _SSE_SET1(hh[(ldh*3)+i-(BLOCK-4)]); 
#endif

#ifdef HAVE_SPRC64_SSE
     h4 = _SSE_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(w1,h4));
     q2 = _SSE_SUB(q2, _SSE_MUL(w2,h4));
     q3 = _SSE_SUB(q3, _SSE_MUL(w3,h4));
     q4 = _SSE_SUB(q4, _SSE_MUL(w4,h4));

#endif /* BLOCK4 || BLOCK6*/

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
     h5 = _SSE_SET1(hh[(ldh*4)+i-1]);
#endif
#ifdef HAVE_SPARC64_SSE
     h5 = _SSE_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
     q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
     q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
     q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));
#ifdef HAVE_SSE_INTRINSICS
     h6 = _SSE_SET1(hh[(ldh*5)+i]);
#endif
#ifdef HAVE_SPARC64_SSE
     h6 = _SSE_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
     q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
     q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
     q4 = _SSE_SUB(q4, _SSE_MUL(t4, h6));
#endif /* BLOCK6 */

     _SSE_STORE(&q[i*ldq],q1);
     _SSE_STORE(&q[(i*ldq)+offset],q2);
     _SSE_STORE(&q[(i*ldq)+2*offset],q3);
     _SSE_STORE(&q[(i*ldq)+3*offset],q4);

   }
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-1)]);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif

   q1 = _SSE_LOAD(&q[nb*ldq]);
   q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);
   q3 = _SSE_LOAD(&q[(nb*ldq)+2*offset]);
   q4 = _SSE_LOAD(&q[(nb*ldq)+3*offset]);

#ifdef BLOCK2
   q1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_ADD(q4, _SSE_MUL(x4, h1));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_SUB(q4, _SSE_MUL(x4, h1));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
   q4 = _SSE_SUB(q4, _SSE_MUL(v4, h5));
#endif /* BLOCK6 */

   _SSE_STORE(&q[nb*ldq],q1);
   _SSE_STORE(&q[(nb*ldq)+offset],q2);
   _SSE_STORE(&q[(nb*ldq)+2*offset],q3);
   _SSE_STORE(&q[(nb*ldq)+3*offset],q4);

#if defined(BLOCK4) || defined(BLOCK6)
   
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-2)], hh[nb-(BLOCK-2)]);
#endif

   q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);
   q3 = _SSE_LOAD(&q[((nb+1)*ldq)+2*offset]);
   q4 = _SSE_LOAD(&q[((nb+1)*ldq)+3*offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_SUB(q4, _SSE_MUL(x4, h1));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-(BLOCK-3)], hh[ldh+nb-(BLOCK-3)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
   q4 = _SSE_SUB(q4, _SSE_MUL(w4, h4));

#endif /* BLOCK6 */

   _SSE_STORE(&q[(nb+1)*ldq],q1);
   _SSE_STORE(&q[((nb+1)*ldq)+offset],q2);
   _SSE_STORE(&q[((nb+1)*ldq)+2*offset],q3);
   _SSE_STORE(&q[((nb+1)*ldq)+3*offset],q4);

#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-3)], hh[nb-(BLOCK-3)]);
#endif

   q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);
   q3 = _SSE_LOAD(&q[((nb+2)*ldq)+2*offset]);
   q4 = _SSE_LOAD(&q[((nb+2)*ldq)+3*offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_SUB(q4, _SSE_MUL(x4, h1));

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   q4 = _SSE_SUB(q4, _SSE_MUL(z4, h3));
#endif /* BLOCK6 */
   _SSE_STORE(&q[(nb+2)*ldq],q1);
   _SSE_STORE(&q[((nb+2)*ldq)+offset],q2);
   _SSE_STORE(&q[((nb+2)*ldq)+2*offset],q3);
   _SSE_STORE(&q[((nb+2)*ldq)+3*offset],q4);

#endif /* BLOCK4 || BLOCK6*/


#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

   q1 = _SSE_LOAD(&q[(nb+3)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+3)*ldq)+offset]);
   q3 = _SSE_LOAD(&q[((nb+3)*ldq)+2*offset]);
   q4 = _SSE_LOAD(&q[((nb+3)*ldq)+3*offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_SUB(q4, _SSE_MUL(x4, h1));
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
   q4 = _SSE_SUB(q4, _SSE_MUL(y4, h2));

   _SSE_STORE(&q[(nb+3)*ldq],q1);
   _SSE_STORE(&q[((nb+3)*ldq)+offset],q2);
   _SSE_STORE(&q[((nb+3)*ldq)+2*offset],q3);
   _SSE_STORE(&q[((nb+3)*ldq)+3*offset],q4);
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif

   q1 = _SSE_LOAD(&q[(nb+4)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+4)*ldq)+offset]);
   q3 = _SSE_LOAD(&q[((nb+4)*ldq)+2*offset]);
   q4 = _SSE_LOAD(&q[((nb+4)*ldq)+3*offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));
   q4 = _SSE_SUB(q4, _SSE_MUL(x4, h1));

   _SSE_STORE(&q[(nb+4)*ldq],q1);
   _SSE_STORE(&q[((nb+4)*ldq)+offset],q2);
   _SSE_STORE(&q[((nb+4)*ldq)+2*offset],q3);
   _SSE_STORE(&q[((nb+4)*ldq)+3*offset],q4);

#endif /* BLOCK6 */
}


/*
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 6 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 12 rows of Q simultaneously, a
#endif
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
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 6
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 12
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
    // Matrix Vector Multiplication, Q [6 x nb+1] * hh
    // hh contains two householder vectors, with offset 1
    /////////////////////////////////////////////////////
#endif
#if defined(BLOCK4) || defined(BLOCK6)
    /////////////////////////////////////////////////////
    // Matrix Vector Multiplication, Q [6 x nb+3] * hh
    // hh contains four householder vectors
    /////////////////////////////////////////////////////
#endif

    int i;
#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
    // Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
    __SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SSE_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif
    __SSE_DATATYPE x1 = _SSE_LOAD(&q[ldq]);
    __SSE_DATATYPE x2 = _SSE_LOAD(&q[ldq+offset]);
    __SSE_DATATYPE x3 = _SSE_LOAD(&q[ldq+2*offset]);

#ifdef HAVE_SSE_INTRINSICS
    __SSE_DATATYPE h1 = _SSE_SET1(hh[ldh+1]);
#endif
#ifdef HAVE_SPARC64_SSE
    __SSE_DATATYPE h1 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
#endif
    __SSE_DATATYPE h2;

    __SSE_DATATYPE q1 = _SSE_LOAD(q);
    __SSE_DATATYPE y1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
    __SSE_DATATYPE q2 = _SSE_LOAD(&q[offset]);
    __SSE_DATATYPE y2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
    __SSE_DATATYPE q3 = _SSE_LOAD(&q[2*offset]);
    __SSE_DATATYPE y3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
#endif /* BLOCK2 */

#ifdef BLOCK4
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

    register __SSE_DATATYPE w1 = _SSE_ADD(a4_1, _SSE_MUL(a3_1, h_4_3));
    w1 = _SSE_ADD(w1, _SSE_MUL(a2_1, h_4_2));                          
    w1 = _SSE_ADD(w1, _SSE_MUL(a1_1, h_4_1));                          
    register __SSE_DATATYPE z1 = _SSE_ADD(a3_1, _SSE_MUL(a2_1, h_3_2));
    z1 = _SSE_ADD(z1, _SSE_MUL(a1_1, h_3_1));                          
    register __SSE_DATATYPE y1 = _SSE_ADD(a2_1, _SSE_MUL(a1_1, h_2_1));
    register __SSE_DATATYPE x1 = a1_1;                                 

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
#endif /* BLOCK4 */

#ifdef BLOCK6
    
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

    __SSE_DATATYPE a1_3 = _SSE_LOAD(&q[(ldq*5)+2*offset]);
    __SSE_DATATYPE a2_3 = _SSE_LOAD(&q[(ldq*4)+2*offset]);
    __SSE_DATATYPE a3_3 = _SSE_LOAD(&q[(ldq*3)+2*offset]);
    __SSE_DATATYPE a4_3 = _SSE_LOAD(&q[(ldq*2)+2*offset]);
    __SSE_DATATYPE a5_3 = _SSE_LOAD(&q[(ldq)+2*offset]);
    __SSE_DATATYPE a6_3 = _SSE_LOAD(&q[2*offset]);

    register __SSE_DATATYPE t3 = _SSE_ADD(a6_3, _SSE_MUL(a5_3, h_6_5));
    t3 = _SSE_ADD(t3, _SSE_MUL(a4_3, h_6_4));
    t3 = _SSE_ADD(t3, _SSE_MUL(a3_3, h_6_3));
    t3 = _SSE_ADD(t3, _SSE_MUL(a2_3, h_6_2));
    t3 = _SSE_ADD(t3, _SSE_MUL(a1_3, h_6_1));
    register __SSE_DATATYPE v3 = _SSE_ADD(a5_3, _SSE_MUL(a4_3, h_5_4));
    v3 = _SSE_ADD(v3, _SSE_MUL(a3_3, h_5_3));
    v3 = _SSE_ADD(v3, _SSE_MUL(a2_3, h_5_2));
    v3 = _SSE_ADD(v3, _SSE_MUL(a1_3, h_5_1));
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
    __SSE_DATATYPE h5;
    __SSE_DATATYPE h6;

#endif /* BLOCK6 */


    for(i = BLOCK; i < nb; i++)
      {
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[i-(BLOCK-1)]);
        h2 = _SSE_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
        h2 = _SSE_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
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

#if defined(BLOCK4) || defined(BLOCK6)
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

        z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
        z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
        z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));
#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

        w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
        w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));
        w3 = _SSE_ADD(w3, _SSE_MUL(q3,h4));
	
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+i-1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif
        v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));
        v2 = _SSE_ADD(v2, _SSE_MUL(q2,h5));
        v3 = _SSE_ADD(v3, _SSE_MUL(q3,h5));

#ifdef HAVE_SSE_INTRINSICS
        h6 = _SSE_SET1(hh[(ldh*5)+i]);
#endif

#ifdef HAVE_SPARC64_SSE
        h6 = _SSE_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif

        t1 = _SSE_ADD(t1, _SSE_MUL(q1,h6));
        t2 = _SSE_ADD(t2, _SSE_MUL(q2,h6));
        t3 = _SSE_ADD(t3, _SSE_MUL(q3,h6));
	
#endif /* BLOCK6 */
      }
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-(BLOCK-1)]);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif

    q1 = _SSE_LOAD(&q[nb*ldq]);
    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    q3 = _SSE_LOAD(&q[(nb*ldq)+2*offset]);
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));

#if defined(BLOCK4) || defined(BLOCK6)
    
#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));

#ifdef HAVE_SSE_INTRINSICS
    h3 = _SSE_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
    h3 = _SSE_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

    z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
    z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
    z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));

#ifdef BLOCK4
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

    q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+1)*ldq)+2*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));

#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[(ldh*1)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[(ldh*1)+nb-1], hh[(ldh*1)+nb-1]);
#endif


    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));

#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif


    q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+2)*ldq)+2*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));

#endif /* BLOCK4 */
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#ifdef HAVE_SSE_INTRINSICS
    h4 = _SSE_SET1(hh[(ldh*3)+nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h4 = _SSE_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif

    w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4)); 
    w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));
    w3 = _SSE_ADD(w3, _SSE_MUL(q3,h4));

#ifdef HAVE_SSE_INTRINSICS
    h5 = _SSE_SET1(hh[(ldh*4)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h5 = _SSE_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif

    v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));
    v2 = _SSE_ADD(v2, _SSE_MUL(q2,h5));
    v3 = _SSE_ADD(v3, _SSE_MUL(q3,h5));
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-4]);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-4], hh[nb-4]);
#endif

    q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+1)*ldq)+2*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-3]);
#endif
#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-3], hh[ldh+nb-3]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));

#ifdef HAVE_SSE_INTRINSICS
    h3 = _SSE_SET1(hh[(ldh*2)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
    h3 = _SSE_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif

    z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
    z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
    z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));

#ifdef HAVE_SSE_INTRINSICS
    h4 = _SSE_SET1(hh[(ldh*3)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
    h4 = _SSE_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif

    w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
    w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));
    w3 = _SSE_ADD(w3, _SSE_MUL(q3,h4));

#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-3]);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-3], hh[nb-3]);
#endif

    q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+2)*ldq)+2*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));

#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
#ifdef HAVE_SSE_INTRINSICS
    h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif

    z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
    z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
    z3 = _SSE_ADD(z3, _SSE_MUL(q3,h3));
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

    q1 = _SSE_LOAD(&q[(nb+3)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+3)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+3)*ldq)+2*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));

#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
    y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));

#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif

    q1 = _SSE_LOAD(&q[(nb+4)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+4)*ldq)+offset]);
    q3 = _SSE_LOAD(&q[((nb+4)*ldq)+2*offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
    x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
#endif /* BLOCK6 */

#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Rank-2 update of Q [6 x nb+1]
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

#ifdef HAVE_SSE_INTRINSICS
    __SSE_DATATYPE tau1 = _SSE_SET1(hh[0]);
    __SSE_DATATYPE tau2 = _SSE_SET1(hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SSE_DATATYPE tau3 = _SSE_SET1(hh[ldh*2]);
   __SSE_DATATYPE tau4 = _SSE_SET1(hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE tau5 = _SSE_SET1(hh[ldh*4]);
   __SSE_DATATYPE tau6 = _SSE_SET1(hh[ldh*5]);       
#endif

#ifdef BLOCK2    
    __SSE_DATATYPE vs = _SSE_SET1(s);
#endif
#ifdef BLOCK4
   __SSE_DATATYPE vs_1_2 = _SSE_SET1(s_1_2);
   __SSE_DATATYPE vs_1_3 = _SSE_SET1(s_1_3);  
   __SSE_DATATYPE vs_2_3 = _SSE_SET1(s_2_3);
   __SSE_DATATYPE vs_1_4 = _SSE_SET1(s_1_4);  
   __SSE_DATATYPE vs_2_4 = _SSE_SET1(s_2_4);  
   __SSE_DATATYPE vs_3_4 = _SSE_SET1(s_3_4);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE vs_1_2 = _SSE_SET1(scalarprods[0]);
   __SSE_DATATYPE vs_1_3 = _SSE_SET1(scalarprods[1]);
   __SSE_DATATYPE vs_2_3 = _SSE_SET1(scalarprods[2]);
   __SSE_DATATYPE vs_1_4 = _SSE_SET1(scalarprods[3]);
   __SSE_DATATYPE vs_2_4 = _SSE_SET1(scalarprods[4]);
   __SSE_DATATYPE vs_3_4 = _SSE_SET1(scalarprods[5]);
   __SSE_DATATYPE vs_1_5 = _SSE_SET1(scalarprods[6]);
   __SSE_DATATYPE vs_2_5 = _SSE_SET1(scalarprods[7]);
   __SSE_DATATYPE vs_3_5 = _SSE_SET1(scalarprods[8]);
   __SSE_DATATYPE vs_4_5 = _SSE_SET1(scalarprods[9]);
   __SSE_DATATYPE vs_1_6 = _SSE_SET1(scalarprods[10]);
   __SSE_DATATYPE vs_2_6 = _SSE_SET1(scalarprods[11]);
   __SSE_DATATYPE vs_3_6 = _SSE_SET1(scalarprods[12]);
   __SSE_DATATYPE vs_4_6 = _SSE_SET1(scalarprods[13]);
   __SSE_DATATYPE vs_5_6 = _SSE_SET1(scalarprods[14]);
#endif
#endif /* HAVE_SSE_INTRINSICS */

#ifdef HAVE_SPARC64_SSE
    __SSE_DATATYPE tau1 = _SSE_SET(hh[0], hh[0]);
    __SSE_DATATYPE tau2 = _SSE_SET(hh[ldh], hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SSE_DATATYPE tau3 = _SSE_SET(hh[ldh*2], hh[ldh*2]);
   __SSE_DATATYPE tau4 = _SSE_SET(hh[ldh*3], hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE tau5 = _SSE_SET(hh[ldh*4], hh[ldh*4]);
   __SSE_DATATYPE tau6 = _SSE_SET(hh[ldh*5], hh[ldh*5]);
#endif

#ifdef BLOCK2
    __SSE_DATATYPE vs = _SSE_SET(s, s);
#endif
#ifdef BLOCK4
   __SSE_DATATYPE vs_1_2 = _SSE_SET(s_1_2, s_1_2);
   __SSE_DATATYPE vs_1_3 = _SSE_SET(s_1_3, s_1_3);
   __SSE_DATATYPE vs_2_3 = _SSE_SET(s_2_3, s_2_3);
   __SSE_DATATYPE vs_1_4 = _SSE_SET(s_1_4, s_1_4);
   __SSE_DATATYPE vs_2_4 = _SSE_SET(s_2_4, s_2_4);
   __SSE_DATATYPE vs_3_4 = _SSE_SET(s_3_4, s_3_4);

#endif
#ifdef BLOCK6
   __SSE_DATATYPE vs_1_2 = _SSE_SET(scalarprods[0], scalarprods[0]);
   __SSE_DATATYPE vs_1_3 = _SSE_SET(scalarprods[1], scalarprods[1]);
   __SSE_DATATYPE vs_2_3 = _SSE_SET(scalarprods[2], scalarprods[2]);
   __SSE_DATATYPE vs_1_4 = _SSE_SET(scalarprods[3], scalarprods[3]);
   __SSE_DATATYPE vs_2_4 = _SSE_SET(scalarprods[4], scalarprods[4]);
   __SSE_DATATYPE vs_3_4 = _SSE_SET(scalarprods[5], scalarprods[5]);
   __SSE_DATATYPE vs_1_5 = _SSE_SET(scalarprods[6], scalarprods[6]);
   __SSE_DATATYPE vs_2_5 = _SSE_SET(scalarprods[7], scalarprods[7]);
   __SSE_DATATYPE vs_3_5 = _SSE_SET(scalarprods[8], scalarprods[8]);
   __SSE_DATATYPE vs_4_5 = _SSE_SET(scalarprods[9], scalarprods[9]);
   __SSE_DATATYPE vs_1_6 = _SSE_SET(scalarprods[10], scalarprods[10]);
   __SSE_DATATYPE vs_2_6 = _SSE_SET(scalarprods[11], scalarprods[11]);
   __SSE_DATATYPE vs_3_6 = _SSE_SET(scalarprods[12], scalarprods[12]);
   __SSE_DATATYPE vs_4_6 = _SSE_SET(scalarprods[13], scalarprods[13]);
   __SSE_DATATYPE vs_5_6 = _SSE_SET(scalarprods[14], scalarprods[14]);
#endif
#endif /* HAVE_SPARC64_SSE */

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_XOR(tau1, sign);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _fjsp_neg_v2r8(tau1);
#endif
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau1;
#endif

   x1 = _SSE_MUL(x1, h1);
   x2 = _SSE_MUL(x2, h1);
   x3 = _SSE_MUL(x3, h1);

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_XOR(tau2, sign);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _fjsp_neg_v2r8(tau2);
#endif
   h2 = _SSE_MUL(h1, vs);
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau2;
   h2 = _SSE_MUL(h1, vs_1_2);
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK2
   y1 = _SSE_ADD(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
   y2 = _SSE_ADD(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
   y3 = _SSE_ADD(_SSE_MUL(y3,h1), _SSE_MUL(x3,h2));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   y1 = _SSE_SUB(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
   y2 = _SSE_SUB(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
   y3 = _SSE_SUB(_SSE_MUL(y3,h1), _SSE_MUL(x3,h2));

   h1 = tau3;
   h2 = _SSE_MUL(h1, vs_1_3);
   h3 = _SSE_MUL(h1, vs_2_3);

   z1 = _SSE_SUB(_SSE_MUL(z1,h1), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)));
   z2 = _SSE_SUB(_SSE_MUL(z2,h1), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2)));
   z3 = _SSE_SUB(_SSE_MUL(z3,h1), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2)));

   h1 = tau4;
   h2 = _SSE_MUL(h1, vs_1_4);
   h3 = _SSE_MUL(h1, vs_2_4);
   h4 = _SSE_MUL(h1, vs_3_4);

   w1 = _SSE_SUB(_SSE_MUL(w1,h1), _SSE_ADD(_SSE_MUL(z1,h4), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)))); 
   w2 = _SSE_SUB(_SSE_MUL(w2,h1), _SSE_ADD(_SSE_MUL(z2,h4), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2))));
   w3 = _SSE_SUB(_SSE_MUL(w3,h1), _SSE_ADD(_SSE_MUL(z3,h4), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2))));

#endif /* BLOCK4 || BLOCK6*/

#ifdef BLOCK6
   h2 = _SSE_MUL(tau5, vs_1_5); 
   h3 = _SSE_MUL(tau5, vs_2_5);
   h4 = _SSE_MUL(tau5, vs_3_5);
   h5 = _SSE_MUL(tau5, vs_4_5);

   v1 = _SSE_SUB(_SSE_MUL(v1,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));
   v2 = _SSE_SUB(_SSE_MUL(v2,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w2,h5), _SSE_MUL(z2,h4)), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2))));
   v3 = _SSE_SUB(_SSE_MUL(v3,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w3,h5), _SSE_MUL(z3,h4)), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2))));

   h2 = _SSE_MUL(tau6, vs_1_6);
   h3 = _SSE_MUL(tau6, vs_2_6);
   h4 = _SSE_MUL(tau6, vs_3_6);
   h5 = _SSE_MUL(tau6, vs_4_6);
   h6 = _SSE_MUL(tau6, vs_5_6);

   t1 = _SSE_SUB(_SSE_MUL(t1,tau6), _SSE_ADD( _SSE_MUL(v1,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)))));
   t2 = _SSE_SUB(_SSE_MUL(t2,tau6), _SSE_ADD( _SSE_MUL(v2,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w2,h5), _SSE_MUL(z2,h4)), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2)))));
   t3 = _SSE_SUB(_SSE_MUL(t3,tau6), _SSE_ADD( _SSE_MUL(v3,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w3,h5), _SSE_MUL(z3,h4)), _SSE_ADD(_SSE_MUL(y3,h3), _SSE_MUL(x3,h2)))));

   /////////////////////////////////////////////////////
   // Rank-1 update of Q [4 x nb+3]
   /////////////////////////////////////////////////////
#endif /* BLOCK6 */

   q1 = _SSE_LOAD(&q[0]);
#ifdef BLOCK2
   q1 = _SSE_ADD(q1, y1);
#endif
#ifdef BLOCK4
   q1 = _SSE_SUB(q1, w1);
#endif
#ifdef BLOCK6
   q1 = _SSE_SUB(q1, t1); 
#endif
   _SSE_STORE(&q[0],q1);
   q2 = _SSE_LOAD(&q[offset]);
#ifdef BLOCK2
   q2 = _SSE_ADD(q2, y2);
#endif
#ifdef BLOCK4
   q2 = _SSE_SUB(q2, w2);
#endif
#ifdef BLOCK6
   q2 = _SSE_SUB(q2, t2);
#endif
   _SSE_STORE(&q[offset],q2);
   q3 = _SSE_LOAD(&q[2*offset]);
#ifdef BLOCK2
   q3 = _SSE_ADD(q3, y3);
#endif
#ifdef BLOCK4
   q3 = _SSE_SUB(q3, w3);
#endif
#ifdef BLOCK6
   q3 = _SSE_SUB(q3, t3);
#endif

   _SSE_STORE(&q[2*offset],q3);

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
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
#endif /* BLOCK2 */

#ifdef BLOCK4
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

   q1 = _SSE_LOAD(&q[ldq]);
   q2 = _SSE_LOAD(&q[ldq+offset]);
   q3 = _SSE_LOAD(&q[ldq+2*offset]);

   q1 = _SSE_SUB(q1, _SSE_ADD(z1, _SSE_MUL(w1, h4)));
   q2 = _SSE_SUB(q2, _SSE_ADD(z2, _SSE_MUL(w2, h4)));
   q3 = _SSE_SUB(q3, _SSE_ADD(z3, _SSE_MUL(w3, h4)));

   _SSE_STORE(&q[ldq],q1);
   _SSE_STORE(&q[ldq+offset],q2);
   _SSE_STORE(&q[ldq+2*offset],q3);

#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+2]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

   q1 = _SSE_LOAD(&q[ldq*2]);
   q2 = _SSE_LOAD(&q[(ldq*2)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*2)+2*offset]);
   q1 = _SSE_SUB(q1, y1);
   q2 = _SSE_SUB(q2, y2);
   q3 = _SSE_SUB(q3, y3);

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));

   _SSE_STORE(&q[ldq*2],q1);
   _SSE_STORE(&q[(ldq*2)+offset],q2);
   _SSE_STORE(&q[(ldq*2)+2*offset],q3);

#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+3]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

   q1 = _SSE_LOAD(&q[ldq*3]);
   q2 = _SSE_LOAD(&q[(ldq*3)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*3)+2*offset]);
   q1 = _SSE_SUB(q1, x1);
   q2 = _SSE_SUB(q2, x2);
   q3 = _SSE_SUB(q3, x3);

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+2]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
   _SSE_STORE(&q[ldq*3], q1);
   _SSE_STORE(&q[(ldq*3)+offset], q2);
   _SSE_STORE(&q[(ldq*3)+2*offset], q3);

#endif /* BLOCK4 */

#ifdef BLOCK6
   
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
#endif

   q1 = _SSE_LOAD(&q[ldq]);
   q2 = _SSE_LOAD(&q[(ldq+offset)]);
   q3 = _SSE_LOAD(&q[(ldq+2*offset)]);
   q1 = _SSE_SUB(q1, v1);
   q2 = _SSE_SUB(q2, v2);
   q3 = _SSE_SUB(q3, v3);

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));

   _SSE_STORE(&q[ldq],q1);
   _SSE_STORE(&q[(ldq+offset)],q2);
   _SSE_STORE(&q[(ldq+2*offset)],q3);
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
#endif
   q1 = _SSE_LOAD(&q[ldq*2]);
   q2 = _SSE_LOAD(&q[(ldq*2)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*2)+2*offset]);
   q1 = _SSE_SUB(q1, w1); 
   q2 = _SSE_SUB(q2, w2);
   q3 = _SSE_SUB(q3, w3);
   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5)); 
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));  
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));  
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));

   _SSE_STORE(&q[ldq*2],q1);
   _SSE_STORE(&q[(ldq*2)+offset],q2);
   _SSE_STORE(&q[(ldq*2)+2*offset],q3);

#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

   q1 = _SSE_LOAD(&q[ldq*3]);
   q2 = _SSE_LOAD(&q[(ldq*3)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*3)+2*offset]);
   q1 = _SSE_SUB(q1, z1);
   q2 = _SSE_SUB(q2, z2);
   q3 = _SSE_SUB(q3, z3);

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));

   _SSE_STORE(&q[ldq*3],q1);
   _SSE_STORE(&q[(ldq*3)+offset],q2);
   _SSE_STORE(&q[(ldq*3)+2*offset],q3);
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif
   q1 = _SSE_LOAD(&q[ldq*4]);
   q2 = _SSE_LOAD(&q[(ldq*4)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*4)+2*offset]);
   q1 = _SSE_SUB(q1, y1);
   q2 = _SSE_SUB(q2, y2);
   q3 = _SSE_SUB(q3, y3);

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+4]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));

   _SSE_STORE(&q[ldq*4],q1);
   _SSE_STORE(&q[(ldq*4)+offset],q2);
   _SSE_STORE(&q[(ldq*4)+2*offset],q3);
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[(ldh)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[(ldh)+1], hh[(ldh)+1]);
#endif
   q1 = _SSE_LOAD(&q[ldq*5]);
   q2 = _SSE_LOAD(&q[(ldq*5)+offset]);
   q3 = _SSE_LOAD(&q[(ldq*5)+2*offset]);
   q1 = _SSE_SUB(q1, x1);
   q2 = _SSE_SUB(q2, x2);
   q3 = _SSE_SUB(q3, x3);

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+4]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+5]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
   q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));

   _SSE_STORE(&q[ldq*5],q1);
   _SSE_STORE(&q[(ldq*5)+offset],q2);
   _SSE_STORE(&q[(ldq*5)+2*offset],q3);

#endif /* BLOCK6 */

   for (i = BLOCK; i < nb; i++)
   {
#ifdef HAVE_SSE_INTRINSICS
     h1 = _SSE_SET1(hh[i-(BLOCK-1)]);
     h2 = _SSE_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#ifdef HAVE_SPARC64_SSE
     h1 = _SSE_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
     h2 = _SSE_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif

     q1 = _SSE_LOAD(&q[i*ldq]);
     q2 = _SSE_LOAD(&q[(i*ldq)+offset]);
     q3 = _SSE_LOAD(&q[(i*ldq)+2*offset]);

#ifdef BLOCK2
     q1 = _SSE_ADD(q1, _SSE_ADD(_SSE_MUL(x1,h1), _SSE_MUL(y1, h2)));
     q2 = _SSE_ADD(q2, _SSE_ADD(_SSE_MUL(x2,h1), _SSE_MUL(y2, h2)));
     q3 = _SSE_ADD(q3, _SSE_ADD(_SSE_MUL(x3,h1), _SSE_MUL(y3, h2)));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
     
     q1 = _SSE_SUB(q1, _SSE_MUL(x1,h1));
     q2 = _SSE_SUB(q2, _SSE_MUL(x2,h1));
     q3 = _SSE_SUB(q3, _SSE_MUL(x3,h1));

     q1 = _SSE_SUB(q1, _SSE_MUL(y1,h2));
     q2 = _SSE_SUB(q2, _SSE_MUL(y2,h2));
     q3 = _SSE_SUB(q3, _SSE_MUL(y3,h2));

#ifdef HAVE_SSE_INTRINSICS
     h3 = _SSE_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
     h3 = _SSE_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(z1,h3));
     q2 = _SSE_SUB(q2, _SSE_MUL(z2,h3));
     q3 = _SSE_SUB(q3, _SSE_MUL(z3,h3));

#ifdef HAVE_SSE_INTRINSICS
     h4 = _SSE_SET1(hh[(ldh*3)+i-(BLOCK-4)]); 
#endif

#ifdef HAVE_SPRC64_SSE
     h4 = _SSE_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(w1,h4));
     q2 = _SSE_SUB(q2, _SSE_MUL(w2,h4));
     q3 = _SSE_SUB(q3, _SSE_MUL(w3,h4));

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
     h5 = _SSE_SET1(hh[(ldh*4)+i-1]);
#endif
#ifdef HAVE_SPARC64_SSE
     h5 = _SSE_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
     q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
     q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
#ifdef HAVE_SSE_INTRINSICS
     h6 = _SSE_SET1(hh[(ldh*5)+i]);
#endif
#ifdef HAVE_SPARC64_SSE
     h6 = _SSE_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
     q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
     q3 = _SSE_SUB(q3, _SSE_MUL(t3, h6));
#endif /* BLOCK6 */
     _SSE_STORE(&q[i*ldq],q1);
     _SSE_STORE(&q[(i*ldq)+offset],q2);
     _SSE_STORE(&q[(i*ldq)+2*offset],q3);

   }
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-1)]);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif

   q1 = _SSE_LOAD(&q[nb*ldq]);
   q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);
   q3 = _SSE_LOAD(&q[(nb*ldq)+2*offset]);

#ifdef BLOCK2
   q1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
   q3 = _SSE_SUB(q3, _SSE_MUL(v3, h5));
#endif /* BLOCK6 */

   _SSE_STORE(&q[nb*ldq],q1);
   _SSE_STORE(&q[(nb*ldq)+offset],q2);
   _SSE_STORE(&q[(nb*ldq)+2*offset],q3);

#if defined(BLOCK4) || defined(BLOCK6)
   
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-2)], hh[nb-(BLOCK-2)]);
#endif

   q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);
   q3 = _SSE_LOAD(&q[((nb+1)*ldq)+2*offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-(BLOCK-3)], hh[ldh+nb-(BLOCK-3)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
   q3 = _SSE_SUB(q3, _SSE_MUL(w3, h4));

#endif /* BLOCK6 */

   _SSE_STORE(&q[(nb+1)*ldq],q1);
   _SSE_STORE(&q[((nb+1)*ldq)+offset],q2);
   _SSE_STORE(&q[((nb+1)*ldq)+2*offset],q3);

#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-3)], hh[nb-(BLOCK-3)]);
#endif

   q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);
   q3 = _SSE_LOAD(&q[((nb+2)*ldq)+2*offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   q3 = _SSE_SUB(q3, _SSE_MUL(z3, h3));
#endif /* BLOCK6 */

   _SSE_STORE(&q[(nb+2)*ldq],q1);
   _SSE_STORE(&q[((nb+2)*ldq)+offset],q2);
   _SSE_STORE(&q[((nb+2)*ldq)+2*offset],q3);

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

   q1 = _SSE_LOAD(&q[(nb+3)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+3)*ldq)+offset]);
   q3 = _SSE_LOAD(&q[((nb+3)*ldq)+2*offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
   q3 = _SSE_SUB(q3, _SSE_MUL(y3, h2));

   _SSE_STORE(&q[(nb+3)*ldq],q1);
   _SSE_STORE(&q[((nb+3)*ldq)+offset],q2);
   _SSE_STORE(&q[((nb+3)*ldq)+2*offset],q3);
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif

   q1 = _SSE_LOAD(&q[(nb+4)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+4)*ldq)+offset]);
   q3 = _SSE_LOAD(&q[((nb+4)*ldq)+2*offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
   q3 = _SSE_SUB(q3, _SSE_MUL(x3, h1));

   _SSE_STORE(&q[(nb+4)*ldq],q1);
   _SSE_STORE(&q[((nb+4)*ldq)+offset],q2);
   _SSE_STORE(&q[((nb+4)*ldq)+2*offset],q3);

#endif /* BLOCK6 */
}


/*
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 4 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 8 rows of Q simultaneously, a
#endif
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
#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 4
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 8
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
    // Matrix Vector Multiplication, Q [4 x nb+1] * hh
    // hh contains two householder vectors, with offset 1
    /////////////////////////////////////////////////////
#endif
#if defined(BLOCK4) || defined(BLOCK6)
    /////////////////////////////////////////////////////
    // Matrix Vector Multiplication, Q [4 x nb+3] * hh
    // hh contains four householder vectors
    /////////////////////////////////////////////////////
#endif

    int i;
#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
    // Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
    __SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SSE_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif
    __SSE_DATATYPE x1 = _SSE_LOAD(&q[ldq]);
    __SSE_DATATYPE x2 = _SSE_LOAD(&q[ldq+offset]);

#ifdef HAVE_SSE_INTRINSICS
    __SSE_DATATYPE h1 = _SSE_SET1(hh[ldh+1]);
#endif
#ifdef HAVE_SPARC64_SSE
    __SSE_DATATYPE h1 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
#endif
    __SSE_DATATYPE h2;

    __SSE_DATATYPE q1 = _SSE_LOAD(q);
    __SSE_DATATYPE y1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
    __SSE_DATATYPE q2 = _SSE_LOAD(&q[offset]);
    __SSE_DATATYPE y2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
#endif /* BLOCK2 */

#ifdef BLOCK4
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

    register __SSE_DATATYPE w1 = _SSE_ADD(a4_1, _SSE_MUL(a3_1, h_4_3));
    w1 = _SSE_ADD(w1, _SSE_MUL(a2_1, h_4_2));                          
    w1 = _SSE_ADD(w1, _SSE_MUL(a1_1, h_4_1));                          
    register __SSE_DATATYPE z1 = _SSE_ADD(a3_1, _SSE_MUL(a2_1, h_3_2));
    z1 = _SSE_ADD(z1, _SSE_MUL(a1_1, h_3_1));                          
    register __SSE_DATATYPE y1 = _SSE_ADD(a2_1, _SSE_MUL(a1_1, h_2_1));
    register __SSE_DATATYPE x1 = a1_1;

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

    __SSE_DATATYPE q1;
    __SSE_DATATYPE q2;

    __SSE_DATATYPE h1;
    __SSE_DATATYPE h2;
    __SSE_DATATYPE h3;
    __SSE_DATATYPE h4;
#endif /* BLOCK4 */

#ifdef BLOCK6
    
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

#endif /* BLOCK6 */

    for(i = BLOCK; i < nb; i++)
      {
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[i-(BLOCK-1)]);
        h2 = _SSE_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
        h2 = _SSE_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif

        q1 = _SSE_LOAD(&q[i*ldq]);
        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
        y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
        q2 = _SSE_LOAD(&q[(i*ldq)+offset]);
        x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
        y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));

#if defined(BLOCK4) || defined(BLOCK6)
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

        z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
        z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));

#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

        w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
        w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+i-1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif
        v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));
        v2 = _SSE_ADD(v2, _SSE_MUL(q2,h5));

#ifdef HAVE_SSE_INTRINSICS
        h6 = _SSE_SET1(hh[(ldh*5)+i]);
#endif

#ifdef HAVE_SPARC64_SSE
        h6 = _SSE_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif

        t1 = _SSE_ADD(t1, _SSE_MUL(q1,h6));
        t2 = _SSE_ADD(t2, _SSE_MUL(q2,h6));
	
#endif /* BLOCK6 */
      }
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-(BLOCK-1)]);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif

    q1 = _SSE_LOAD(&q[nb*ldq]);
    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));

#if defined(BLOCK4) || defined(BLOCK6)
    
#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));

#ifdef HAVE_SSE_INTRINSICS
    h3 = _SSE_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
    h3 = _SSE_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

    z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
    z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));

#ifdef BLOCK4

#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

    q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));

#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[(ldh*1)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[(ldh*1)+nb-1], hh[(ldh*1)+nb-1]);
#endif

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
#endif /* BLOCK4 */
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#ifdef HAVE_SSE_INTRINSICS
    h4 = _SSE_SET1(hh[(ldh*3)+nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h4 = _SSE_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif

    w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4)); 
    w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));

#ifdef HAVE_SSE_INTRINSICS
    h5 = _SSE_SET1(hh[(ldh*4)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h5 = _SSE_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif

    v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));
    v2 = _SSE_ADD(v2, _SSE_MUL(q2,h5));
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-4]);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-4], hh[nb-4]);
#endif

    q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-3]);
#endif
#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-3], hh[ldh+nb-3]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));

#ifdef HAVE_SSE_INTRINSICS
    h3 = _SSE_SET1(hh[(ldh*2)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
    h3 = _SSE_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif

    z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
    z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));

#ifdef HAVE_SSE_INTRINSICS
    h4 = _SSE_SET1(hh[(ldh*3)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
    h4 = _SSE_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif

    w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
    w2 = _SSE_ADD(w2, _SSE_MUL(q2,h4));

#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-3]);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-3], hh[nb-3]);
#endif

    q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));

#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
#ifdef HAVE_SSE_INTRINSICS
    h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif

    z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));
    z2 = _SSE_ADD(z2, _SSE_MUL(q2,h3));
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

    q1 = _SSE_LOAD(&q[(nb+3)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+3)*ldq)+offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));

#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
    y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));

#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif

    q1 = _SSE_LOAD(&q[(nb+4)*ldq]);
    q2 = _SSE_LOAD(&q[((nb+4)*ldq)+offset]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
    x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
#endif /* BLOCK6 */

#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Rank-2 update of Q [4 x nb+1]
    /////////////////////////////////////////////////////
#endif
#ifdef BLOCK4
    /////////////////////////////////////////////////////
    // Rank-1 update of Q [4 x nb+3]
    /////////////////////////////////////////////////////
#endif
#ifdef BLOCK6
    /////////////////////////////////////////////////////
    // Apply tau, correct wrong calculation using pre-calculated scalar products
    /////////////////////////////////////////////////////
#endif

#ifdef HAVE_SSE_INTRINSICS
    __SSE_DATATYPE tau1 = _SSE_SET1(hh[0]);
    __SSE_DATATYPE tau2 = _SSE_SET1(hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SSE_DATATYPE tau3 = _SSE_SET1(hh[ldh*2]);
   __SSE_DATATYPE tau4 = _SSE_SET1(hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE tau5 = _SSE_SET1(hh[ldh*4]);
   __SSE_DATATYPE tau6 = _SSE_SET1(hh[ldh*5]);       
#endif

#ifdef BLOCK2    
    __SSE_DATATYPE vs = _SSE_SET1(s);
#endif
#ifdef BLOCK4
   __SSE_DATATYPE vs_1_2 = _SSE_SET1(s_1_2);
   __SSE_DATATYPE vs_1_3 = _SSE_SET1(s_1_3);  
   __SSE_DATATYPE vs_2_3 = _SSE_SET1(s_2_3);
   __SSE_DATATYPE vs_1_4 = _SSE_SET1(s_1_4);  
   __SSE_DATATYPE vs_2_4 = _SSE_SET1(s_2_4);  
   __SSE_DATATYPE vs_3_4 = _SSE_SET1(s_3_4);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE vs_1_2 = _SSE_SET1(scalarprods[0]);
   __SSE_DATATYPE vs_1_3 = _SSE_SET1(scalarprods[1]);
   __SSE_DATATYPE vs_2_3 = _SSE_SET1(scalarprods[2]);
   __SSE_DATATYPE vs_1_4 = _SSE_SET1(scalarprods[3]);
   __SSE_DATATYPE vs_2_4 = _SSE_SET1(scalarprods[4]);
   __SSE_DATATYPE vs_3_4 = _SSE_SET1(scalarprods[5]);
   __SSE_DATATYPE vs_1_5 = _SSE_SET1(scalarprods[6]);
   __SSE_DATATYPE vs_2_5 = _SSE_SET1(scalarprods[7]);
   __SSE_DATATYPE vs_3_5 = _SSE_SET1(scalarprods[8]);
   __SSE_DATATYPE vs_4_5 = _SSE_SET1(scalarprods[9]);
   __SSE_DATATYPE vs_1_6 = _SSE_SET1(scalarprods[10]);
   __SSE_DATATYPE vs_2_6 = _SSE_SET1(scalarprods[11]);
   __SSE_DATATYPE vs_3_6 = _SSE_SET1(scalarprods[12]);
   __SSE_DATATYPE vs_4_6 = _SSE_SET1(scalarprods[13]);
   __SSE_DATATYPE vs_5_6 = _SSE_SET1(scalarprods[14]);
#endif
#endif /* HAVE_SSE_INTRINSICS */

#ifdef HAVE_SPARC64_SSE
    __SSE_DATATYPE tau1 = _SSE_SET(hh[0], hh[0]);
    __SSE_DATATYPE tau2 = _SSE_SET(hh[ldh], hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SSE_DATATYPE tau3 = _SSE_SET(hh[ldh*2], hh[ldh*2]);
   __SSE_DATATYPE tau4 = _SSE_SET(hh[ldh*3], hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE tau5 = _SSE_SET(hh[ldh*4], hh[ldh*4]);
   __SSE_DATATYPE tau6 = _SSE_SET(hh[ldh*5], hh[ldh*5]);
#endif

#ifdef BLOCK2
    __SSE_DATATYPE vs = _SSE_SET(s, s);
#endif
#ifdef BLOCK4
   __SSE_DATATYPE vs_1_2 = _SSE_SET(s_1_2, s_1_2);
   __SSE_DATATYPE vs_1_3 = _SSE_SET(s_1_3, s_1_3);
   __SSE_DATATYPE vs_2_3 = _SSE_SET(s_2_3, s_2_3);
   __SSE_DATATYPE vs_1_4 = _SSE_SET(s_1_4, s_1_4);
   __SSE_DATATYPE vs_2_4 = _SSE_SET(s_2_4, s_2_4);
   __SSE_DATATYPE vs_3_4 = _SSE_SET(s_3_4, s_3_4);

#endif
#ifdef BLOCK6
   __SSE_DATATYPE vs_1_2 = _SSE_SET(scalarprods[0], scalarprods[0]);
   __SSE_DATATYPE vs_1_3 = _SSE_SET(scalarprods[1], scalarprods[1]);
   __SSE_DATATYPE vs_2_3 = _SSE_SET(scalarprods[2], scalarprods[2]);
   __SSE_DATATYPE vs_1_4 = _SSE_SET(scalarprods[3], scalarprods[3]);
   __SSE_DATATYPE vs_2_4 = _SSE_SET(scalarprods[4], scalarprods[4]);
   __SSE_DATATYPE vs_3_4 = _SSE_SET(scalarprods[5], scalarprods[5]);
   __SSE_DATATYPE vs_1_5 = _SSE_SET(scalarprods[6], scalarprods[6]);
   __SSE_DATATYPE vs_2_5 = _SSE_SET(scalarprods[7], scalarprods[7]);
   __SSE_DATATYPE vs_3_5 = _SSE_SET(scalarprods[8], scalarprods[8]);
   __SSE_DATATYPE vs_4_5 = _SSE_SET(scalarprods[9], scalarprods[9]);
   __SSE_DATATYPE vs_1_6 = _SSE_SET(scalarprods[10], scalarprods[10]);
   __SSE_DATATYPE vs_2_6 = _SSE_SET(scalarprods[11], scalarprods[11]);
   __SSE_DATATYPE vs_3_6 = _SSE_SET(scalarprods[12], scalarprods[12]);
   __SSE_DATATYPE vs_4_6 = _SSE_SET(scalarprods[13], scalarprods[13]);
   __SSE_DATATYPE vs_5_6 = _SSE_SET(scalarprods[14], scalarprods[14]);
#endif
#endif /* HAVE_SPARC64_SSE */

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_XOR(tau1, sign);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _fjsp_neg_v2r8(tau1);
#endif
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau1;
#endif

   x1 = _SSE_MUL(x1, h1);
   x2 = _SSE_MUL(x2, h1);

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_XOR(tau2, sign);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _fjsp_neg_v2r8(tau2);
#endif
   h2 = _SSE_MUL(h1, vs);
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau2;
   h2 = _SSE_MUL(h1, vs_1_2); 
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK2
   y1 = _SSE_ADD(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
   y2 = _SSE_ADD(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   y1 = _SSE_SUB(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
   y2 = _SSE_SUB(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
#endif

#if defined(BLOCK4) || defined(BLOCK6)

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

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
   h2 = _SSE_MUL(tau5, vs_1_5); 
   h3 = _SSE_MUL(tau5, vs_2_5);
   h4 = _SSE_MUL(tau5, vs_3_5);
   h5 = _SSE_MUL(tau5, vs_4_5);

   v1 = _SSE_SUB(_SSE_MUL(v1,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));
   v2 = _SSE_SUB(_SSE_MUL(v2,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w2,h5), _SSE_MUL(z2,h4)), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2))));

   h2 = _SSE_MUL(tau6, vs_1_6);
   h3 = _SSE_MUL(tau6, vs_2_6);
   h4 = _SSE_MUL(tau6, vs_3_6);
   h5 = _SSE_MUL(tau6, vs_4_6);
   h6 = _SSE_MUL(tau6, vs_5_6);

   t1 = _SSE_SUB(_SSE_MUL(t1,tau6), _SSE_ADD( _SSE_MUL(v1,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)))));
   t2 = _SSE_SUB(_SSE_MUL(t2,tau6), _SSE_ADD( _SSE_MUL(v2,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w2,h5), _SSE_MUL(z2,h4)), _SSE_ADD(_SSE_MUL(y2,h3), _SSE_MUL(x2,h2)))));

   /////////////////////////////////////////////////////
   // Rank-1 update of Q [4 x nb+3]
   /////////////////////////////////////////////////////
#endif /* BLOCK6 */

   q1 = _SSE_LOAD(&q[0]);
#ifdef BLOCK2
   q1 = _SSE_ADD(q1, y1);
#endif
#ifdef BLOCK4
   q1 = _SSE_SUB(q1, w1);
#endif
#ifdef BLOCK6
   q1 = _SSE_SUB(q1, t1); 
#endif
   _SSE_STORE(&q[0],q1);
   q2 = _SSE_LOAD(&q[offset]);
#ifdef BLOCK2
   q2 = _SSE_ADD(q2, y2);
#endif
#ifdef BLOCK4
   q2 = _SSE_SUB(q2, w2);
#endif
#ifdef BLOCK6
   q2 = _SSE_SUB(q2, t2);
#endif
   _SSE_STORE(&q[offset],q2);

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
#endif

   q1 = _SSE_LOAD(&q[ldq]);
   q1 = _SSE_ADD(q1, _SSE_ADD(x1, _SSE_MUL(y1, h2)));
   _SSE_STORE(&q[ldq],q1);
   q2 = _SSE_LOAD(&q[ldq+offset]);
   q2 = _SSE_ADD(q2, _SSE_ADD(x2, _SSE_MUL(y2, h2)));
   _SSE_STORE(&q[ldq+offset],q2);
#endif /* BLOCK2 */

#ifdef BLOCK4
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
   h4 = _SSE_SET1(hh[(ldh*3)+2]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

   q1 = _SSE_LOAD(&q[ldq*2]);
   q2 = _SSE_LOAD(&q[(ldq*2)+offset]);
   q1 = _SSE_SUB(q1, y1);
   q2 = _SSE_SUB(q2, y2);

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));

   _SSE_STORE(&q[ldq*2],q1);
   _SSE_STORE(&q[(ldq*2)+offset],q2);

#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+3]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

   q1 = _SSE_LOAD(&q[ldq*3]);
   q2 = _SSE_LOAD(&q[(ldq*3)+offset]);
   q1 = _SSE_SUB(q1, x1);
   q2 = _SSE_SUB(q2, x2);

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+2]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
   _SSE_STORE(&q[ldq*3], q1);
   _SSE_STORE(&q[(ldq*3)+offset], q2);

#endif /* BLOCK4 */

#ifdef BLOCK6
   
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+1], hh[(ldh*5)+1]);
#endif

   q1 = _SSE_LOAD(&q[ldq]);
   q2 = _SSE_LOAD(&q[(ldq+offset)]);
   q1 = _SSE_SUB(q1, v1);
   q2 = _SSE_SUB(q2, v2);

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));

   _SSE_STORE(&q[ldq],q1);
   _SSE_STORE(&q[(ldq+offset)],q2);
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+1], hh[(ldh*4)+1]);
#endif
   q1 = _SSE_LOAD(&q[ldq*2]);
   q2 = _SSE_LOAD(&q[(ldq*2)+offset]);
   q1 = _SSE_SUB(q1, w1); 
   q2 = _SSE_SUB(q2, w2);
   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5)); 
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));  
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+2], hh[(ldh*5)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));

   _SSE_STORE(&q[ldq*2],q1);
   _SSE_STORE(&q[(ldq*2)+offset],q2);

#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+1], hh[(ldh*3)+1]);
#endif

   q1 = _SSE_LOAD(&q[ldq*3]);
   q2 = _SSE_LOAD(&q[(ldq*3)+offset]);
   q1 = _SSE_SUB(q1, z1);
   q2 = _SSE_SUB(q2, z2);

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+2], hh[(ldh*4)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+3], hh[(ldh*5)+3]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));

   _SSE_STORE(&q[ldq*3],q1);
   _SSE_STORE(&q[(ldq*3)+offset],q2);
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif
   q1 = _SSE_LOAD(&q[ldq*4]);
   q2 = _SSE_LOAD(&q[(ldq*4)+offset]);
   q1 = _SSE_SUB(q1, y1);
   q2 = _SSE_SUB(q2, y2);

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+3], hh[(ldh*4)+3]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+4]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+4], hh[(ldh*5)+4]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));

   _SSE_STORE(&q[ldq*4],q1);
   _SSE_STORE(&q[(ldq*4)+offset],q2);
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[(ldh)+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[(ldh)+1], hh[(ldh)+1]);
#endif
   q1 = _SSE_LOAD(&q[ldq*5]);
   q2 = _SSE_LOAD(&q[(ldq*5)+offset]);
   q1 = _SSE_SUB(q1, x1);
   q2 = _SSE_SUB(q2, x2);

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+3]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+4]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+4], hh[(ldh*4)+4]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
#ifdef HAVE_SSE_INTRINSICS
   h6 = _SSE_SET1(hh[(ldh*5)+5]);
#endif
#ifdef HAVE_SPARC64_SSE
   h6 = _SSE_SET(hh[(ldh*5)+5], hh[(ldh*5)+5]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
   q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));

   _SSE_STORE(&q[ldq*5],q1);
   _SSE_STORE(&q[(ldq*5)+offset],q2);

#endif /* BLOCK6 */

   for (i = BLOCK; i < nb; i++)
   {
#ifdef HAVE_SSE_INTRINSICS
     h1 = _SSE_SET1(hh[i-(BLOCK-1)]);
     h2 = _SSE_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#ifdef HAVE_SPARC64_SSE
     h1 = _SSE_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
     h2 = _SSE_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif

     q1 = _SSE_LOAD(&q[i*ldq]);
     q2 = _SSE_LOAD(&q[(i*ldq)+offset]);

#ifdef BLOCK2
     q1 = _SSE_ADD(q1, _SSE_ADD(_SSE_MUL(x1,h1), _SSE_MUL(y1, h2)));
     q2 = _SSE_ADD(q2, _SSE_ADD(_SSE_MUL(x2,h1), _SSE_MUL(y2, h2)));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
     q1 = _SSE_SUB(q1, _SSE_MUL(x1,h1));
     q2 = _SSE_SUB(q2, _SSE_MUL(x2,h1));

     q1 = _SSE_SUB(q1, _SSE_MUL(y1,h2));
     q2 = _SSE_SUB(q2, _SSE_MUL(y2,h2));

#ifdef HAVE_SSE_INTRINSICS
     h3 = _SSE_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
     h3 = _SSE_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(z1,h3));
     q2 = _SSE_SUB(q2, _SSE_MUL(z2,h3));

#ifdef HAVE_SSE_INTRINSICS
     h4 = _SSE_SET1(hh[(ldh*3)+i-(BLOCK-4)]); 
#endif

#ifdef HAVE_SPRC64_SSE
     h4 = _SSE_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(w1,h4));
     q2 = _SSE_SUB(q2, _SSE_MUL(w2,h4));

#endif /* BLOCK4 || BLOCK6*/

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
     h5 = _SSE_SET1(hh[(ldh*4)+i-1]);
#endif
#ifdef HAVE_SPARC64_SSE
     h5 = _SSE_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
     q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
#ifdef HAVE_SSE_INTRINSICS
     h6 = _SSE_SET1(hh[(ldh*5)+i]);
#endif
#ifdef HAVE_SPARC64_SSE
     h6 = _SSE_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(t1, h6));
     q2 = _SSE_SUB(q2, _SSE_MUL(t2, h6));
#endif /* BLOCK6 */

     _SSE_STORE(&q[i*ldq],q1);
     _SSE_STORE(&q[(i*ldq)+offset],q2);

   }
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-1)]);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif

   q1 = _SSE_LOAD(&q[nb*ldq]);
   q2 = _SSE_LOAD(&q[(nb*ldq)+offset]);

#ifdef BLOCK2
   q1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+nb-2], hh[(ldh*3)+nb-2]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));
#ifdef HAVE_SSE_INTRINSICS
   h5 = _SSE_SET1(hh[(ldh*4)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h5 = _SSE_SET(hh[(ldh*4)+nb-1], hh[(ldh*4)+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(v1, h5));
   q2 = _SSE_SUB(q2, _SSE_MUL(v2, h5));
#endif /* BLOCK6 */

   _SSE_STORE(&q[nb*ldq],q1);
   _SSE_STORE(&q[(nb*ldq)+offset],q2);

#if defined(BLOCK4) || defined(BLOCK6)
   
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-2)], hh[nb-(BLOCK-2)]);
#endif

   q1 = _SSE_LOAD(&q[(nb+1)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+1)*ldq)+offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-(BLOCK-3)], hh[ldh+nb-(BLOCK-3)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+nb-2], hh[(ldh*2)+nb-2]);
#endif
   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+nb-1], hh[(ldh*3)+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));
   q2 = _SSE_SUB(q2, _SSE_MUL(w2, h4));

#endif /* BLOCK6 */
   _SSE_STORE(&q[(nb+1)*ldq],q1);
   _SSE_STORE(&q[((nb+1)*ldq)+offset],q2);

#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-3)], hh[nb-(BLOCK-3)]);
#endif

   q1 = _SSE_LOAD(&q[(nb+2)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+2)*ldq)+offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-2], hh[ldh+nb-2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));
#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+nb-1], hh[(ldh*2)+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   q2 = _SSE_SUB(q2, _SSE_MUL(z2, h3));
#endif /* BLOCK6 */

   _SSE_STORE(&q[(nb+2)*ldq],q1);
   _SSE_STORE(&q[((nb+2)*ldq)+offset],q2);

#endif /* BLOCK4 || BLOCK6*/

#ifdef BLOCK6
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-2]);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

   q1 = _SSE_LOAD(&q[(nb+3)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+3)*ldq)+offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-1], hh[ldh+nb-1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));
   q2 = _SSE_SUB(q2, _SSE_MUL(y2, h2));

   _SSE_STORE(&q[(nb+3)*ldq],q1);
   _SSE_STORE(&q[((nb+3)*ldq)+offset],q2);
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif

   q1 = _SSE_LOAD(&q[(nb+4)*ldq]);
   q2 = _SSE_LOAD(&q[((nb+4)*ldq)+offset]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));
   q2 = _SSE_SUB(q2, _SSE_MUL(x2, h1));

   _SSE_STORE(&q[(nb+4)*ldq],q1);
   _SSE_STORE(&q[((nb+4)*ldq)+offset],q2);

#endif /* BLOCK6 */

}


/*
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 2 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 4 rows of Q simultaneously, a
#endif
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


#ifdef DOUBLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 2
#endif
#ifdef SINGLE_PRECISION_REAL
#undef ROW_LENGTH
#define ROW_LENGTH 4
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
    // Matrix Vector Multiplication, Q [2 x nb+1] * hh
    // hh contains two householder vectors, with offset 1
    /////////////////////////////////////////////////////
#endif
#if defined(BLOCK4) || defined(BLOCK6)
    /////////////////////////////////////////////////////
    // Matrix Vector Multiplication, Q [2 x nb+3] * hh
    // hh contains four householder vectors
    /////////////////////////////////////////////////////
#endif

    int i;
#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
    // Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
    __SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
    __SSE_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif
    __SSE_DATATYPE x1 = _SSE_LOAD(&q[ldq]);

#ifdef HAVE_SSE_INTRINSICS
    __SSE_DATATYPE h1 = _SSE_SET1(hh[ldh+1]);
#endif
#ifdef HAVE_SPARC64_SSE
    __SSE_DATATYPE h1 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
#endif
    __SSE_DATATYPE h2;

    __SSE_DATATYPE q1 = _SSE_LOAD(q);
    __SSE_DATATYPE y1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
#endif /* BLOCK2 */

#ifdef BLOCK4
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

    register __SSE_DATATYPE w1 = _SSE_ADD(a4_1, _SSE_MUL(a3_1, h_4_3));
    w1 = _SSE_ADD(w1, _SSE_MUL(a2_1, h_4_2));                          
    w1 = _SSE_ADD(w1, _SSE_MUL(a1_1, h_4_1));                          
    register __SSE_DATATYPE z1 = _SSE_ADD(a3_1, _SSE_MUL(a2_1, h_3_2));
    z1 = _SSE_ADD(z1, _SSE_MUL(a1_1, h_3_1));                          
    register __SSE_DATATYPE y1 = _SSE_ADD(a2_1, _SSE_MUL(a1_1, h_2_1));
    register __SSE_DATATYPE x1 = a1_1;

    __SSE_DATATYPE q1;

    __SSE_DATATYPE h1;
    __SSE_DATATYPE h2;
    __SSE_DATATYPE h3;
    __SSE_DATATYPE h4;
#endif /* BLOCK4 */

#ifdef BLOCK6
    
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

#endif /* BLOCK6 */

    for(i = BLOCK; i < nb; i++)
      {
#ifdef HAVE_SSE_INTRINSICS
        h1 = _SSE_SET1(hh[i-(BLOCK-1)]);
        h2 = _SSE_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#ifdef HAVE_SPARC64_SSE
        h1 = _SSE_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
        h2 = _SSE_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif

        q1 = _SSE_LOAD(&q[i*ldq]);
        x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
        y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));

#if defined(BLOCK4) || defined(BLOCK6)
#ifdef HAVE_SSE_INTRINSICS
        h3 = _SSE_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
        h3 = _SSE_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

        z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));

#ifdef HAVE_SSE_INTRINSICS
        h4 = _SSE_SET1(hh[(ldh*3)+i-(BLOCK-4)]);
#endif

#ifdef HAVE_SPARC64_SSE
        h4 = _SSE_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

        w1 = _SSE_ADD(w1, _SSE_MUL(q1,h4));
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

#ifdef HAVE_SSE_INTRINSICS
        h5 = _SSE_SET1(hh[(ldh*4)+i-1]);
#endif
#ifdef HAVE_SPARC64_SSE
        h5 = _SSE_SET(hh[(ldh*4)+i-1], hh[(ldh*4)+i-1]);
#endif
        v1 = _SSE_ADD(v1, _SSE_MUL(q1,h5));

#ifdef HAVE_SSE_INTRINSICS
        h6 = _SSE_SET1(hh[(ldh*5)+i]);
#endif

#ifdef HAVE_SPARC64_SSE
        h6 = _SSE_SET(hh[(ldh*5)+i], hh[(ldh*5)+i]);
#endif

        t1 = _SSE_ADD(t1, _SSE_MUL(q1,h6));
	
#endif /* BLOCK6 */
      }
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-(BLOCK-1)]);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif

    q1 = _SSE_LOAD(&q[nb*ldq]);
    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));

#if defined(BLOCK4) || defined(BLOCK6)
    
#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));

#ifdef HAVE_SSE_INTRINSICS
    h3 = _SSE_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
    h3 = _SSE_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

    z1 = _SSE_ADD(z1, _SSE_MUL(q1,h3));

#ifdef BLOCK4

#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-2]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-2], hh[nb-2]);
#endif

    q1 = _SSE_LOAD(&q[(nb+1)*ldq]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));

#ifdef HAVE_SSE_INTRINSICS
    h2 = _SSE_SET1(hh[(ldh*1)+nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h2 = _SSE_SET(hh[(ldh*1)+nb-1], hh[(ldh*1)+nb-1]);
#endif

    y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));

#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_SET1(hh[nb-1]);
#endif

#ifdef HAVE_SPARC64_SSE
    h1 = _SSE_SET(hh[nb-1], hh[nb-1]);
#endif


    q1 = _SSE_LOAD(&q[(nb+2)*ldq]);

    x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
#endif /* BLOCK4 */
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6

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
#endif /* BLOCK6 */

#ifdef BLOCK2
    /////////////////////////////////////////////////////
    // Rank-2 update of Q [4 x nb+1]
    /////////////////////////////////////////////////////
#endif
#ifdef BLOCK4
    /////////////////////////////////////////////////////
    // Rank-1 update of Q [4 x nb+3]
    /////////////////////////////////////////////////////
#endif
#ifdef BLOCK6
    /////////////////////////////////////////////////////
    // Apply tau, correct wrong calculation using pre-calculated scalar products
    /////////////////////////////////////////////////////
#endif


#ifdef HAVE_SSE_INTRINSICS
    __SSE_DATATYPE tau1 = _SSE_SET1(hh[0]);
    __SSE_DATATYPE tau2 = _SSE_SET1(hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SSE_DATATYPE tau3 = _SSE_SET1(hh[ldh*2]);
   __SSE_DATATYPE tau4 = _SSE_SET1(hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE tau5 = _SSE_SET1(hh[ldh*4]);
   __SSE_DATATYPE tau6 = _SSE_SET1(hh[ldh*5]);       
#endif

#ifdef BLOCK2    
    __SSE_DATATYPE vs = _SSE_SET1(s);
#endif
#ifdef BLOCK4
   __SSE_DATATYPE vs_1_2 = _SSE_SET1(s_1_2);
   __SSE_DATATYPE vs_1_3 = _SSE_SET1(s_1_3);  
   __SSE_DATATYPE vs_2_3 = _SSE_SET1(s_2_3);
   __SSE_DATATYPE vs_1_4 = _SSE_SET1(s_1_4);  
   __SSE_DATATYPE vs_2_4 = _SSE_SET1(s_2_4);  
   __SSE_DATATYPE vs_3_4 = _SSE_SET1(s_3_4);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE vs_1_2 = _SSE_SET1(scalarprods[0]);
   __SSE_DATATYPE vs_1_3 = _SSE_SET1(scalarprods[1]);
   __SSE_DATATYPE vs_2_3 = _SSE_SET1(scalarprods[2]);
   __SSE_DATATYPE vs_1_4 = _SSE_SET1(scalarprods[3]);
   __SSE_DATATYPE vs_2_4 = _SSE_SET1(scalarprods[4]);
   __SSE_DATATYPE vs_3_4 = _SSE_SET1(scalarprods[5]);
   __SSE_DATATYPE vs_1_5 = _SSE_SET1(scalarprods[6]);
   __SSE_DATATYPE vs_2_5 = _SSE_SET1(scalarprods[7]);
   __SSE_DATATYPE vs_3_5 = _SSE_SET1(scalarprods[8]);
   __SSE_DATATYPE vs_4_5 = _SSE_SET1(scalarprods[9]);
   __SSE_DATATYPE vs_1_6 = _SSE_SET1(scalarprods[10]);
   __SSE_DATATYPE vs_2_6 = _SSE_SET1(scalarprods[11]);
   __SSE_DATATYPE vs_3_6 = _SSE_SET1(scalarprods[12]);
   __SSE_DATATYPE vs_4_6 = _SSE_SET1(scalarprods[13]);
   __SSE_DATATYPE vs_5_6 = _SSE_SET1(scalarprods[14]);
#endif
#endif /* HAVE_SSE_INTRINSICS */

#ifdef HAVE_SPARC64_SSE
    __SSE_DATATYPE tau1 = _SSE_SET(hh[0], hh[0]);
    __SSE_DATATYPE tau2 = _SSE_SET(hh[ldh], hh[ldh]);
#if defined(BLOCK4) || defined(BLOCK6)
   __SSE_DATATYPE tau3 = _SSE_SET(hh[ldh*2], hh[ldh*2]);
   __SSE_DATATYPE tau4 = _SSE_SET(hh[ldh*3], hh[ldh*3]);
#endif
#ifdef BLOCK6
   __SSE_DATATYPE tau5 = _SSE_SET(hh[ldh*4], hh[ldh*4]);
   __SSE_DATATYPE tau6 = _SSE_SET(hh[ldh*5], hh[ldh*5]);
#endif

#ifdef BLOCK2
    __SSE_DATATYPE vs = _SSE_SET(s, s);
#endif
#ifdef BLOCK4
   __SSE_DATATYPE vs_1_2 = _SSE_SET(s_1_2, s_1_2);
   __SSE_DATATYPE vs_1_3 = _SSE_SET(s_1_3, s_1_3);
   __SSE_DATATYPE vs_2_3 = _SSE_SET(s_2_3, s_2_3);
   __SSE_DATATYPE vs_1_4 = _SSE_SET(s_1_4, s_1_4);
   __SSE_DATATYPE vs_2_4 = _SSE_SET(s_2_4, s_2_4);
   __SSE_DATATYPE vs_3_4 = _SSE_SET(s_3_4, s_3_4);

#endif
#ifdef BLOCK6
   __SSE_DATATYPE vs_1_2 = _SSE_SET(scalarprods[0], scalarprods[0]);
   __SSE_DATATYPE vs_1_3 = _SSE_SET(scalarprods[1], scalarprods[1]);
   __SSE_DATATYPE vs_2_3 = _SSE_SET(scalarprods[2], scalarprods[2]);
   __SSE_DATATYPE vs_1_4 = _SSE_SET(scalarprods[3], scalarprods[3]);
   __SSE_DATATYPE vs_2_4 = _SSE_SET(scalarprods[4], scalarprods[4]);
   __SSE_DATATYPE vs_3_4 = _SSE_SET(scalarprods[5], scalarprods[5]);
   __SSE_DATATYPE vs_1_5 = _SSE_SET(scalarprods[6], scalarprods[6]);
   __SSE_DATATYPE vs_2_5 = _SSE_SET(scalarprods[7], scalarprods[7]);
   __SSE_DATATYPE vs_3_5 = _SSE_SET(scalarprods[8], scalarprods[8]);
   __SSE_DATATYPE vs_4_5 = _SSE_SET(scalarprods[9], scalarprods[9]);
   __SSE_DATATYPE vs_1_6 = _SSE_SET(scalarprods[10], scalarprods[10]);
   __SSE_DATATYPE vs_2_6 = _SSE_SET(scalarprods[11], scalarprods[11]);
   __SSE_DATATYPE vs_3_6 = _SSE_SET(scalarprods[12], scalarprods[12]);
   __SSE_DATATYPE vs_4_6 = _SSE_SET(scalarprods[13], scalarprods[13]);
   __SSE_DATATYPE vs_5_6 = _SSE_SET(scalarprods[14], scalarprods[14]);
#endif
#endif /* HAVE_SPARC64_SSE */

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
    h1 = _SSE_XOR(tau1, sign);
#endif
#ifdef HAVE_SPARC64_SSE
    h1 = _fjsp_neg_v2r8(tau1);
#endif
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau1;
#endif

   x1 = _SSE_MUL(x1, h1);

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_XOR(tau2, sign);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _fjsp_neg_v2r8(tau2);
#endif
   h2 = _SSE_MUL(h1, vs);
#endif /* BLOCK2 */

#if defined(BLOCK4) || defined(BLOCK6)
   h1 = tau2;
   h2 = _SSE_MUL(h1, vs_1_2); 
#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK2
   y1 = _SSE_ADD(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   y1 = _SSE_SUB(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
#endif

#if defined(BLOCK4) || defined(BLOCK6)

   h1 = tau3;
   h2 = _SSE_MUL(h1, vs_1_3);
   h3 = _SSE_MUL(h1, vs_2_3);

   z1 = _SSE_SUB(_SSE_MUL(z1,h1), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)));

   h1 = tau4;
   h2 = _SSE_MUL(h1, vs_1_4);
   h3 = _SSE_MUL(h1, vs_2_4);
   h4 = _SSE_MUL(h1, vs_3_4);

   w1 = _SSE_SUB(_SSE_MUL(w1,h1), _SSE_ADD(_SSE_MUL(z1,h4), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)))); 

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
   h2 = _SSE_MUL(tau5, vs_1_5); 
   h3 = _SSE_MUL(tau5, vs_2_5);

   h4 = _SSE_MUL(tau5, vs_3_5);
   h5 = _SSE_MUL(tau5, vs_4_5);

   v1 = _SSE_SUB(_SSE_MUL(v1,tau5), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2))));

   h2 = _SSE_MUL(tau6, vs_1_6);
   h3 = _SSE_MUL(tau6, vs_2_6);
   h4 = _SSE_MUL(tau6, vs_3_6);
   h5 = _SSE_MUL(tau6, vs_4_6);
   h6 = _SSE_MUL(tau6, vs_5_6);

   t1 = _SSE_SUB(_SSE_MUL(t1,tau6), _SSE_ADD( _SSE_MUL(v1,h6), _SSE_ADD(_SSE_ADD(_SSE_MUL(w1,h5), _SSE_MUL(z1,h4)), _SSE_ADD(_SSE_MUL(y1,h3), _SSE_MUL(x1,h2)))));

   /////////////////////////////////////////////////////
   // Rank-1 update of Q [4 x nb+3]
   /////////////////////////////////////////////////////
#endif /* BLOCK6 */

   q1 = _SSE_LOAD(&q[0]);
#ifdef BLOCK2
   q1 = _SSE_ADD(q1, y1);
#endif
#ifdef BLOCK4
   q1 = _SSE_SUB(q1, w1);
#endif
#ifdef BLOCK6
   q1 = _SSE_SUB(q1, t1); 
#endif
   _SSE_STORE(&q[0],q1);

#ifdef BLOCK2
#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+1]);
#endif
#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
#endif

   q1 = _SSE_LOAD(&q[ldq]);
   q1 = _SSE_ADD(q1, _SSE_ADD(x1, _SSE_MUL(y1, h2)));
   _SSE_STORE(&q[ldq],q1);
#endif /* BLOCK2 */

#ifdef BLOCK4
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
   h4 = _SSE_SET1(hh[(ldh*3)+2]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+2], hh[(ldh*3)+2]);
#endif

   q1 = _SSE_LOAD(&q[ldq*2]);
   q1 = _SSE_SUB(q1, y1);

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+1], hh[(ldh*2)+1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));

   _SSE_STORE(&q[ldq*2],q1);

#ifdef HAVE_SSE_INTRINSICS
   h4 = _SSE_SET1(hh[(ldh*3)+3]);
#endif

#ifdef HAVE_SPARC64_SSE
   h4 = _SSE_SET(hh[(ldh*3)+3], hh[(ldh*3)+3]);
#endif

   q1 = _SSE_LOAD(&q[ldq*3]);
   q1 = _SSE_SUB(q1, x1);

   q1 = _SSE_SUB(q1, _SSE_MUL(w1, h4));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+1]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+1], hh[ldh+1]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+2]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+2], hh[(ldh*2)+2]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));
   _SSE_STORE(&q[ldq*3], q1);

#endif /* BLOCK4 */

#ifdef BLOCK6
   
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

#endif /* BLOCK6 */

   for (i = BLOCK; i < nb; i++)
   {
#ifdef HAVE_SSE_INTRINSICS
     h1 = _SSE_SET1(hh[i-(BLOCK-1)]);
     h2 = _SSE_SET1(hh[ldh+i-(BLOCK-2)]);
#endif
#ifdef HAVE_SPARC64_SSE
     h1 = _SSE_SET(hh[i-(BLOCK-1)], hh[i-(BLOCK-1)]);
     h2 = _SSE_SET(hh[ldh+i-(BLOCK-2)], hh[ldh+i-(BLOCK-2)]);
#endif

     q1 = _SSE_LOAD(&q[i*ldq]);

#ifdef BLOCK2
     q1 = _SSE_ADD(q1, _SSE_ADD(_SSE_MUL(x1,h1), _SSE_MUL(y1, h2)));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
     q1 = _SSE_SUB(q1, _SSE_MUL(x1,h1));

     q1 = _SSE_SUB(q1, _SSE_MUL(y1,h2));

#ifdef HAVE_SSE_INTRINSICS
     h3 = _SSE_SET1(hh[(ldh*2)+i-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
     h3 = _SSE_SET(hh[(ldh*2)+i-(BLOCK-3)], hh[(ldh*2)+i-(BLOCK-3)]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(z1,h3));

#ifdef HAVE_SSE_INTRINSICS
     h4 = _SSE_SET1(hh[(ldh*3)+i-(BLOCK-4)]); 
#endif

#ifdef HAVE_SPRC64_SSE
     h4 = _SSE_SET(hh[(ldh*3)+i-(BLOCK-4)], hh[(ldh*3)+i-(BLOCK-4)]);
#endif

     q1 = _SSE_SUB(q1, _SSE_MUL(w1,h4));

#endif /* BLOCK4 || BLOCK6*/

#ifdef BLOCK6
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
#endif /* BLOCK6 */

     _SSE_STORE(&q[i*ldq],q1);

   }
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-1)]);
#endif
#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-1)], hh[nb-(BLOCK-1)]);
#endif

   q1 = _SSE_LOAD(&q[nb*ldq]);

#ifdef BLOCK2
   q1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
#endif

#if defined(BLOCK4) || defined(BLOCK6)
   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-(BLOCK-2)], hh[ldh+nb-(BLOCK-2)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));

#ifdef HAVE_SSE_INTRINSICS
   h3 = _SSE_SET1(hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h3 = _SSE_SET(hh[(ldh*2)+nb-(BLOCK-3)], hh[(ldh*2)+nb-(BLOCK-3)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(z1, h3));

#endif /* BLOCK4 || BLOCK6 */

#ifdef BLOCK6
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
#endif /* BLOCK6 */

   _SSE_STORE(&q[nb*ldq],q1);

#if defined(BLOCK4) || defined(BLOCK6)
   
#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-2)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-2)], hh[nb-(BLOCK-2)]);
#endif

   q1 = _SSE_LOAD(&q[(nb+1)*ldq]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));

#ifdef HAVE_SSE_INTRINSICS
   h2 = _SSE_SET1(hh[ldh+nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h2 = _SSE_SET(hh[ldh+nb-(BLOCK-3)], hh[ldh+nb-(BLOCK-3)]);
#endif

   q1 = _SSE_SUB(q1, _SSE_MUL(y1, h2));

#ifdef BLOCK6
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

#endif /* BLOCK6 */
   _SSE_STORE(&q[(nb+1)*ldq],q1);

#ifdef HAVE_SSE_INTRINSICS
   h1 = _SSE_SET1(hh[nb-(BLOCK-3)]);
#endif

#ifdef HAVE_SPARC64_SSE
   h1 = _SSE_SET(hh[nb-(BLOCK-3)], hh[nb-(BLOCK-3)]);
#endif

   q1 = _SSE_LOAD(&q[(nb+2)*ldq]);

   q1 = _SSE_SUB(q1, _SSE_MUL(x1, h1));

#ifdef BLOCK6
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
#endif /* BLOCK6 */

   _SSE_STORE(&q[(nb+2)*ldq],q1);

#endif /* BLOCK4 || BLOCK6*/

#ifdef BLOCK6
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

#endif /* BLOCK6 */

}

