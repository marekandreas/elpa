//    Copyright 2018, A. Marek
//
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
// Author: Andreas Marek, MPCDF

#include "config.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#include "elpa/elpa_simd_constants.h"

static inline void get_cpu_manufacturer(int *set)
{
  u_int32_t registers[4];
  registers[0] = 0;
  asm volatile("cpuid": "=a" (registers[0]),"=b" (registers[1]),"=c" (registers[3]),"=d" (registers[2]): "0" (registers[0]), "2" (registers[2]): "memory");

  char str[13]="GenuineIntel\0";
  char manufacturer[13];

  memcpy(manufacturer, registers[1], 12); 
  manufacturer[12] = '\0';

  if (strcmp(manufacturer, str) == 0) {
    set[CPU_MANUFACTURER - 1] = 1;
  } else { 
    set[CPU_MANUFACTURER - 1] = 0;
  }
}

#ifdef HAVE_HETEROGENOUS_CLUSTER_SUPPORT
#include <cpuid.h>
void cpuid(int info[4], int InfoType){
    __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}
#endif

/*
!f>#ifdef HAVE_HETEROGENOUS_CLUSTER_SUPPORT
!f> interface
!f>   subroutine get_cpuid_set(simdSet, n) &
!f>              bind(C, name="get_cpuid_set")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int), value :: n
!f>     integer(kind=c_int)        :: simdSet(n)
!f>   end subroutine
!f> end interface
!f>#endif
*/
void get_cpuid_set(int *set, int nlength){

  get_cpu_manufacturer(set);


  // Code below taken from http://stackoverflow.com/questions/6121792/how-to-check-if-a-cpu-supports-the-sse3-instruction-set/7495023#7495023

  //  Misc.
  bool HW_MMX;
  bool HW_x64;
  bool HW_ABM;      // Advanced Bit Manipulation
  bool HW_RDRAND;
  bool HW_BMI1;
  bool HW_BMI2;
  bool HW_ADX;
  bool HW_PREFETCHWT1;
  
  //  SIMD: 128-bit
  bool HW_SSE;
  bool HW_SSE2;
  bool HW_SSE3;
  bool HW_SSSE3;
  bool HW_SSE41;
  bool HW_SSE42;
  bool HW_SSE4a;
  bool HW_AES;
  bool HW_SHA;
  
  //  SIMD: 256-bit
  bool HW_AVX;
  bool HW_XOP;
  bool HW_FMA3;
  bool HW_FMA4;
  bool HW_AVX2;
  //  SIMD: 512-bit
  bool HW_AVX512F;    //  AVX512 Foundation
  bool HW_AVX512CD;   //  AVX512 Conflict Detection
  bool HW_AVX512PF;   //  AVX512 Prefetch
  bool HW_AVX512ER;   //  AVX512 Exponential + Reciprocal
  bool HW_AVX512VL;   //  AVX512 Vector Length Extensions
  bool HW_AVX512BW;   //  AVX512 Byte + Word
  bool HW_AVX512DQ;   //  AVX512 Doubleword + Quadword
  bool HW_AVX512IFMA; //  AVX512 Integer 52-bit Fused Multiply-Add
  bool HW_AVX512VBMI; //  AVX512 Vector Byte Manipulation Instructions
  
  int info[4];
#ifdef HAVE_HETEROGENOUS_CLUSTER_SUPPORT

  cpuid(info, 0);
  int nIds = info[0];
  
  cpuid(info, 0x80000000);
  unsigned nExIds = info[0];
#endif  
  //  Detect Features
  if (nIds >= 0x00000001){
#ifdef HAVE_HETEROGENOUS_CLUSTER_SUPPORT
    cpuid(info,0x00000001);
#endif
    HW_MMX    = (info[3] & ((int)1 << 23)) != 0;
    HW_SSE    = (info[3] & ((int)1 << 25)) != 0;
    HW_SSE2   = (info[3] & ((int)1 << 26)) != 0;
    HW_SSE3   = (info[2] & ((int)1 <<  0)) != 0;

    HW_SSSE3  = (info[2] & ((int)1 <<  9)) != 0;
    HW_SSE41  = (info[2] & ((int)1 << 19)) != 0;
    HW_SSE42  = (info[2] & ((int)1 << 20)) != 0;
    HW_AES    = (info[2] & ((int)1 << 25)) != 0;

    HW_AVX    = (info[2] & ((int)1 << 28)) != 0;
    HW_FMA3   = (info[2] & ((int)1 << 12)) != 0;
    HW_RDRAND = (info[2] & ((int)1 << 30)) != 0;
  }
  if (nIds >= 0x00000007){
#ifdef HAVE_HETEROGENOUS_CLUSTER_SUPPORT
    cpuid(info,0x00000007);
#endif
    HW_AVX2   = (info[1] & ((int)1 <<  5)) != 0;

    HW_BMI1        = (info[1] & ((int)1 <<  3)) != 0;
    HW_BMI2        = (info[1] & ((int)1 <<  8)) != 0;
    HW_ADX         = (info[1] & ((int)1 << 19)) != 0;
    HW_SHA         = (info[1] & ((int)1 << 29)) != 0;
    HW_PREFETCHWT1 = (info[2] & ((int)1 <<  0)) != 0;

    HW_AVX512F     = (info[1] & ((int)1 << 16)) != 0;
    HW_AVX512CD    = (info[1] & ((int)1 << 28)) != 0;
    HW_AVX512PF    = (info[1] & ((int)1 << 26)) != 0;
    HW_AVX512ER    = (info[1] & ((int)1 << 27)) != 0;
    HW_AVX512VL    = (info[1] & ((int)1 << 31)) != 0;
    HW_AVX512BW    = (info[1] & ((int)1 << 30)) != 0;
    HW_AVX512DQ    = (info[1] & ((int)1 << 17)) != 0;
    HW_AVX512IFMA  = (info[1] & ((int)1 << 21)) != 0;
    HW_AVX512VBMI  = (info[2] & ((int)1 <<  1)) != 0;
  }

  if (nExIds >= 0x80000001){
#ifdef HAVE_HETEROGENOUS_CLUSTER_SUPPORT
    cpuid(info,0x80000001);
#endif
    HW_x64   = (info[3] & ((int)1 << 29)) != 0;
    HW_ABM   = (info[2] & ((int)1 <<  5)) != 0;
    HW_SSE4a = (info[2] & ((int)1 <<  6)) != 0;
    HW_FMA4  = (info[2] & ((int)1 << 16)) != 0;
    HW_XOP   = (info[2] & ((int)1 << 11)) != 0;
  }

  //allways allow GENERIC
  set[GENERIC_INSTR -1] =1;

  // the rest depends on the CPU
  if (HW_SSE42) {
    set[SSE_INSTR - 1] = 1;
  }
  if (HW_AVX) {
    set[AVX_INSTR - 1] = 1;
  }
  if (HW_AVX2) {
    set[AVX2_INSTR - 1] = 1;
  }
  if (HW_AVX512F) {
    set[AVX512_INSTR -1] = 1;
  }

}


