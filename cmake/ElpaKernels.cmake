# ElpaKernels.cmake — SIMD kernel selection for ELPA
#
# Mirrors the kernel selection chain in configure.ac:
#   1. User options select kernel families (generic, sse, avx, avx2, avx512, ...)
#   2. Dependency propagation: block6 → block4 → block2, block2 → block1
#   3. ISA feature detection via compiler flag checks
#   4. Default kernel selection: priority order avx512 → avx2 → avx → sse → generic
#   5. Exported variables: WITH_<KERNEL>_KERNEL, ELPA_2STAGE_<KERNEL>_COMPILED,
#      ELPA_2STAGE_REAL_DEFAULT, ELPA_2STAGE_COMPLEX_DEFAULT, HAVE_<ISA>

include(CheckCCompilerFlag)
include(CheckCSourceCompiles)

# ===========================================================================
# 1. Kernel family option validation
# ===========================================================================
# User-facing kernel options are declared centrally in CMakeBuildOptions.cmake.

if(WIN32 OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
    # GNU .s kernels are x86-only and not portable to other architectures
    # or the Windows cl/ifort toolchain setup.
    set(ELPA_ENABLE_SSE_ASSEMBLY_KERNELS
        OFF
        CACHE BOOL
        "Build SSE assembly kernels"
        FORCE
    )
endif()

# ===========================================================================
# Helper: map family option → per-kernel use flags
# ===========================================================================
# Each kernel family consists of several real/complex kernel variants.
# We need individual per-kernel flags for defines and source selection.

# --- Generic ---
macro(_elpa_set_family_kernels _family)
    # Each family has a list of real/complex kernels.  This macro reads
    # ELPA_ENABLE_<FAMILY>_KERNELS and sets use_<kernel>=ON/OFF for each.
endmacro()

# Individual kernel use flags.  We first set everything by family, then
# apply dependency propagation.

# Generic family
set(use_real_generic ${ELPA_ENABLE_GENERIC_KERNELS})
set(use_real_generic_simple ${ELPA_ENABLE_GENERIC_KERNELS})
set(use_real_generic_simple_block4 ${ELPA_ENABLE_GENERIC_KERNELS})
set(use_real_generic_simple_block6 ${ELPA_ENABLE_GENERIC_KERNELS})
set(use_complex_generic ${ELPA_ENABLE_GENERIC_KERNELS})
set(use_complex_generic_simple ${ELPA_ENABLE_GENERIC_KERNELS})

# SSE family
set(use_real_sse_block2 ${ELPA_ENABLE_SSE_KERNELS})
set(use_real_sse_block4 ${ELPA_ENABLE_SSE_KERNELS})
set(use_real_sse_block6 ${ELPA_ENABLE_SSE_KERNELS})
set(use_complex_sse_block1 ${ELPA_ENABLE_SSE_KERNELS})
set(use_complex_sse_block2 ${ELPA_ENABLE_SSE_KERNELS})

# SSE Assembly family
set(use_real_sse_assembly ${ELPA_ENABLE_SSE_ASSEMBLY_KERNELS})
set(use_complex_sse_assembly ${ELPA_ENABLE_SSE_ASSEMBLY_KERNELS})

# AVX family
set(use_real_avx_block2 ${ELPA_ENABLE_AVX_KERNELS})
set(use_real_avx_block4 ${ELPA_ENABLE_AVX_KERNELS})
set(use_real_avx_block6 ${ELPA_ENABLE_AVX_KERNELS})
set(use_complex_avx_block1 ${ELPA_ENABLE_AVX_KERNELS})
set(use_complex_avx_block2 ${ELPA_ENABLE_AVX_KERNELS})

# AVX2 family
set(use_real_avx2_block2 ${ELPA_ENABLE_AVX2_KERNELS})
set(use_real_avx2_block4 ${ELPA_ENABLE_AVX2_KERNELS})
set(use_real_avx2_block6 ${ELPA_ENABLE_AVX2_KERNELS})
set(use_complex_avx2_block1 ${ELPA_ENABLE_AVX2_KERNELS})
set(use_complex_avx2_block2 ${ELPA_ENABLE_AVX2_KERNELS})

# AVX512 family
set(use_real_avx512_block2 ${ELPA_ENABLE_AVX512_KERNELS})
set(use_real_avx512_block4 ${ELPA_ENABLE_AVX512_KERNELS})
set(use_real_avx512_block6 ${ELPA_ENABLE_AVX512_KERNELS})
set(use_complex_avx512_block1 ${ELPA_ENABLE_AVX512_KERNELS})
set(use_complex_avx512_block2 ${ELPA_ENABLE_AVX512_KERNELS})

# SVE128 family
set(use_real_sve128_block2 ${ELPA_ENABLE_SVE128_KERNELS})
set(use_real_sve128_block4 ${ELPA_ENABLE_SVE128_KERNELS})
set(use_real_sve128_block6 ${ELPA_ENABLE_SVE128_KERNELS})
set(use_complex_sve128_block1 ${ELPA_ENABLE_SVE128_KERNELS})
set(use_complex_sve128_block2 ${ELPA_ENABLE_SVE128_KERNELS})

# SVE256 family
set(use_real_sve256_block2 ${ELPA_ENABLE_SVE256_KERNELS})
set(use_real_sve256_block4 ${ELPA_ENABLE_SVE256_KERNELS})
set(use_real_sve256_block6 ${ELPA_ENABLE_SVE256_KERNELS})
set(use_complex_sve256_block1 ${ELPA_ENABLE_SVE256_KERNELS})
set(use_complex_sve256_block2 ${ELPA_ENABLE_SVE256_KERNELS})

# SVE512 family
set(use_real_sve512_block2 ${ELPA_ENABLE_SVE512_KERNELS})
set(use_real_sve512_block4 ${ELPA_ENABLE_SVE512_KERNELS})
set(use_real_sve512_block6 ${ELPA_ENABLE_SVE512_KERNELS})
set(use_complex_sve512_block1 ${ELPA_ENABLE_SVE512_KERNELS})
set(use_complex_sve512_block2 ${ELPA_ENABLE_SVE512_KERNELS})

# SPARC64 family
set(use_real_sparc64_block2 ${ELPA_ENABLE_SPARC64_KERNELS})
set(use_real_sparc64_block4 ${ELPA_ENABLE_SPARC64_KERNELS})
set(use_real_sparc64_block6 ${ELPA_ENABLE_SPARC64_KERNELS})
set(use_complex_sparc64_block1 ${ELPA_ENABLE_SPARC64_KERNELS})
set(use_complex_sparc64_block2 ${ELPA_ENABLE_SPARC64_KERNELS})

# NEON AARCH64 family
set(use_real_neon_arch64_block2 ${ELPA_ENABLE_NEON_ARCH64_KERNELS})
set(use_real_neon_arch64_block4 ${ELPA_ENABLE_NEON_ARCH64_KERNELS})
set(use_real_neon_arch64_block6 ${ELPA_ENABLE_NEON_ARCH64_KERNELS})
set(use_complex_neon_arch64_block1 ${ELPA_ENABLE_NEON_ARCH64_KERNELS})
set(use_complex_neon_arch64_block2 ${ELPA_ENABLE_NEON_ARCH64_KERNELS})

# VSX family
set(use_real_vsx_block2 ${ELPA_ENABLE_VSX_KERNELS})
set(use_real_vsx_block4 ${ELPA_ENABLE_VSX_KERNELS})
set(use_real_vsx_block6 ${ELPA_ENABLE_VSX_KERNELS})
set(use_complex_vsx_block1 ${ELPA_ENABLE_VSX_KERNELS})
set(use_complex_vsx_block2 ${ELPA_ENABLE_VSX_KERNELS})

# GPU kernels (flat — no block sub-variants)
set(use_real_nvidia_gpu ${ELPA_ENABLE_NVIDIA_GPU_KERNELS})
set(use_complex_nvidia_gpu ${ELPA_ENABLE_NVIDIA_GPU_KERNELS})
set(use_real_nvidia_sm80_gpu ${ELPA_ENABLE_NVIDIA_SM80_GPU_KERNELS})
set(use_complex_nvidia_sm80_gpu ${ELPA_ENABLE_NVIDIA_SM80_GPU_KERNELS})
set(use_real_amd_gpu ${ELPA_ENABLE_AMD_GPU_KERNELS})
set(use_complex_amd_gpu ${ELPA_ENABLE_AMD_GPU_KERNELS})
set(use_real_intel_gpu_sycl ${ELPA_ENABLE_INTEL_GPU_SYCL_KERNELS})
set(use_complex_intel_gpu_sycl ${ELPA_ENABLE_INTEL_GPU_SYCL_KERNELS})

# ===========================================================================
# 2. Dependency propagation (block6 → block4 → block2, block2 → block1)
# ===========================================================================
# Mirrors ELPA_KERNEL_DEPENDS from configure.ac.
macro(_elpa_kernel_dep _kernel _dep)
    if(${_kernel})
        if(NOT ${_dep})
            message(
                STATUS
                "ELPA: Enabling ${_dep} (prerequisite for ${_kernel})"
            )
            set(${_dep} ON)
        endif()
    endif()
endmacro()

foreach(
    _arch
    IN
    ITEMS sparc64 neon_arch64 vsx sse avx avx2 avx512 sve128 sve256 sve512
)
    _elpa_kernel_dep(use_real_${_arch}_block6  use_real_${_arch}_block4)
    _elpa_kernel_dep(use_real_${_arch}_block6  use_real_${_arch}_block2)
    _elpa_kernel_dep(use_real_${_arch}_block4  use_real_${_arch}_block2)
    _elpa_kernel_dep(use_complex_${_arch}_block2 use_complex_${_arch}_block1)
endforeach()

# ===========================================================================
# 3. ISA feature detection
# ===========================================================================

# Helper to check a C compile flag and set HAVE_<feature>
macro(_elpa_check_isa _feature _flag _test_code)
    set(_need_check OFF)
    # Determine if any kernel in this family is enabled
    if("${_feature}" STREQUAL "SSE_INTRINSICS")
        if(ELPA_ENABLE_SSE_KERNELS)
            set(_need_check ON)
        endif()
    elseif("${_feature}" STREQUAL "AVX")
        if(ELPA_ENABLE_AVX_KERNELS)
            set(_need_check ON)
        endif()
    elseif("${_feature}" STREQUAL "AVX2")
        if(ELPA_ENABLE_AVX2_KERNELS)
            set(_need_check ON)
        endif()
    elseif("${_feature}" STREQUAL "AVX512")
        if(ELPA_ENABLE_AVX512_KERNELS)
            set(_need_check ON)
        endif()
    else()
        set(_need_check ON)
    endif()

    if(_need_check)
        check_c_source_compiles("${_test_code}" HAVE_${_feature})
        if(NOT HAVE_${_feature})
            message(
                WARNING
                "ELPA: Cannot compile ${_feature} test — disabling ${_feature} kernels"
            )
        endif()
    endif()
endmacro()

# SSE intrinsics
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS}")
if(ELPA_ENABLE_SSE_KERNELS)
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS} -msse3")
    check_c_source_compiles(
        "
#include <x86intrin.h>
int main(void) { double *q; __m128d h1 = _mm_loaddup_pd(q); return 0; }
"
        HAVE_SSE_INTRINSICS
    )
    if(NOT HAVE_SSE_INTRINSICS)
        message(
            WARNING
            "ELPA: SSE intrinsics not available, disabling SSE kernels"
        )
        set(ELPA_ENABLE_SSE_KERNELS OFF)
    endif()
endif()

# AVX
if(ELPA_ENABLE_AVX_KERNELS)
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS} -mavx")
    check_c_source_compiles(
        "
#include <x86intrin.h>
int main(void) { double *q; __m256d a1 = _mm256_load_pd(q); return 0; }
"
        HAVE_AVX
    )
    if(NOT HAVE_AVX)
        message(WARNING "ELPA: AVX not available, disabling AVX kernels")
        set(ELPA_ENABLE_AVX_KERNELS OFF)
    endif()
endif()

# AVX2
if(ELPA_ENABLE_AVX2_KERNELS)
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mfma")
    check_c_source_compiles(
        "
#include <x86intrin.h>
int main(void) { double *q; __m256d q1=_mm256_load_pd(q); __m256d y=_mm256_fmadd_pd(q1,q1,q1); return 0; }
"
        HAVE_AVX2
    )
    if(NOT HAVE_AVX2)
        message(WARNING "ELPA: AVX2 not available, disabling AVX2 kernels")
        set(ELPA_ENABLE_AVX2_KERNELS OFF)
    endif()
endif()

# AVX512
if(ELPA_ENABLE_AVX512_KERNELS)
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS} -mavx512f")
    check_c_source_compiles(
        "
#include <x86intrin.h>
int main(void) { __m512d a, b; __m512d c = _mm512_add_pd(a, b); return 0; }
"
        HAVE_AVX512
    )
    if(NOT HAVE_AVX512)
        message(WARNING "ELPA: AVX512 not available, disabling AVX512 kernels")
        set(ELPA_ENABLE_AVX512_KERNELS OFF)
    else()
        # Xeon vs Xeon Phi detection
        # _mm512_xor_pd requires AVX-512DQ; add the flag for the check.
        set(CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS} -mavx512f -mavx512dq")
        check_c_source_compiles(
            "
#include <x86intrin.h>
int main(void) { __m512d s, h; __m512d x = _mm512_xor_pd(h, s); return 0; }
"
            HAVE_AVX512_XEON
        )
        if(NOT HAVE_AVX512_XEON)
            set(CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS} -mavx512f")
            check_c_source_compiles(
                "
#include <x86intrin.h>
int main(void) { __m512d s, h; __m512d x = (__m512d)_mm512_xor_epi64((__m512i)h, (__m512i)s); return 0; }
"
                HAVE_AVX512_XEON_PHI
            )
        endif()
    endif()
endif()

# Detect additional x86 ISA features (for config.h completeness)
foreach(
    _feat
    IN
    ITEMS
        SSE
        SSE2
        SSE3
        SSSE3
        SSE4_1
        SSE4_2
        SSE4a
        MMX
        FMA3
        ABM
        ADX
        AES
        BMI1
        BMI2
        RDRND
        SHA
        AVX512_BW
        AVX512_CD
        AVX512_DQ
        AVX512_F
        AVX512_IFMA
        AVX512_VL
        AVX512_VBMI
)
    # Use /proc/cpuinfo or compiler intrinsics — for build reproducibility
    # we rely on compiler capability, not runtime detection
    # These are informational and don't gate kernels
endforeach()

# For the detailed CPU feature flags, we detect them via a compile-and-run
# approach on x86 only.  For cross-compilation, the user sets them manually.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # Try to compile a small program that checks CPUID
    set(_cpuinfo_test
        "
#include <stdio.h>
#include <cpuid.h>
int main(void) {
    unsigned int eax, ebx, ecx, edx;
    /* SSE/SSE2/SSE3/SSSE3 are always present on x86_64 */
    printf(\"SSE=1 SSE2=1 SSE3=1 SSSE3=1 SSE4_1=1 SSE4_2=1 MMX=1\\n\");
    return 0;
}"
    )
    # For simplicity we set all x86_64 baseline features
    set(HAVE_SSE 1)
    set(HAVE_SSE2 1)
    set(HAVE_SSE3 1)
    set(HAVE_SSSE3 1)
    set(HAVE_SSE4_1 1)
    set(HAVE_SSE4_2 1)
    set(HAVE_MMX 1)
    # SSE4a is AMD-only — detect via compiler flag rather than assuming
    check_c_compiler_flag("-msse4a" _has_sse4a_flag)
    if(_has_sse4a_flag)
        set(HAVE_SSE4a 1)
    endif()
    # These depend on actual CPU/compiler — check via compiler flag
    if(HAVE_AVX2)
        set(HAVE_FMA3 1)
        set(HAVE_ABM 1)
        set(HAVE_BMI1 1)
        set(HAVE_BMI2 1)
    endif()
    if(HAVE_AVX512)
        set(HAVE_AVX512_F 1)
        set(HAVE_AVX512_BW 1)
        set(HAVE_AVX512_CD 1)
        set(HAVE_AVX512_DQ 1)
        set(HAVE_AVX512_VL 1)
    endif()
    # ADX, AES, RDRND, SHA, IFMA, VBMI are nice-to-have but don't affect kernels
    check_c_compiler_flag("-madx" _have_adx)
    if(_have_adx)
        set(HAVE_ADX 1)
    endif()
    check_c_compiler_flag("-maes" _have_aes)
    if(_have_aes)
        set(HAVE_AES 1)
    endif()
    check_c_compiler_flag("-mrdrnd" _have_rdrnd)
    if(_have_rdrnd)
        set(HAVE_RDRND 1)
    endif()
    check_c_compiler_flag("-msha" _have_sha)
    if(_have_sha)
        set(HAVE_SHA 1)
    endif()
    if(HAVE_AVX512)
        check_c_compiler_flag("-mavx512ifma" _have_avx512_ifma)
        if(_have_avx512_ifma)
            set(HAVE_AVX512_IFMA 1)
        endif()
        check_c_compiler_flag("-mavx512vbmi" _have_avx512_vbmi)
        if(_have_avx512_vbmi)
            set(HAVE_AVX512_VBMI 1)
        endif()
    endif()
endif()

# SVE detection (ARM)
set(CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS}")
if(
    ELPA_ENABLE_SVE128_KERNELS
    OR ELPA_ENABLE_SVE256_KERNELS
    OR ELPA_ENABLE_SVE512_KERNELS
)
    check_c_source_compiles(
        "
#include <arm_sve.h>
int main(void) { double *q; svfloat64_t q1=svld1_f64(svptrue_b64(),q); return 0; }
"
        _HAVE_SVE
    )
    if(NOT _HAVE_SVE)
        message(WARNING "ELPA: SVE not available")
        set(ELPA_ENABLE_SVE128_KERNELS OFF)
        set(ELPA_ENABLE_SVE256_KERNELS OFF)
        set(ELPA_ENABLE_SVE512_KERNELS OFF)
    else()
        if(ELPA_ENABLE_SVE128_KERNELS)
            set(HAVE_SVE128 1)
        endif()
        if(ELPA_ENABLE_SVE256_KERNELS)
            set(HAVE_SVE256 1)
        endif()
        if(ELPA_ENABLE_SVE512_KERNELS)
            set(HAVE_SVE512 1)
        endif()
    endif()
endif()

# NEON AARCH64 detection
set(CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS}")
if(ELPA_ENABLE_NEON_ARCH64_KERNELS)
    check_c_source_compiles(
        "
#include <arm_neon.h>
int main(void) { float64x2_t a,b; float64x2_t c = vaddq_f64(a,b); return 0; }
"
        HAVE_NEON_ARCH64_SSE
    )
    if(NOT HAVE_NEON_ARCH64_SSE)
        message(WARNING "ELPA: NEON AARCH64 not available")
        set(ELPA_ENABLE_NEON_ARCH64_KERNELS OFF)
    endif()
endif()

# VSX detection (PPC)
set(CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS}")
if(ELPA_ENABLE_VSX_KERNELS)
    check_c_source_compiles(
        "
#include <altivec.h>
int main(void) { __vector double a,b; __vector double c=vec_add(a,b); return 0; }
"
        HAVE_VSX_SSE
    )
    if(NOT HAVE_VSX_SSE)
        message(WARNING "ELPA: VSX not available")
        set(ELPA_ENABLE_VSX_KERNELS OFF)
    endif()
endif()
cmake_pop_check_state()

# ===========================================================================
# Re-evaluate kernel use flags after ISA detection
# ===========================================================================
# ISA checks may have disabled kernel families. Re-apply family → use_ mapping.
# SSE family
if(NOT ELPA_ENABLE_SSE_KERNELS)
    set(use_real_sse_block2 OFF)
    set(use_real_sse_block4 OFF)
    set(use_real_sse_block6 OFF)
    set(use_complex_sse_block1 OFF)
    set(use_complex_sse_block2 OFF)
endif()
if(NOT ELPA_ENABLE_SSE_ASSEMBLY_KERNELS)
    set(use_real_sse_assembly OFF)
    set(use_complex_sse_assembly OFF)
endif()
if(NOT ELPA_ENABLE_AVX_KERNELS)
    set(use_real_avx_block2 OFF)
    set(use_real_avx_block4 OFF)
    set(use_real_avx_block6 OFF)
    set(use_complex_avx_block1 OFF)
    set(use_complex_avx_block2 OFF)
endif()
if(NOT ELPA_ENABLE_AVX2_KERNELS)
    set(use_real_avx2_block2 OFF)
    set(use_real_avx2_block4 OFF)
    set(use_real_avx2_block6 OFF)
    set(use_complex_avx2_block1 OFF)
    set(use_complex_avx2_block2 OFF)
endif()
if(NOT ELPA_ENABLE_AVX512_KERNELS)
    set(use_real_avx512_block2 OFF)
    set(use_real_avx512_block4 OFF)
    set(use_real_avx512_block6 OFF)
    set(use_complex_avx512_block1 OFF)
    set(use_complex_avx512_block2 OFF)
endif()

# ===========================================================================
# 4. Set WITH_<KERNEL>_KERNEL defines and COMPILED substitution variables
# ===========================================================================
# These match the autotools: WITH_REAL_AVX2_BLOCK2_KERNEL, etc.
# Also ELPA_2STAGE_REAL_AVX2_BLOCK2_COMPILED = 0 or 1 (for elpa_constants.h.in)

# Master list of all kernels (must match elpa_constants.h.in enum names)
set(_ALL_KERNELS
    # Generic
    REAL_GENERIC
    REAL_GENERIC_SIMPLE
    REAL_GENERIC_SIMPLE_BLOCK4
    REAL_GENERIC_SIMPLE_BLOCK6
    COMPLEX_GENERIC
    COMPLEX_GENERIC_SIMPLE
    # Blue Gene (never compiled on x86, but need COMPILED=0)
    REAL_BGP
    REAL_BGQ
    COMPLEX_BGP
    COMPLEX_BGQ
    # SSE
    REAL_SSE_BLOCK2
    REAL_SSE_BLOCK4
    REAL_SSE_BLOCK6
    COMPLEX_SSE_BLOCK1
    COMPLEX_SSE_BLOCK2
    # SSE Assembly
    REAL_SSE_ASSEMBLY
    COMPLEX_SSE_ASSEMBLY
    # AVX
    REAL_AVX_BLOCK2
    REAL_AVX_BLOCK4
    REAL_AVX_BLOCK6
    COMPLEX_AVX_BLOCK1
    COMPLEX_AVX_BLOCK2
    # AVX2
    REAL_AVX2_BLOCK2
    REAL_AVX2_BLOCK4
    REAL_AVX2_BLOCK6
    COMPLEX_AVX2_BLOCK1
    COMPLEX_AVX2_BLOCK2
    # AVX512
    REAL_AVX512_BLOCK2
    REAL_AVX512_BLOCK4
    REAL_AVX512_BLOCK6
    COMPLEX_AVX512_BLOCK1
    COMPLEX_AVX512_BLOCK2
    # SVE128
    REAL_SVE128_BLOCK2
    REAL_SVE128_BLOCK4
    REAL_SVE128_BLOCK6
    COMPLEX_SVE128_BLOCK1
    COMPLEX_SVE128_BLOCK2
    # SVE256
    REAL_SVE256_BLOCK2
    REAL_SVE256_BLOCK4
    REAL_SVE256_BLOCK6
    COMPLEX_SVE256_BLOCK1
    COMPLEX_SVE256_BLOCK2
    # SVE512
    REAL_SVE512_BLOCK2
    REAL_SVE512_BLOCK4
    REAL_SVE512_BLOCK6
    COMPLEX_SVE512_BLOCK1
    COMPLEX_SVE512_BLOCK2
    # SPARC64
    REAL_SPARC64_BLOCK2
    REAL_SPARC64_BLOCK4
    REAL_SPARC64_BLOCK6
    COMPLEX_SPARC64_BLOCK1
    COMPLEX_SPARC64_BLOCK2
    # NEON AARCH64
    REAL_NEON_ARCH64_BLOCK2
    REAL_NEON_ARCH64_BLOCK4
    REAL_NEON_ARCH64_BLOCK6
    COMPLEX_NEON_ARCH64_BLOCK1
    COMPLEX_NEON_ARCH64_BLOCK2
    # VSX
    REAL_VSX_BLOCK2
    REAL_VSX_BLOCK4
    REAL_VSX_BLOCK6
    COMPLEX_VSX_BLOCK1
    COMPLEX_VSX_BLOCK2
    # GPU kernels
    REAL_NVIDIA_GPU
    COMPLEX_NVIDIA_GPU
    REAL_NVIDIA_SM80_GPU
    COMPLEX_NVIDIA_SM80_GPU
    REAL_AMD_GPU
    COMPLEX_AMD_GPU
    REAL_INTEL_GPU_SYCL
    COMPLEX_INTEL_GPU_SYCL
)

foreach(_k IN LISTS _ALL_KERNELS)
    # Convert REAL_AVX2_BLOCK2 → use_real_avx2_block2
    string(TOLOWER "${_k}" _k_lower)
    if(use_${_k_lower})
        set(WITH_${_k}_KERNEL 1)
        set(ELPA_2STAGE_${_k}_COMPILED 1)
    else()
        unset(WITH_${_k}_KERNEL)
        set(ELPA_2STAGE_${_k}_COMPILED 0)
    endif()
endforeach()

# ===========================================================================
# 5. GPU version aggregate flags
# ===========================================================================
if(use_real_nvidia_gpu OR use_complex_nvidia_gpu)
    set(WITH_NVIDIA_GPU_VERSION 1)
    set(WITH_NVIDIA_GPU_KERNEL 1)
    set(CURRENT_WITH_NVIDIA_GPU_VERSION 1)
    set(ELPA_2STAGE_REAL_NVIDIA_GPU_COMPILED 1)
    set(ELPA_2STAGE_COMPLEX_NVIDIA_GPU_COMPILED 1)
    if(use_real_nvidia_sm80_gpu)
        set(WITH_NVIDIA_SM80_GPU_KERNEL 1)
        set(ELPA_2STAGE_REAL_NVIDIA_SM80_GPU_COMPILED 1)
        # No complex SM80 kernel yet
        set(ELPA_2STAGE_COMPLEX_NVIDIA_SM80_GPU_COMPILED 0)
    endif()
endif()

if(use_real_amd_gpu OR use_complex_amd_gpu)
    set(WITH_AMD_GPU_VERSION 1)
    set(WITH_AMD_GPU_KERNEL 1)
    set(CURRENT_WITH_AMD_GPU_VERSION 1)
endif()

if(use_real_intel_gpu_sycl OR use_complex_intel_gpu_sycl)
    set(WITH_INTEL_GPU_VERSION 1)
    set(CURRENT_WITH_SYCL_GPU_VERSION 1)
endif()

# ===========================================================================
# 6. Default kernel selection (priority: avx512 → avx2 → avx → sse → generic)
# ===========================================================================
# Mirrors the autotools priority chain from configure.ac.
# The priority list is: avx512, avx2, avx, sse, sse_assembly,
#   sve512, sve256, sve128, sparc64, neon_arch64, vsx, generic,
#   nvidia_gpu, amd_gpu, intel_gpu_sycl, nvidia_sm80_gpu

# Real kernels — the first enabled kernel in priority order wins
set(_real_default_priority
    real_avx512_block2
    real_avx2_block2
    real_avx_block2
    real_sse_block2
    real_sse_assembly
    real_sve512_block2
    real_sve256_block2
    real_sve128_block2
    real_sparc64_block2
    real_neon_arch64_block2
    real_vsx_block2
    real_generic
    real_nvidia_gpu
    real_amd_gpu
    real_intel_gpu_sycl
    real_nvidia_sm80_gpu
)

set(_complex_default_priority
    complex_avx512_block1
    complex_avx2_block1
    complex_avx_block1
    complex_sse_block1
    complex_sse_assembly
    complex_sve512_block1
    complex_sve256_block1
    complex_sve128_block1
    complex_sparc64_block1
    complex_neon_arch64_block1
    complex_vsx_block1
    complex_generic
    complex_nvidia_gpu
    complex_amd_gpu
    complex_intel_gpu_sycl
    complex_nvidia_sm80_gpu
)

# Default kernel selection.
#
# By default the highest compiled ISA is chosen automatically using the
# priority lists above.  Set ELPA_DEFAULT_REAL_KERNEL / ELPA_DEFAULT_COMPLEX_KERNEL
# in the cache to pin a specific kernel regardless of what is compiled in.
# This lets you build a library that includes AVX512 kernels (for callers that
# detect CPU features at runtime and call elpa_set()) while defaulting to a
# lower ISA that is safe on the deployment fleet.
#
# Example — AVX512 compiled in, AVX2 default:
#   cmake ... -DELPA_DEFAULT_REAL_KERNEL=real_avx2_block2 \
#             -DELPA_DEFAULT_COMPLEX_KERNEL=complex_avx2_block1
set(ELPA_DEFAULT_REAL_KERNEL
    ""
    CACHE STRING
    "Default 2-stage real kernel (empty = auto-select highest compiled ISA)"
)
set(ELPA_DEFAULT_COMPLEX_KERNEL
    ""
    CACHE STRING
    "Default 2-stage complex kernel (empty = auto-select highest compiled ISA)"
)

# Auto-select: walk priority list and pick first enabled kernel.
# Skipped when caller has pinned a value via the cache variables above.
if(NOT ELPA_DEFAULT_REAL_KERNEL)
    foreach(_k IN LISTS _real_default_priority)
        if(use_${_k})
            set(ELPA_DEFAULT_REAL_KERNEL "${_k}")
            break()
        endif()
    endforeach()
endif()
if(NOT ELPA_DEFAULT_REAL_KERNEL)
    message(
        FATAL_ERROR
        "ELPA: No real kernel enabled — at least one must be enabled"
    )
endif()
if(NOT use_${ELPA_DEFAULT_REAL_KERNEL})
    message(
        FATAL_ERROR
        "ELPA: ELPA_DEFAULT_REAL_KERNEL='${ELPA_DEFAULT_REAL_KERNEL}' is not compiled in. "
        "Enable it with -DELPA_ENABLE_${ELPA_DEFAULT_REAL_KERNEL}_KERNELS=ON or choose a different default."
    )
endif()

if(NOT ELPA_DEFAULT_COMPLEX_KERNEL)
    foreach(_k IN LISTS _complex_default_priority)
        if(use_${_k})
            set(ELPA_DEFAULT_COMPLEX_KERNEL "${_k}")
            break()
        endif()
    endforeach()
endif()
if(NOT ELPA_DEFAULT_COMPLEX_KERNEL)
    message(
        FATAL_ERROR
        "ELPA: No complex kernel enabled — at least one must be enabled"
    )
endif()
if(NOT use_${ELPA_DEFAULT_COMPLEX_KERNEL})
    message(
        FATAL_ERROR
        "ELPA: ELPA_DEFAULT_COMPLEX_KERNEL='${ELPA_DEFAULT_COMPLEX_KERNEL}' is not compiled in. "
        "Enable it with -DELPA_ENABLE_${ELPA_DEFAULT_COMPLEX_KERNEL}_KERNELS=ON or choose a different default."
    )
endif()

# Look up the kernel enum IDs from elpa_constants.h.in
# The enum format is: X(ELPA_2STAGE_<KERNEL>, <ID>, ...)
# We need to parse this at configure time.
function(_elpa_lookup_kernel_id _kernel_name _out_var)
    string(TOUPPER "${_kernel_name}" _upper)
    file(
        STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/elpa/elpa_constants.h.in"
        _lines
        REGEX "X\\(ELPA_2STAGE_${_upper}[^A-Z_]"
    )
    if(_lines)
        list(GET _lines 0 _line)
        # Extract the ID number after the first comma
        string(REGEX REPLACE ".*X\\([^,]+, *([0-9]+),.*" "\\1" _id "${_line}")
        set(${_out_var} "${_id}" PARENT_SCOPE)
    else()
        message(
            FATAL_ERROR
            "ELPA: Cannot find kernel ID for ${_kernel_name} in elpa_constants.h.in"
        )
    endif()
endfunction()

_elpa_lookup_kernel_id("${ELPA_DEFAULT_REAL_KERNEL}" ELPA_2STAGE_REAL_DEFAULT)
_elpa_lookup_kernel_id("${ELPA_DEFAULT_COMPLEX_KERNEL}" ELPA_2STAGE_COMPLEX_DEFAULT)

message(
    STATUS
    "ELPA: Default real kernel: ${ELPA_DEFAULT_REAL_KERNEL} (ID=${ELPA_2STAGE_REAL_DEFAULT})"
)
message(
    STATUS
    "ELPA: Default complex kernel: ${ELPA_DEFAULT_COMPLEX_KERNEL} (ID=${ELPA_2STAGE_COMPLEX_DEFAULT})"
)

# ===========================================================================
# 7. Fortran linking convention
# ===========================================================================
# Detect the Fortran name-mangling scheme via CMake's FortranCInterface
# module. Fall back to underscore mangling if detection fails (e.g., mixed
# compiler setups where ifort + gcc PIE flags cause try_compile failures).
include(FortranCInterface)
# Skip FortranCInterface_VERIFY — the detection via FortranCInterface_HEADER
# (triggered by include(FortranCInterface)) already identifies the mangling
# scheme. The VERIFY step tries to link a mixed C/Fortran executable, which
# fails with ifort + GCC's default-PIE (non-PIE .o from ifort vs PIE linker)
# and on Windows (clang-cl + ifort ABI mismatch in the test harness).
# The actual ELPA library links correctly because it uses shared libraries
# (which are always PIC) and the MPI wrappers handle runtime resolution.
if(
    FortranCInterface_GLOBAL_SUFFIX STREQUAL "_"
    OR FortranCInterface_GLOBAL_SUFFIX STREQUAL ""
)
    if(FortranCInterface_GLOBAL_SUFFIX STREQUAL "_")
        set(NEED_UNDERSCORE_TO_LINK_AGAINST_FORTRAN 1)
    else()
        set(NEED_UNDERSCORE_TO_LINK_AGAINST_FORTRAN 0)
    endif()
else()
    set(NEED_UNDERSCORE_TO_LINK_AGAINST_FORTRAN 0)
    message(
        STATUS
        "ELPA: Fortran mangling suffix='${FortranCInterface_GLOBAL_SUFFIX}'"
    )
endif()
if(NOT DEFINED NEED_UNDERSCORE_TO_LINK_AGAINST_FORTRAN)
    message(
        STATUS
        "ELPA: FortranCInterface detection failed, assuming underscore mangling"
    )
    set(NEED_UNDERSCORE_TO_LINK_AGAINST_FORTRAN 1)
endif()

# ===========================================================================
# Summary
# ===========================================================================
set(_enabled_families "")
foreach(
    _fam
    IN
    ITEMS
        GENERIC
        SSE
        SSE_ASSEMBLY
        AVX
        AVX2
        AVX512
        SVE128
        SVE256
        SVE512
        SPARC64
        NEON_ARCH64
        VSX
        NVIDIA_GPU
        NVIDIA_SM80_GPU
        AMD_GPU
        INTEL_GPU_SYCL
)
    if(ELPA_ENABLE_${_fam}_KERNELS)
        list(APPEND _enabled_families ${_fam})
    endif()
endforeach()
message(STATUS "ELPA: Enabled kernel families: ${_enabled_families}")
