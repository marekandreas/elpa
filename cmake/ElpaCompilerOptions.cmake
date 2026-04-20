set(ELPA_FRAMEWORK_ISA
    "native"
    CACHE STRING
    "Baseline SIMD ISA for framework (non-kernel) Fortran and C code (native, avx2, avx512)"
)
set_property(CACHE ELPA_FRAMEWORK_ISA PROPERTY STRINGS "native" "avx2" "avx512")

# On non-x86 platforms, force ISA to "native" — avx2/avx512 are meaningless.
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
    if(NOT ELPA_FRAMEWORK_ISA STREQUAL "native")
        message(
            STATUS
            "ELPA: Overriding ELPA_FRAMEWORK_ISA='${ELPA_FRAMEWORK_ISA}' → 'native' (non-x86 platform)"
        )
        set(ELPA_FRAMEWORK_ISA "native" CACHE STRING "" FORCE)
    endif()
endif()

# C/C++ language standards are expressed as target requirements via
# target_compile_features() in src/CMakeLists.txt so they do not fight
# with any CMake toolchain file that may also set language standards.

if(WIN32 AND CMAKE_C_COMPILER_ID STREQUAL "Clang")
    add_compile_definitions($<$<COMPILE_LANGUAGE:C>:complex=_Complex>)
endif()

set(ELPA_GCC_C_OPT_FLAGS "")
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(ELPA_GCC_C_OPT_FLAGS
        -funsafe-loop-optimizations
        -funsafe-math-optimizations
        -ftree-vect-loop-version
        -ftree-vectorize
    )
endif()

set(ELPA_FORTRAN_OPT_FLAGS "")
set(ELPA_FORTRAN_DEBUG_FLAGS "")
set(ELPA_FORTRAN_ISA_FLAGS "")

if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    if(WIN32)
        add_compile_options($<$<COMPILE_LANGUAGE:Fortran>:/heap-arrays>)
        set(ELPA_FORTRAN_OPT_FLAGS /Qunroll)
        set(ELPA_FORTRAN_DEBUG_FLAGS
            /check:all
            /check:bounds
            /check:uninit
            -traceback
        )
    else()
        add_compile_options($<$<COMPILE_LANGUAGE:Fortran>:-heap-arrays>)
        set(ELPA_FORTRAN_OPT_FLAGS -unroll)
        set(ELPA_FORTRAN_DEBUG_FLAGS
            -check
            all
            -check
            bounds
            -check
            uninit
            -traceback
        )
    endif()

    if(ELPA_FRAMEWORK_ISA STREQUAL "native")
        if(WIN32)
            set(ELPA_FORTRAN_ISA_FLAGS /QxHost)
        else()
            set(ELPA_FORTRAN_ISA_FLAGS -march=native)
        endif()
    elseif(ELPA_FRAMEWORK_ISA STREQUAL "avx512")
        if(WIN32)
            set(ELPA_FORTRAN_ISA_FLAGS /arch:SKYLAKE-AVX512)
        else()
            set(ELPA_FORTRAN_ISA_FLAGS -march=skylake-avx512)
        endif()
    else()
        if(WIN32)
            set(ELPA_FORTRAN_ISA_FLAGS /arch:CORE-AVX2)
        else()
            set(ELPA_FORTRAN_ISA_FLAGS -march=core-avx2)
        endif()
    endif()
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    if(ELPA_FRAMEWORK_ISA STREQUAL "native")
        set(ELPA_FORTRAN_ISA_FLAGS -march=native)
    elseif(ELPA_FRAMEWORK_ISA STREQUAL "avx512")
        set(ELPA_FORTRAN_ISA_FLAGS -mavx512f -mavx512dq -mfma)
    else()
        set(ELPA_FORTRAN_ISA_FLAGS -mavx2 -mfma)
    endif()
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "LLVMFlang")
    # LLVM flang-new only supports -march=, not individual -mavx2/-mfma flags.
    if(ELPA_FRAMEWORK_ISA STREQUAL "native")
        set(ELPA_FORTRAN_ISA_FLAGS -march=native)
    elseif(ELPA_FRAMEWORK_ISA STREQUAL "avx512")
        set(ELPA_FORTRAN_ISA_FLAGS -march=skylake-avx512)
    else()
        set(ELPA_FORTRAN_ISA_FLAGS -march=haswell)
    endif()
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Flang")
    # Classic Flang (AOCC) accepts GCC-compatible SIMD flags.
    if(ELPA_FRAMEWORK_ISA STREQUAL "native")
        set(ELPA_FORTRAN_ISA_FLAGS -march=native)
    elseif(ELPA_FRAMEWORK_ISA STREQUAL "avx512")
        set(ELPA_FORTRAN_ISA_FLAGS -mavx512f -mavx512dq -mfma)
    else()
        set(ELPA_FORTRAN_ISA_FLAGS -mavx2 -mfma)
    endif()
    # Classic Flang (AOCC) does not support -nocpp; CMake's Flang-Fortran module
    # sets CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_OFF to "-nocpp" but the
    # compiler rejects it.  Clear both the cache entry and the in-memory variable.
    if(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_OFF STREQUAL "-nocpp")
        set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_OFF "")
        set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_OFF "" CACHE STRING "" FORCE)
    endif()
endif()

if(ELPA_FRAMEWORK_ISA STREQUAL "native")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C>:-march=native>
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    )
endif()

# C++17 is required — enforced via target_compile_features(); no runtime check needed.
set(HAVE_CXX17 1)

# ── Compiler-specific workarounds (matching autotools configure.ac) ──────────

# IntelLLVM Fortran (ifx) has a bug with variable-length character function
# results in formatted I/O.  The autotools build gates this behind
# --enable-ifx-compiler; we auto-detect instead.
if(CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM")
    set(ENABLE_IFX_COMPILER 1)
endif()

# Both ifx (IntelLLVM) and classic ifort (Intel) link against Intel Fortran
# runtime libraries (libifport, libifcoremt, libsvml, libintlc) that live
# in the Intel compiler lib directory.  When the C compiler is gcc/clang,
# cmake propagates the Fortran implicit link libraries (-lifport, …) to
# C/CXX link commands, but NOT the Fortran implicit link directories.
# Inject -L and RPATH so the host linker (g++) can find these at build time
# and tests can find them at run time.
if(CMAKE_Fortran_COMPILER_ID MATCHES "IntelLLVM|Intel" AND NOT WIN32)
    get_filename_component(_ifx_bin_dir "${CMAKE_Fortran_COMPILER}" DIRECTORY)
    get_filename_component(_ifx_lib_dir "${_ifx_bin_dir}/../lib" ABSOLUTE)
    if(IS_DIRECTORY "${_ifx_lib_dir}")
        list(APPEND CMAKE_BUILD_RPATH "${_ifx_lib_dir}")
        add_link_options("-L${_ifx_lib_dir}")
        message(STATUS "ELPA: Intel Fortran runtime rpath + link dir: ${_ifx_lib_dir}")
    endif()
    unset(_ifx_bin_dir)
    unset(_ifx_lib_dir)
endif()

# PGI / NVIDIA HPC SDK (nvfortran) has a bug where c_f_pointer to a
# variable-length character string cannot be used directly as a function
# result.  The source works around it with a local copy.
if(CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
    set(PGI_VARIABLE_STRING_BUG 1)
endif()
