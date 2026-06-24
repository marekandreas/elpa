# SPDX-License-Identifier: LGPL-3.0-or-later

include_guard(GLOBAL)

include(CheckCCompilerFlag)
include(CheckCSourceCompiles)
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
include(CheckFortranCompilerFlag)
include(CheckFortranSourceCompiles)
include(CMakePushCheckState)
include(FortranCInterface)

# Check Fortran behavior that affects ELPA's source selection.
function(_elpa_check_fortran_support)
  check_fortran_source_compiles(
    [[
      program main
        use, intrinsic :: iso_fortran_env, only : int64
        integer(int64) :: value
        value = 0_int64
      end program
    ]]
    ELPA_HAVE_ISO_FORTRAN_ENV
    SRC_EXT F90
  )

  # A compiler that accepts these calls does not compare actual argument types
  # across calls to an external procedure without an explicit interface. This
  # is the same distinction used by the Autotools build for PACK_REAL_TO_COMPLEX.
  check_fortran_source_compiles(
    [[
      program main
        implicit none
        integer :: integer_value
        real(kind=8) :: real_value
        call external_procedure(integer_value)
        call external_procedure(real_value)
      end program
    ]]
    ELPA_FORTRAN_ACCEPTS_ARGUMENT_MISMATCH
    SRC_EXT F90
  )

  set(
    ELPA_PACK_REAL_TO_COMPLEX
    "${ELPA_FORTRAN_ACCEPTS_ARGUMENT_MISMATCH}"
    CACHE INTERNAL "Whether ELPA may use the real-to-complex packing path"
    FORCE
  )

  check_fortran_source_compiles(
    [[
      program main
        implicit none
        character(len=1) :: value
        call get_environment_variable("PATH", value)
      end program
    ]]
    ELPA_HAVE_ENVIRONMENT_CHECKING
    SRC_EXT F90
  )
endfunction()

# Check the non-standard affinity interfaces used by ELPA on Linux systems.
function(_elpa_check_affinity_support)
  if(NOT ELPA_ENABLE_AFFINITY)
    return()
  endif()

  check_c_source_compiles(
    [[
      #define _GNU_SOURCE
      #include <pthread.h>
      #include <sched.h>

      int main(void) {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(0, &mask);
        return pthread_getaffinity_np(pthread_self(), sizeof(mask), &mask);
      }
    ]]
    ELPA_HAVE_AFFINITY_SUPPORT
  )

  if(NOT ELPA_HAVE_AFFINITY_SUPPORT)
    message(
      FATAL_ERROR
        "ELPA_ENABLE_AFFINITY=ON, but the required pthread affinity interfaces "
        "are unavailable. Configure with -DELPA_ENABLE_AFFINITY=OFF on this platform."
    )
  endif()
endfunction()

# Check which Fortran MPI interface is available and whether it covers the
# choice-buffer types used by ELPA. The existence of the mpi module and the
# need for -fallow-argument-mismatch are independent properties.
function(_elpa_check_mpi_support)
  # The MPI implementation may be changed in an existing build tree. Do not
  # reuse interface results obtained from a previously selected implementation.
  foreach(_elpa_mpi_check IN ITEMS
    ELPA_HAVE_MPI_MODULE
    ELPA_HAVE_MPIF_H
    ELPA_MPI_HAS_COMPLETE_CHOICE_BUFFER_INTERFACES
    ELPA_MPI_INTERFACES_WORK_WITH_ARGUMENT_MISMATCH_FLAG
    ELPA_HAVE_CUDA_AWARE_MPI
  )
    unset(${_elpa_mpi_check} CACHE)
  endforeach()

  set(
    ELPA_MPI_ARGUMENT_MISMATCH_FLAG
    ""
    CACHE INTERNAL "Compatibility option required by the detected MPI interfaces"
    FORCE
  )

  if(NOT ELPA_USE_MPI)
    set(ELPA_HAVE_MPI_MODULE FALSE CACHE INTERNAL "Fortran MPI module availability" FORCE)
    return()
  endif()

  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_Fortran)

  check_fortran_source_compiles(
    [[
      program main
        use mpi
        implicit none
        real :: time
        time = MPI_Wtime()
      end program
    ]]
    ELPA_HAVE_MPI_MODULE
    SRC_EXT F90
  )

  if(ELPA_HAVE_MPI_MODULE)
    set(
      _elpa_mpi_choice_buffer_test
      [[
        program main
          use mpi
          implicit none
          integer :: error
          real(kind=8) :: real_buffer(2)
          complex(kind=8) :: complex_buffer(2)
          call MPI_Bcast(real_buffer, 2, MPI_REAL8, 0, MPI_COMM_WORLD, error)
          call MPI_Bcast(complex_buffer, 2, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD, error)
        end program
      ]]
    )
  else()
    check_fortran_source_compiles(
      [[
        program main
          implicit none
          include 'mpif.h'
          integer :: error
          real :: time
          time = MPI_Wtime()
        end program
      ]]
      ELPA_HAVE_MPIF_H
      SRC_EXT F90
    )

    if(NOT ELPA_HAVE_MPIF_H)
      message(
        FATAL_ERROR
          "The detected MPI implementation provides neither a usable Fortran mpi module nor mpif.h."
      )
    endif()

    set(
      _elpa_mpi_choice_buffer_test
      [[
        program main
          implicit none
          include 'mpif.h'
          integer :: error
          real(kind=8) :: real_buffer(2)
          complex(kind=8) :: complex_buffer(2)
          call MPI_Bcast(real_buffer, 2, MPI_REAL8, 0, MPI_COMM_WORLD, error)
          call MPI_Bcast(complex_buffer, 2, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD, error)
        end program
      ]]
    )
  endif()

  check_fortran_source_compiles(
    "${_elpa_mpi_choice_buffer_test}"
    ELPA_MPI_HAS_COMPLETE_CHOICE_BUFFER_INTERFACES
    SRC_EXT F90
  )

  if(NOT ELPA_MPI_HAS_COMPLETE_CHOICE_BUFFER_INTERFACES)
    check_fortran_compiler_flag(
      "-fallow-argument-mismatch"
      ELPA_Fortran_HAS_ALLOW_ARGUMENT_MISMATCH
    )

    if(ELPA_Fortran_HAS_ALLOW_ARGUMENT_MISMATCH)
      set(CMAKE_REQUIRED_FLAGS "-fallow-argument-mismatch")
      check_fortran_source_compiles(
        "${_elpa_mpi_choice_buffer_test}"
        ELPA_MPI_INTERFACES_WORK_WITH_ARGUMENT_MISMATCH_FLAG
        SRC_EXT F90
      )
    endif()

    if(ELPA_MPI_INTERFACES_WORK_WITH_ARGUMENT_MISMATCH_FLAG)
      set(
        ELPA_MPI_ARGUMENT_MISMATCH_FLAG
        "-fallow-argument-mismatch"
        CACHE INTERNAL "Compatibility option required by the detected MPI interfaces"
        FORCE
      )
    else()
      message(
        FATAL_ERROR
          "The Fortran compiler checks procedure arguments, but the detected MPI Fortran "
          "interfaces do not support every choice-buffer type used by ELPA."
      )
    endif()
  endif()

  cmake_pop_check_state()

  if(ELPA_ENABLE_CUDA_AWARE_MPI)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_C)
    check_c_source_compiles(
      [[
        #include <mpi.h>
        #include <mpi-ext.h>

        #if !(defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT)
        #error The MPI implementation is not CUDA-aware.
        #endif

        int main(void) { return 0; }
      ]]
      ELPA_HAVE_CUDA_AWARE_MPI
    )
    cmake_pop_check_state()

    if(NOT ELPA_HAVE_CUDA_AWARE_MPI)
      message(
        FATAL_ERROR
          "ELPA_ENABLE_CUDA_AWARE_MPI=ON, but the detected MPI implementation does not "
          "advertise CUDA-aware support through mpi-ext.h."
      )
    endif()
  endif()
endfunction()

# Compile an intrinsic test with options that will later be attached only to
# the corresponding kernel source files.
function(_elpa_check_c_kernel variable flags source)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_FLAGS "${flags}")
  check_c_source_compiles("${source}" "${variable}")
  cmake_pop_check_state()
endfunction()

# Check architecture-specific CPU kernels supported by the target architecture.
function(_elpa_check_cpu_kernel_support)
  if(ELPA_CPU_KERNELS STREQUAL "GENERIC")
    return()
  endif()

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    _elpa_check_c_kernel(
      ELPA_C_HAS_SSE3
      "-msse3"
      [[
        #include <x86intrin.h>
        int main(void) {
          double value[2] = {0.0, 0.0};
          __m128d vector = _mm_loaddup_pd(value);
          return _mm_cvtsd_f64(vector) != 0.0;
        }
      ]]
    )
    _elpa_check_c_kernel(
      ELPA_C_HAS_AVX
      "-mavx"
      [[
        #include <x86intrin.h>
        int main(void) {
          double value[4] = {0.0, 0.0, 0.0, 0.0};
          __m256d vector = _mm256_loadu_pd(value);
          return _mm256_movemask_pd(vector);
        }
      ]]
    )
    _elpa_check_c_kernel(
      ELPA_C_HAS_AVX2
      "-mavx2 -mfma"
      [[
        #include <x86intrin.h>
        int main(void) {
          __m256d zero = _mm256_setzero_pd();
          __m256d result = _mm256_fmadd_pd(zero, zero, zero);
          return _mm256_movemask_pd(result);
        }
      ]]
    )
    _elpa_check_c_kernel(
      ELPA_C_HAS_AVX512
      "-mavx512f -mavx512dq -mavx512vl -mfma"
      [[
        #include <x86intrin.h>
        int main(void) {
          __m512d zero = _mm512_setzero_pd();
          __m512d result = _mm512_fmadd_pd(zero, zero, zero);
          return _mm512_cmp_pd_mask(result, zero, _CMP_NEQ_OQ);
        }
      ]]
    )
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    _elpa_check_c_kernel(
      ELPA_C_HAS_NEON_ARCH64
      ""
      [[
        #include <arm_neon.h>
        int main(void) {
          float64x2_t value = vdupq_n_f64(0.0);
          value = vfmaq_f64(value, value, value);
          return vgetq_lane_f64(value, 0) != 0.0;
        }
      ]]
    )
    foreach(_elpa_sve_width IN ITEMS 128 256 512)
      _elpa_check_c_kernel(
        "ELPA_C_HAS_SVE${_elpa_sve_width}"
        "-march=armv8-a+sve -msve-vector-bits=${_elpa_sve_width}"
        [[
          #include <arm_sve.h>
          int main(void) {
            svfloat64_t value = svdup_n_f64(0.0);
            return svptest_any(svptrue_b64(), svcmpne_n_f64(svptrue_b64(), value, 0.0));
          }
        ]]
      )
    endforeach()
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ppc64|ppc64le|powerpc64|powerpc64le)$")
    _elpa_check_c_kernel(
      ELPA_C_HAS_VSX
      "-mvsx"
      [[
        #include <altivec.h>
        int main(void) {
          __vector double left = {0.0, 0.0};
          __vector double right = {0.0, 0.0};
          __vector double result = vec_add(left, right);
          return vec_extract(result, 0) != 0.0;
        }
      ]]
    )
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(sparc64|sparc)$")
    _elpa_check_c_kernel(
      ELPA_C_HAS_SPARC64
      ""
      [[
        #include <fjmfunc.h>
        #include <emmintrin.h>
        int main(void) {
          __m128d value = _mm_setzero_pd();
          value = _fjsp_neg_v2r8(value);
          return 0;
        }
      ]]
    )
  endif()
endfunction()

# Check compiler support required by the selected accelerator backend.
function(_elpa_check_accelerator_support)
  set(
    ELPA_ACCELERATOR_COMPILE_OPTIONS
    ""
    CACHE INTERNAL "Compiler options required by the selected accelerator backend"
    FORCE
  )
  set(
    ELPA_ACCELERATOR_LINK_OPTIONS
    ""
    CACHE INTERNAL "Link options required by the selected accelerator backend"
    FORCE
  )

  if(ELPA_ACCELERATOR STREQUAL "SYCL")
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_FLAGS "-fsycl")
    set(CMAKE_REQUIRED_LIBRARIES MKL::SYCL)
    check_cxx_source_compiles(
      [[
        #include <sycl/sycl.hpp>
        #include <oneapi/mkl.hpp>
        int main() {
          sycl::queue queue;
          return queue.get_device().is_gpu() ? 0 : 0;
        }
      ]]
      ELPA_CXX_HAS_SYCL
    )
    cmake_pop_check_state()

    if(NOT ELPA_CXX_HAS_SYCL)
      message(
        FATAL_ERROR
          "ELPA_ACCELERATOR=SYCL requires a SYCL-capable C++ compiler and oneMKL SYCL headers."
      )
    endif()

    set(
      ELPA_ACCELERATOR_COMPILE_OPTIONS
      "-fsycl"
      CACHE INTERNAL "Compiler options required by the selected accelerator backend"
      FORCE
    )
    set(
      ELPA_ACCELERATOR_LINK_OPTIONS
      "-fsycl"
      CACHE INTERNAL "Link options required by the selected accelerator backend"
      FORCE
    )
  elseif(ELPA_ACCELERATOR STREQUAL "OPENMP")
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_FLAGS "-fiopenmp -fopenmp-targets=spir64")
    set(CMAKE_REQUIRED_LIBRARIES MKL::MKL)
    check_cxx_source_compiles(
      [[
        #include <mkl.h>
        #include <mkl_omp_offload.h>
        #include <omp.h>
        int main() { return omp_get_num_devices() < 0; }
      ]]
      ELPA_CXX_HAS_OPENMP_OFFLOAD
    )
    cmake_pop_check_state()

    if(NOT ELPA_CXX_HAS_OPENMP_OFFLOAD)
      message(
        FATAL_ERROR
          "ELPA_ACCELERATOR=OPENMP requires Intel OpenMP target offloading "
          "and oneMKL offload headers."
      )
    endif()

    set(
      ELPA_ACCELERATOR_COMPILE_OPTIONS
      "-fiopenmp;-fopenmp-targets=spir64"
      CACHE INTERNAL "Compiler options required by the selected accelerator backend"
      FORCE
    )
    set(
      ELPA_ACCELERATOR_LINK_OPTIONS
      "-fiopenmp;-fopenmp-targets=spir64"
      CACHE INTERNAL "Link options required by the selected accelerator backend"
      FORCE
    )
  endif()
endfunction()

# Run all compiler and ABI checks required before ELPA targets are created.
function(elpa_check_compiler_support)
  set(_elpa_saved_try_compile_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

  _elpa_check_fortran_support()
  _elpa_check_affinity_support()
  _elpa_check_mpi_support()
  _elpa_check_cpu_kernel_support()
  _elpa_check_accelerator_support()

  if(_elpa_saved_try_compile_target_type)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_elpa_saved_try_compile_target_type}")
  else()
    unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
  endif()

  FortranCInterface_VERIFY()
  set(
    ELPA_FORTRAN_GLOBAL_SUFFIX
    "${FortranCInterface_GLOBAL_SUFFIX}"
    CACHE INTERNAL "Fortran global-symbol suffix"
    FORCE
  )
endfunction()
