    # ElpaCUDA.cmake — NVIDIA CUDA configuration for ELPA
#
# Handles CUDA compiler setup, compute capability selection, library detection
# (cublas, cublasLt, cudart, cusolver, nccl), and GPU-specific defines.
#
# Requires: ELPA_CUDA=ON passed to the top-level project.

if(NOT ELPA_CUDA)
    return()
endif()

# ---------------------------------------------------------------------------
# Enable the CUDA language
# ---------------------------------------------------------------------------
# On Windows with clang-cl as the C/C++ compiler, nvcc still requires MSVC
# cl.exe as its host compiler.  CMake may not find cl.exe if it is not in
# PATH (e.g. when the VS environment is not activated).  Detect it via
# vswhere and set CMAKE_CUDA_HOST_COMPILER before enable_language(CUDA).
#
# Concurrency note: this block must ALWAYS set CMAKE_CUDA_HOST_COMPILER when
# on Windows so that the path is embedded into the generated ninja rules via
# -ccbin.  Without -ccbin, nvcc falls back to a PATH search at BUILD time,
# but a fresh shell (e.g. cmake --build) may not have MSVC on PATH.
if(WIN32)
    # Re-run detection whenever the cached value is absent or empty so that
    # a stale build tree (configured without cl.exe on PATH) self-corrects.
    if(NOT CMAKE_CUDA_HOST_COMPILER)
        execute_process(
            COMMAND
                "C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
                -products * -latest -property installationPath
            OUTPUT_VARIABLE _vs_install_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_vs_install_dir)
            file(GLOB _msvc_versions "${_vs_install_dir}/VC/Tools/MSVC/*")
            list(SORT _msvc_versions ORDER DESCENDING)
            list(GET _msvc_versions 0 _msvc_latest)
            set(_cl_path "${_msvc_latest}/bin/Hostx64/x64/cl.exe")
            if(EXISTS "${_cl_path}")
                set(CMAKE_CUDA_HOST_COMPILER
                    "${_cl_path}"
                    CACHE FILEPATH
                    "MSVC cl.exe for nvcc host compilation"
                    FORCE
                )
                message(
                    STATUS
                    "ELPA CUDA: using MSVC host compiler ${_cl_path}"
                )
            endif()
        endif()
        unset(_vs_install_dir)
        unset(_msvc_versions)
        unset(_msvc_latest)
        unset(_cl_path)
    endif()

    # Hard error if cl.exe could not be located — failing here is far more
    # useful than a cryptic nvcc 'Cannot find compiler cl.exe in PATH' at
    # build time.
    if(NOT CMAKE_CUDA_HOST_COMPILER)
        message(
            FATAL_ERROR
            "ELPA CUDA: Could not find MSVC cl.exe via vswhere on Windows.\n"
            "nvcc requires cl.exe as its host compiler — it cannot use clang-cl.\n"
            "Options:\n"
            "  1. Install Visual Studio Build Tools (includes MSVC).\n"
            "  2. Set CMAKE_CUDA_HOST_COMPILER manually:\n"
            "       -DCMAKE_CUDA_HOST_COMPILER=<path/to/cl.exe>"
        )
    endif()
endif()

# ---------------------------------------------------------------------------
# Linux / non-Windows: nvcc requires a GCC-compatible host compiler.
# When the project C compiler is Clang, nvcc cannot parse Clang-flavoured
# system headers (e.g. __real__() in <complex.h>), so default to gcc.
# ---------------------------------------------------------------------------
if(NOT WIN32 AND NOT CMAKE_CUDA_HOST_COMPILER)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang|LLVMFlang")
        find_program(_gcc_for_cuda gcc)
        if(_gcc_for_cuda)
            set(CMAKE_CUDA_HOST_COMPILER
                "${_gcc_for_cuda}"
                CACHE FILEPATH
                "GCC as nvcc host compiler (Clang is not supported by nvcc)"
                FORCE
            )
            message(STATUS "ELPA CUDA: using GCC host compiler for nvcc: ${_gcc_for_cuda}")
        else()
            message(WARNING
                "ELPA CUDA: CC is Clang but gcc not found in PATH.\n"
                "nvcc may fail — set CMAKE_CUDA_HOST_COMPILER to a GCC executable."
            )
        endif()
        unset(_gcc_for_cuda)
    endif()
endif()

# ---------------------------------------------------------------------------
# Compute capabilities — cache declaration (must precede enable_language so
# "native" triggers host-GPU detection during the configure step).
# ---------------------------------------------------------------------------
# Accepts:
#   "native"      auto-detect GPU architecture(s) at configure time.
#   "75;80;90"    semicolon-separated explicit SM numbers: SASS for each,
#                 PTX only for the highest arch (JIT-able on future GPUs).
set(ELPA_CUDA_ARCHITECTURES
    "native"
    CACHE STRING
    "CUDA compute capabilities: semicolon-separated SM list or \"native\" to auto-detect the host GPU"
)
if(ELPA_CUDA_ARCHITECTURES STREQUAL "native")
    # Set CMAKE_CUDA_ARCHITECTURES before enable_language(CUDA) so CMake
    # queries the host GPU during the configure step and populates
    # CMAKE_CUDA_ARCHITECTURES_NATIVE with the detected SM number(s).
    set(CMAKE_CUDA_ARCHITECTURES
        native
        CACHE STRING "NVIDIA GPU architectures to compile for" FORCE
    )
    message(STATUS "ELPA CUDA: native GPU architecture detection requested")
endif()

enable_language(CUDA)

# Validate: on Windows, nvcc requires MSVC cl.exe as host compiler.
# clang-cl is not supported by nvcc.
if(WIN32 AND CMAKE_CUDA_HOST_COMPILER)
    get_filename_component(_cuda_host_name "${CMAKE_CUDA_HOST_COMPILER}" NAME)
    if(NOT _cuda_host_name STREQUAL "cl.exe")
        message(
            FATAL_ERROR
            "ELPA CUDA: nvcc on Windows requires MSVC cl.exe as host compiler, "
            "but CMAKE_CUDA_HOST_COMPILER is set to '${CMAKE_CUDA_HOST_COMPILER}'.\n"
            "The vswhere auto-detection above should have found cl.exe. "
            "Check your Visual Studio installation."
        )
    endif()
endif()

# ---------------------------------------------------------------------------
# Compute capabilities — resolve and detect SM80
# ---------------------------------------------------------------------------
if(ELPA_CUDA_ARCHITECTURES STREQUAL "native")
    # CMake populated CMAKE_CUDA_ARCHITECTURES_NATIVE after enable_language(CUDA).
    # Use it to determine whether the host GPU supports SM80+ kernel paths.
    if(CMAKE_CUDA_ARCHITECTURES_NATIVE)
        set(_elpa_cuda_resolved "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
    else()
        # Fallback: CMake < 3.24 may not set CMAKE_CUDA_ARCHITECTURES_NATIVE.
        set(_elpa_cuda_resolved "${CMAKE_CUDA_ARCHITECTURES}")
    endif()
    message(STATUS "ELPA CUDA: native architecture(s): ${_elpa_cuda_resolved}")
    set(_elpa_cuda_max_arch 0)
    foreach(_arch IN LISTS _elpa_cuda_resolved)
        if(_arch GREATER _elpa_cuda_max_arch)
            set(_elpa_cuda_max_arch ${_arch})
        endif()
    endforeach()
else()
    # Explicit SM list: build SASS for every arch, PTX only for the highest
    # arch so the CUDA driver can JIT-compile for future GPUs.
    list(SORT ELPA_CUDA_ARCHITECTURES COMPARE NATURAL)
    list(GET ELPA_CUDA_ARCHITECTURES -1 _elpa_cuda_highest_arch)

    set(_elpa_cuda_arch_list "")
    foreach(_arch IN LISTS ELPA_CUDA_ARCHITECTURES)
        if(_arch STREQUAL _elpa_cuda_highest_arch)
            # Highest arch: embed both SASS and PTX (CMake default for plain number)
            list(APPEND _elpa_cuda_arch_list "${_arch}")
        else()
            # Lower arches: SASS only (no PTX — saves binary size)
            list(APPEND _elpa_cuda_arch_list "${_arch}-real")
        endif()
    endforeach()
    set(CMAKE_CUDA_ARCHITECTURES ${_elpa_cuda_arch_list})

    set(_elpa_cuda_max_arch 0)
    foreach(_arch IN LISTS ELPA_CUDA_ARCHITECTURES)
        if(_arch GREATER _elpa_cuda_max_arch)
            set(_elpa_cuda_max_arch ${_arch})
        endif()
    endforeach()
endif()

if(_elpa_cuda_max_arch GREATER_EQUAL 80)
    set(WITH_NVIDIA_GPU_SM80_COMPUTE_CAPABILITY 1)
    message(STATUS "ELPA CUDA: SM80+ (A100) kernel support enabled")
else()
    set(WITH_NVIDIA_GPU_SM80_COMPUTE_CAPABILITY 0)
endif()

# ---------------------------------------------------------------------------
# CUDA compiler flags
# ---------------------------------------------------------------------------
# CMAKE_BUILD_TYPE already supplies the optimization level; repeating it here
# triggers nvcc's "incompatible redefinition for option 'optimize'" warning.
#
# On Windows, CCCL (bundled with CUDA >= 12.x) requires cl.exe to use the
# standard-conforming preprocessor (/Zc:preprocessor).  Without it, the CCCL
# headers emit a fatal #error.  Pass the flag through nvcc's -Xcompiler relay
# when the host compiler is MSVC cl.exe.
if(WIN32 AND CMAKE_CUDA_HOST_COMPILER MATCHES "cl\.exe$")
    set(_elpa_cuda_flags_default "--extended-lambda;--expt-relaxed-constexpr;-Xcompiler=/Zc:preprocessor")
    message(STATUS "ELPA CUDA: adding -Xcompiler=/Zc:preprocessor for CCCL + MSVC host")
else()
    set(_elpa_cuda_flags_default "--extended-lambda;--expt-relaxed-constexpr")
endif()
set(ELPA_CUDA_FLAGS
    "${_elpa_cuda_flags_default}"
    CACHE STRING
    "Extra CUDA compiler flags"
)
unset(_elpa_cuda_flags_default)
# CMake handles -fPIC → -Xcompiler -fPIC natively, so nvcc_wrap is not needed.

# ---------------------------------------------------------------------------
# Find CUDA libraries
# ---------------------------------------------------------------------------
# CMake's FindCUDAToolkit provides imported targets for all CUDA libraries
find_package(CUDAToolkit REQUIRED)

# cublas (required)
if(NOT TARGET CUDA::cublas)
    message(FATAL_ERROR "ELPA: Could not find cublas")
endif()

# cublasLt (optional, needed for cublasLtHeuristicsCacheSetCapacity)
if(TARGET CUDA::cublasLt)
    set(HAVE_CUBLASLT 1)
    message(STATUS "ELPA CUDA: cublasLt found")
else()
    set(HAVE_CUBLASLT 0)
    message(STATUS "ELPA CUDA: cublasLt not found (optional)")
endif()

# cudart (required)
if(NOT TARGET CUDA::cudart)
    message(FATAL_ERROR "ELPA: Could not find cudart")
endif()

# ---------------------------------------------------------------------------
# cuSOLVER (optional, enabled by default when GPU streams are active)
# ---------------------------------------------------------------------------
option(ELPA_CUSOLVER "Use NVIDIA cuSOLVER library" ON)
if(ELPA_CUSOLVER)
    if(TARGET CUDA::cusolver)
        set(WITH_NVIDIA_CUSOLVER 1)
        message(STATUS "ELPA CUDA: cuSOLVER enabled")
    else()
        message(WARNING "ELPA: cuSOLVER requested but not found")
        set(WITH_NVIDIA_CUSOLVER 0)
    endif()
else()
    set(WITH_NVIDIA_CUSOLVER 0)
endif()

# ---------------------------------------------------------------------------
# Neutralise /Qoption,link,/LIBPATH: that leaks to C/CXX targets (Windows)
# ---------------------------------------------------------------------------
# When IntelLLVM Fortran (ifx) is enabled alongside CUDA, CMake's Ninja
# generator translates every link-directory source into the Intel Fortran
# linker-driver syntax  /Qoption,link,/LIBPATH:<dir>.  That syntax leaks
# transitively into downstream CUDA device-link and C/CXX link rules.
# Depending on the consuming linker driver, that causes failures such as
# nvcc "Don't know what to do with 'C:/Qoption,link,/LIBPATH:...'" or
# lld-link "could not open '/Qoption,link,/LIBPATH:...': invalid argument".
#
# Classic ifort emits plain -LIBPATH:<dir> instead and does not need this workaround.
#
# Two independent sources contribute to the bad LINK_PATH entries:
#   1. INTERFACE_LINK_DIRECTORIES on CUDA::* imported targets
#      (set by FindCUDAToolkit).
#   2. CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES / HOST variant
#      (set by CMake's CUDA compiler detection in CMakeCUDACompiler.cmake).
#      These also cause bare library names (cudadevrt.lib, cudart_static.lib)
#      to appear in the link line — the /Qoption LIBPATH was their search
#      path.
#
# Fix for (1): clear INTERFACE_LINK_DIRECTORIES on every CUDA imported
#   target — their .lib files are already linked by absolute path.
# Fix for (2): clear the implicit-link-directory variables and replace the
#   bare CUDA runtime library names with absolute paths so no search path
#   is needed.
if(WIN32 AND CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM")
    # --- (1) CUDA imported targets ---
    set(_cuda_targets_to_fix CUDA::cublas CUDA::cudart CUDA::cudart_static)
    foreach(_maybe CUDA::cublasLt CUDA::cusolver CUDA::cusparse CUDA::nvJitLink CUDA::cuda_driver CUDA::toolkit)
        if(TARGET ${_maybe})
            list(APPEND _cuda_targets_to_fix ${_maybe})
        endif()
    endforeach()
    foreach(_tgt IN LISTS _cuda_targets_to_fix)
        if(TARGET ${_tgt})
            set_target_properties(${_tgt} PROPERTIES INTERFACE_LINK_DIRECTORIES "")
        endif()
    endforeach()
    unset(_cuda_targets_to_fix)

    # --- (2) CUDA implicit link directories & runtime library names ---
    set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "")
    set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "")

    # CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES contains bare names such as
    # "cudadevrt;cudart" that the linker resolves via the search path we
    # just cleared.  CMake 4.x no longer honours the
    # CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_* substitution reliably, so
    # clear the bare-name list entirely.  The required libraries are still
    # linked via the CUDA::cudadevrt / CUDA::cudart imported targets whose
    # IMPORTED_LOCATION already carries the full absolute path.
    set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "")
    set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")

    # CMake's Compiler/NVIDIA.cmake (via __compiler_nvidia_cuda_flags macro) sets
    # CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_SHARED = "cudadevrt;cudart" as a
    # *normal* variable in the root scope.  Normal variables shadow cache
    # variables, so set(... CACHE INTERNAL FORCE) cannot override them.
    # We must use a plain set() in the same (root) scope to override.
    find_library(_elpa_cudadevrt      cudadevrt      PATHS "${CUDAToolkit_LIBRARY_DIR}" NO_DEFAULT_PATH)
    find_library(_elpa_cudart_static  cudart_static  PATHS "${CUDAToolkit_LIBRARY_DIR}" NO_DEFAULT_PATH)
    find_library(_elpa_cudart         cudart         PATHS "${CUDAToolkit_LIBRARY_DIR}" NO_DEFAULT_PATH)
    if(_elpa_cudadevrt AND _elpa_cudart_static)
        # Plain set() overrides the bare-name normal variable from NVIDIA.cmake
        set(CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_STATIC
            "${_elpa_cudadevrt};${_elpa_cudart_static}")
    endif()
    if(_elpa_cudadevrt AND _elpa_cudart)
        # Plain set() overrides the bare-name normal variable from NVIDIA.cmake
        set(CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_SHARED
            "${_elpa_cudadevrt};${_elpa_cudart}")
    endif()

    message(
        STATUS
        "ELPA CUDA: neutralised /Qoption,link,/LIBPATH: leaks — cleared "
        "INTERFACE_LINK_DIRECTORIES, CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES, "
        "CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES, and resolved CUDA runtime libs to absolute paths"
    )
endif()

# ---------------------------------------------------------------------------
# NCCL (optional)
# ---------------------------------------------------------------------------
option(ELPA_NCCL "Use NVIDIA NCCL library" OFF)
set(ELPA_NCCL_ROOT
    ""
    CACHE PATH
    "Optional root path hint for NCCL headers/libs"
)
if(ELPA_NCCL)
    set(_nccl_hints "")
    if(ELPA_NCCL_ROOT)
        list(APPEND _nccl_hints "${ELPA_NCCL_ROOT}")
    endif()
    if(DEFINED ENV{NCCL_PATH})
        list(APPEND _nccl_hints "$ENV{NCCL_PATH}")
    endif()
    find_library(
        NCCL_LIBRARY
        NAMES nccl
        HINTS ${_nccl_hints}
        PATH_SUFFIXES lib lib64
    )
    find_path(
        NCCL_INCLUDE_DIR
        NAMES nccl.h
        HINTS ${_nccl_hints}
        PATH_SUFFIXES include
    )
    if(NCCL_LIBRARY AND NCCL_INCLUDE_DIR)
        set(WITH_NVIDIA_NCCL 1)
        message(STATUS "ELPA CUDA: NCCL found at ${NCCL_LIBRARY}")
    else()
        message(
            FATAL_ERROR
            "ELPA: ELPA_NCCL=ON but NCCL not found. Set ELPA_NCCL_ROOT "
            "or NCCL_PATH to a valid NCCL installation prefix."
        )
    endif()
else()
    set(WITH_NVIDIA_NCCL 0)
endif()

# ---------------------------------------------------------------------------
# GPU streams
# Upstream default is ON. Was causing access violations on Windows due to
# private(my_stream) in an OMP parallel directive (should be firstprivate).
# ---------------------------------------------------------------------------
option(ELPA_GPU_STREAMS "Use CUDA streams" ON)
if(ELPA_GPU_STREAMS)
    set(WITH_GPU_STREAMS 1)
else()
    set(WITH_GPU_STREAMS 0)
endif()

# ---------------------------------------------------------------------------
# CUDA-aware MPI
# ---------------------------------------------------------------------------
option(ELPA_CUDA_AWARE_MPI "Use CUDA-aware MPI" OFF)
if(ELPA_CUDA_AWARE_MPI)
    set(WITH_CUDA_AWARE_MPI 1)
endif()

# ---------------------------------------------------------------------------
# GPU memory debugging
# ---------------------------------------------------------------------------
option(ELPA_CUDA_DEBUG "Enable CUDA memory debugging" OFF)
if(ELPA_CUDA_DEBUG)
    set(DEBUG_CUDA 1)
endif()

# ---------------------------------------------------------------------------
# CUB usage for real GPU kernels
# ---------------------------------------------------------------------------
option(ELPA_NVIDIA_CUB "Use CUB reductions in real NVIDIA GPU kernel" OFF)
if(ELPA_NVIDIA_CUB)
    set(NVIDIA_REAL_KERNEL_WITH_CUB 1)
endif()

# ---------------------------------------------------------------------------
# NVTX profiling support
# ---------------------------------------------------------------------------
option(ELPA_NVTX "Enable NVTX profiler annotations" OFF)
if(ELPA_NVTX)
    find_library(NVTOOLSEXT_LIBRARY NAMES nvToolsExt)
    if(NVTOOLSEXT_LIBRARY)
        set(WITH_NVTX 1)
    else()
        message(WARNING "ELPA: NVTX requested but nvToolsExt not found")
    endif()
endif()

# ---------------------------------------------------------------------------
# Enable GPU kernel options in ElpaKernels
# ---------------------------------------------------------------------------
# Standard NVIDIA GPU kernels are always enabled when CUDA is active.
# SM80 (A100+) kernels are auto-enabled when the targeted architectures
# support SM80 (compute capability >= 80).  This is more convenient than
# autotools (which requires a separate --enable-nvidia-sm80-gpu-kernels).
set(ELPA_ENABLE_NVIDIA_GPU_KERNELS ON CACHE BOOL "" FORCE)
if(WITH_NVIDIA_GPU_SM80_COMPUTE_CAPABILITY)
    set(ELPA_ENABLE_NVIDIA_SM80_GPU_KERNELS ON CACHE BOOL "" FORCE)
endif()

# GPU version defines (consumed by config.h and ElpaKernels)
set(WITH_NVIDIA_GPU_VERSION 1)
set(CURRENT_WITH_NVIDIA_GPU_VERSION 1)

message(
    STATUS
    "ELPA CUDA: architectures=${ELPA_CUDA_ARCHITECTURES} max_arch=${_elpa_cuda_max_arch}"
)
