set(
    ELPA_BLAS_VENDOR
    "AUTO"
    CACHE STRING
    "BLAS/LAPACK vendor (AUTO, MKL, BLAS, or a CMake BLA_VENDOR value)"
)
set_property(
    CACHE ELPA_BLAS_VENDOR
    PROPERTY STRINGS AUTO MKL BLAS OpenBLAS FlexiBLAS Generic
)

# Deprecated command-line variables are recognized below but are no longer
# exposed as ELPA build options.
set(ELPA_INTEL_COMPILER_RUNTIME_LDFLAGS
    ""
    CACHE STRING
    "Optional extra linker flags for Intel compiler runtime"
)

if(DEFINED ELPA_USE_MKL)
    message(
        DEPRECATION
        "ELPA_USE_MKL is deprecated; use ELPA_BLAS_VENDOR=MKL or a BLAS vendor name"
    )
    if(ELPA_USE_MKL)
        set(ELPA_BLAS_VENDOR "MKL" CACHE STRING "" FORCE)
    else()
        set(ELPA_BLAS_VENDOR "BLAS" CACHE STRING "" FORCE)
    endif()
endif()

if(DEFINED ELPA_MKL_ROOT AND ELPA_MKL_ROOT)
    message(DEPRECATION "ELPA_MKL_ROOT is deprecated; use MKL_ROOT instead")
    set(MKL_ROOT "${ELPA_MKL_ROOT}" CACHE PATH "MKL root directory" FORCE)
endif()

string(TOUPPER "${ELPA_BLAS_VENDOR}" _elpa_blas_vendor_upper)
set(_elpa_math_backend "GENERIC")

if(_elpa_blas_vendor_upper STREQUAL "AUTO")
    # Respect an explicitly selected standard CMake BLAS vendor first.
    if(BLA_VENDOR AND NOT BLA_VENDOR STREQUAL "All")
        if(BLA_VENDOR MATCHES "^Intel" OR BLA_VENDOR STREQUAL "MKL")
            set(_elpa_math_backend "MKL")
        endif()
    # As in ABACUS, an MKL environment/root hint is treated as an explicit
    # request. Otherwise use the generic CMake BLAS/LAPACK discovery path.
    elseif(MKL_ROOT OR DEFINED ENV{MKLROOT})
        set(_elpa_math_backend "MKL")
    endif()
elseif(_elpa_blas_vendor_upper STREQUAL "MKL")
    set(_elpa_math_backend "MKL")
elseif(_elpa_blas_vendor_upper STREQUAL "BLAS")
    # Use CMake's regular BLAS/LAPACK discovery without selecting MKL through
    # the ELPA-specific MKLConfig.cmake path.
    set(_elpa_math_backend "GENERIC")
else()
    # CMake's FindBLAS supports OpenBLAS, FlexiBLAS, Apple Accelerate, ArmPL,
    # BLIS/FLAME, and other implementations through BLA_VENDOR.
    set(BLA_VENDOR "${ELPA_BLAS_VENDOR}")
endif()

if(
    _elpa_math_backend STREQUAL "MKL"
    AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86"
)
    if(_elpa_blas_vendor_upper STREQUAL "AUTO")
        message(
            STATUS
            "ELPA: Ignoring MKL hint on unsupported architecture ${CMAKE_SYSTEM_PROCESSOR}"
        )
        set(_elpa_math_backend "GENERIC")
    else()
        message(
            FATAL_ERROR
            "ELPA_BLAS_VENDOR=MKL is not supported on ${CMAKE_SYSTEM_PROCESSOR}"
        )
    endif()
endif()

if(_elpa_math_backend STREQUAL "MKL")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(MKL_ARCH "intel64" CACHE STRING "MKL architecture")
    else()
        set(MKL_ARCH "ia32" CACHE STRING "MKL architecture")
    endif()
    set(MKL_LINK "dynamic" CACHE STRING "MKL link mode (static|dynamic)")
    set(MKL_THREADING "intel_thread" CACHE STRING "MKL threading model")
    if(ELPA_64BIT_INTEGER_MATH)
        set(MKL_INTERFACE "ilp64" CACHE STRING "MKL integer interface" FORCE)
    else()
        set(MKL_INTERFACE "lp64" CACHE STRING "MKL integer interface" FORCE)
    endif()

    # On Windows with Intel oneAPI the only supported MPI is Intel MPI, so
    # default MKL_MPI accordingly (selects the right BLACS variant in MKLConfig).
    if(WIN32 AND NOT MKL_MPI)
        set(MKL_MPI
            "intelmpi"
            CACHE STRING
            "MKL MPI variant (intelmpi|openmpi|mpich|msmpi)"
        )
    endif()
    if(ELPA_MPI)
        set(ENABLE_BLACS ON CACHE BOOL "Enable MKL cluster BLAS" FORCE)
        set(ENABLE_SCALAPACK ON CACHE BOOL "Enable MKL ScaLAPACK" FORCE)
    endif()

    find_package(MKL CONFIG REQUIRED)

    set(HAVE_MKL 1)
    set(WITH_MKL 1)
    message(STATUS "ELPA: Using Intel MKL (root=${MKL_ROOT})")

    set(_elpa_math_libraries MKL::MKL)
else()
    set(WITH_MKL 0)
    set(HAVE_MKL 0)

    if(ELPA_64BIT_INTEGER_MATH)
        set(BLA_SIZEOF_INTEGER 8)
    else()
        set(BLA_SIZEOF_INTEGER 4)
    endif()

    if(NOT TARGET BLAS::BLAS)
        if(BLAS_LIBRARIES)
            add_library(BLAS::BLAS INTERFACE IMPORTED)
            set_property(
                TARGET BLAS::BLAS
                PROPERTY INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}"
            )
        else()
            find_package(BLAS REQUIRED)
        endif()
    endif()

    if(NOT TARGET LAPACK::LAPACK)
        if(LAPACK_LIBRARIES)
            add_library(LAPACK::LAPACK INTERFACE IMPORTED)
            set_property(
                TARGET LAPACK::LAPACK
                PROPERTY INTERFACE_LINK_LIBRARIES "${LAPACK_LIBRARIES}"
            )
        else()
            find_package(LAPACK REQUIRED)
        endif()
    endif()

    set(_elpa_math_libraries BLAS::BLAS LAPACK::LAPACK)

    if(ELPA_MPI)
        if(TARGET ScaLAPACK::ScaLAPACK)
            list(APPEND _elpa_math_libraries ScaLAPACK::ScaLAPACK)
        else()
            find_library(
                SCALAPACK_LIBRARY
                NAMES scalapack mpiscalapack scalapack-openmpi scalapack-mpich
            )
            if(NOT SCALAPACK_LIBRARY)
                message(FATAL_ERROR "Could not find ScaLAPACK")
            endif()
            list(APPEND _elpa_math_libraries ${SCALAPACK_LIBRARY})
        endif()
        set(WITH_BLACS 1)
    endif()

    if(_elpa_blas_vendor_upper STREQUAL "BLAS")
        message(STATUS "ELPA: Using standard BLAS/LAPACK discovery")
    elseif(NOT _elpa_blas_vendor_upper STREQUAL "AUTO")
        message(STATUS "ELPA: Using BLAS/LAPACK vendor ${ELPA_BLAS_VENDOR}")
    elseif(BLA_VENDOR AND NOT BLA_VENDOR STREQUAL "All")
        message(STATUS "ELPA: Using BLAS/LAPACK vendor ${BLA_VENDOR}")
    else()
        message(STATUS "ELPA: Using automatic BLAS/LAPACK discovery")
    endif()
endif()

if(NOT TARGET ELPA::scalapack)
    add_library(ELPA::scalapack INTERFACE IMPORTED)
    set_property(
        TARGET ELPA::scalapack
        PROPERTY INTERFACE_LINK_LIBRARIES "${_elpa_math_libraries}"
    )
endif()

unset(_elpa_math_backend)
unset(_elpa_math_libraries)
unset(_elpa_blas_vendor_upper)

if(ELPA_INTEL_COMPILER_RUNTIME_LDFLAGS)
    separate_arguments(
        _iomp_link_opts
        NATIVE_COMMAND
        "${ELPA_INTEL_COMPILER_RUNTIME_LDFLAGS}"
    )
    add_link_options(${_iomp_link_opts})
endif()
