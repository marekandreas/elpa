set(ELPA_MKL_ROOT
    ""
    CACHE PATH
    "MKL installation root hint (passed as MKL_ROOT to MKLConfig.cmake)"
)
set(ELPA_INTEL_COMPILER_RUNTIME_LDFLAGS
    ""
    CACHE STRING
    "Optional extra linker flags for Intel compiler runtime"
)

option(ELPA_USE_MKL "Use Intel MKL for BLAS/LAPACK/ScaLAPACK" ON)

# MKL is only available on x86/x86_64.  Auto-disable on other architectures.
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
    if(ELPA_USE_MKL)
        message(STATUS "ELPA: Disabling MKL (not available on ${CMAKE_SYSTEM_PROCESSOR})")
        set(ELPA_USE_MKL OFF CACHE BOOL "Use Intel MKL for BLAS/LAPACK/ScaLAPACK" FORCE)
    endif()
endif()

if(ELPA_USE_MKL)
    if(ELPA_MKL_ROOT)
        set(MKL_ROOT "${ELPA_MKL_ROOT}" CACHE PATH "MKL root directory" FORCE)
    endif()
    set(MKL_ARCH "intel64" CACHE STRING "MKL architecture")
    set(MKL_LINK "dynamic" CACHE STRING "MKL link mode (static|dynamic)")
    set(MKL_THREADING "intel_thread" CACHE STRING "MKL threading model")
    set(MKL_INTERFACE "lp64" CACHE STRING "MKL integer interface (lp64|ilp64)")
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
    message(STATUS "ELPA: MKL detected via MKLConfig.cmake (root=${MKL_ROOT})")

    if(NOT TARGET ELPA::scalapack)
        add_library(ELPA::scalapack INTERFACE IMPORTED)
        set_property(
            TARGET ELPA::scalapack
            PROPERTY INTERFACE_LINK_LIBRARIES MKL::MKL
        )
    endif()
else()
    set(WITH_MKL 0)
    set(HAVE_MKL 0)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    if(ELPA_MPI)
        find_library(
            SCALAPACK_LIBRARY
            NAMES scalapack mpiscalapack scalapack-openmpi
        )
        if(NOT SCALAPACK_LIBRARY)
            message(FATAL_ERROR "Could not find SCALAPACK.")
        endif()
        set(WITH_BLACS 1)
    endif()
    message(STATUS "ELPA: Using generic BLAS/LAPACK/SCALAPACK")

    if(NOT TARGET ELPA::scalapack)
        add_library(ELPA::scalapack INTERFACE IMPORTED)
        if(BLAS_LIBRARIES)
            set_property(
                TARGET ELPA::scalapack
                APPEND
                PROPERTY INTERFACE_LINK_LIBRARIES ${BLAS_LIBRARIES}
            )
        endif()
        if(LAPACK_LIBRARIES)
            set_property(
                TARGET ELPA::scalapack
                APPEND
                PROPERTY INTERFACE_LINK_LIBRARIES ${LAPACK_LIBRARIES}
            )
        endif()
        if(SCALAPACK_LIBRARY)
            set_property(
                TARGET ELPA::scalapack
                APPEND
                PROPERTY INTERFACE_LINK_LIBRARIES ${SCALAPACK_LIBRARY}
            )
        endif()
    endif()
endif()

if(ELPA_INTEL_COMPILER_RUNTIME_LDFLAGS)
    separate_arguments(
        _iomp_link_opts
        NATIVE_COMMAND
        "${ELPA_INTEL_COMPILER_RUNTIME_LDFLAGS}"
    )
    add_link_options(${_iomp_link_opts})
endif()
