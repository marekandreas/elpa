set(ELPA_MPI_ROOT
    ""
    CACHE PATH
    "MPI installation root hint (sets MPI_HOME and I_MPI_ROOT for FindMPI)"
)
set(ELPA_MPI_C_COMPILER
    ""
    CACHE FILEPATH
    "Optional override for MPI C wrapper compiler"
)
set(ELPA_MPI_CXX_COMPILER
    ""
    CACHE FILEPATH
    "Optional override for MPI C++ wrapper compiler"
)
set(ELPA_MPI_Fortran_COMPILER
    ""
    CACHE FILEPATH
    "Optional override for MPI Fortran wrapper compiler"
)

if(ELPA_MPI)
    if(ELPA_MPI_ROOT)
        set(MPI_HOME "${ELPA_MPI_ROOT}" CACHE PATH "MPI root hint" FORCE)
        set(I_MPI_ROOT
            "${ELPA_MPI_ROOT}"
            CACHE PATH
            "Intel MPI root hint"
            FORCE
        )
    endif()
    if(ELPA_MPI_C_COMPILER)
        set(MPI_C_COMPILER
            "${ELPA_MPI_C_COMPILER}"
            CACHE FILEPATH
            "MPI C compiler"
            FORCE
        )
    endif()
    if(ELPA_MPI_CXX_COMPILER)
        set(MPI_CXX_COMPILER
            "${ELPA_MPI_CXX_COMPILER}"
            CACHE FILEPATH
            "MPI CXX compiler"
            FORCE
        )
    endif()
    if(ELPA_MPI_Fortran_COMPILER)
        set(MPI_Fortran_COMPILER
            "${ELPA_MPI_Fortran_COMPILER}"
            CACHE FILEPATH
            "MPI Fortran compiler"
            FORCE
        )
    endif()

    if(WIN32 AND CMAKE_C_COMPILER_ID STREQUAL "Clang")
        # Intel MPI on Windows names its libraries impi/impicxx rather than
        # mpi/mpicxx.  CMake's FindMPI has no built-in knowledge of this naming.
        # Synthesise the required hints automatically from ELPA_MPI_ROOT (or
        # MPI_HOME / ENV{I_MPI_ROOT} as fallbacks) when an Intel MPI layout is
        # detected, so callers only need -DELPA_MPI_ROOT=<root>.
        if(NOT MPI_C_LIB_NAMES OR NOT MPI_CXX_HEADER_DIR)
            set(_elpa_impi_root "${ELPA_MPI_ROOT}")
            if(NOT _elpa_impi_root AND MPI_HOME)
                set(_elpa_impi_root "${MPI_HOME}")
            endif()
            if(NOT _elpa_impi_root AND DEFINED ENV{I_MPI_ROOT})
                set(_elpa_impi_root "$ENV{I_MPI_ROOT}")
            endif()
            if(_elpa_impi_root AND EXISTS "${_elpa_impi_root}/lib/impi.lib")
                message(
                    STATUS
                    "ELPA: Intel MPI detected at ${_elpa_impi_root}; "
                    "synthesising FindMPI hints (impi/impicxx)"
                )
                set(MPI_C_HEADER_DIR
                    "${_elpa_impi_root}/include"
                    CACHE PATH
                    "MPI C header directory"
                    FORCE
                )
                set(MPI_CXX_HEADER_DIR
                    "${_elpa_impi_root}/include"
                    CACHE PATH
                    "MPI C++ header directory"
                    FORCE
                )
                set(MPI_C_LIB_NAMES
                    "impi"
                    CACHE STRING
                    "MPI C library names"
                    FORCE
                )
                set(MPI_CXX_LIB_NAMES
                    "impicxx;impi"
                    CACHE STRING
                    "MPI C++ library names"
                    FORCE
                )
                set(MPI_impi_LIBRARY
                    "${_elpa_impi_root}/lib/impi.lib"
                    CACHE FILEPATH
                    "Intel MPI C import library"
                    FORCE
                )
                set(MPI_impicxx_LIBRARY
                    "${_elpa_impi_root}/lib/impicxx.lib"
                    CACHE FILEPATH
                    "Intel MPI C++ import library"
                    FORCE
                )
            endif()
            unset(_elpa_impi_root)
        endif()

        find_package(MPI REQUIRED COMPONENTS C CXX)
        set(MPI_Fortran_FOUND TRUE)
        set(MPI_Fortran_WORKS TRUE)
        set(MPI_Fortran_INCLUDE_DIRS "${MPI_C_INCLUDE_DIRS}")
        set(MPI_Fortran_LIBRARIES "${MPI_C_LIBRARIES}")
        set(MPI_Fortran_COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}")
        set(MPI_Fortran_LINK_FLAGS "${MPI_C_LINK_FLAGS}")
        if(NOT TARGET MPI::MPI_Fortran)
            add_library(MPI::MPI_Fortran INTERFACE IMPORTED)
            if(MPI_Fortran_INCLUDE_DIRS)
                set_property(
                    TARGET MPI::MPI_Fortran
                    PROPERTY
                        INTERFACE_INCLUDE_DIRECTORIES
                            "${MPI_Fortran_INCLUDE_DIRS}"
                )
            endif()
            if(MPI_Fortran_LIBRARIES)
                set_property(
                    TARGET MPI::MPI_Fortran
                    PROPERTY INTERFACE_LINK_LIBRARIES "${MPI_Fortran_LIBRARIES}"
                )
            endif()
            if(MPI_Fortran_COMPILE_FLAGS)
                string(STRIP "${MPI_Fortran_COMPILE_FLAGS}" _mpi_fcflags)
                separate_arguments(_mpi_fcopts NATIVE_COMMAND "${_mpi_fcflags}")
                set_property(
                    TARGET MPI::MPI_Fortran
                    PROPERTY INTERFACE_COMPILE_OPTIONS "${_mpi_fcopts}"
                )
            endif()
            if(MPI_Fortran_LINK_FLAGS)
                string(STRIP "${MPI_Fortran_LINK_FLAGS}" _mpi_fldflags)
                separate_arguments(
                    _mpi_fldopts
                    NATIVE_COMMAND
                    "${_mpi_fldflags}"
                )
                set_property(
                    TARGET MPI::MPI_Fortran
                    PROPERTY INTERFACE_LINK_OPTIONS "${_mpi_fldopts}"
                )
            endif()
        endif()
        message(
            STATUS
            "ELPA: reusing MPI C discovery for Fortran on Windows+Clang"
        )
    else()
        find_package(MPI REQUIRED COMPONENTS C CXX Fortran)
    endif()
    set(WITH_MPI 1)

    if(MPI_Fortran_FOUND AND TARGET MPI::MPI_Fortran)
        if(
            NOT MPI_Fortran_MODULE_DIR
            OR MPI_Fortran_MODULE_DIR STREQUAL "MPI_Fortran_MODULE_DIR-NOTFOUND"
        )
            foreach(
                _dir
                IN
                LISTS
                    MPI_Fortran_COMPILER_INCLUDE_DIRS
                    MPI_Fortran_F77_HEADER_DIR
            )
                foreach(_subdir "" "mpi")
                    if(_subdir)
                        set(_try_dir "${_dir}/${_subdir}")
                    else()
                        set(_try_dir "${_dir}")
                    endif()
                    if(EXISTS "${_try_dir}/mpi.mod")
                        set(MPI_Fortran_MODULE_DIR
                            "${_try_dir}"
                            CACHE PATH
                            "MPI Fortran module directory"
                            FORCE
                        )
                        break()
                    endif()
                endforeach()
                if(
                    MPI_Fortran_MODULE_DIR
                    AND NOT MPI_Fortran_MODULE_DIR
                        STREQUAL
                        "MPI_Fortran_MODULE_DIR-NOTFOUND"
                )
                    break()
                endif()
            endforeach()
        endif()
        if(
            MPI_Fortran_MODULE_DIR
            AND NOT MPI_Fortran_MODULE_DIR STREQUAL "MPI_Fortran_MODULE_DIR-NOTFOUND"
        )
            message(
                STATUS
                "ELPA: MPI Fortran module directory: ${MPI_Fortran_MODULE_DIR}"
            )
            get_target_property(
                _mpi_f_incdirs
                MPI::MPI_Fortran
                INTERFACE_INCLUDE_DIRECTORIES
            )
            if(NOT "${MPI_Fortran_MODULE_DIR}" IN_LIST _mpi_f_incdirs)
                set_property(
                    TARGET MPI::MPI_Fortran
                    APPEND
                    PROPERTY
                        INTERFACE_INCLUDE_DIRECTORIES
                            "${MPI_Fortran_MODULE_DIR}"
                )
            endif()
        endif()
    endif()

    include(CheckCSourceRuns)
    set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_C)

    # Ensure the C MPI header directory (where mpi.h lives) is present in
    # MPI::MPI_C's interface.  With Intel MPI + a non-gfortran Fortran wrapper,
    # FindMPI may inject only the Fortran module sub-directory (include/mpi)
    # but not the parent include/ that contains mpi.h, causing C compilation
    # failures when MPI::MPI_Fortran is linked PUBLIC.
    if(MPI_C_HEADER_DIR AND TARGET MPI::MPI_C)
        get_target_property(
            _mpi_c_incdirs
            MPI::MPI_C
            INTERFACE_INCLUDE_DIRECTORIES
        )
        if(NOT "${MPI_C_HEADER_DIR}" IN_LIST _mpi_c_incdirs)
            set_property(
                TARGET MPI::MPI_C
                APPEND
                PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_C_HEADER_DIR}"
            )
        endif()
        unset(_mpi_c_incdirs)
    endif()
    if(MPI_CXX_HEADER_DIR AND TARGET MPI::MPI_CXX)
        get_target_property(
            _mpi_cxx_incdirs
            MPI::MPI_CXX
            INTERFACE_INCLUDE_DIRECTORIES
        )
        if(NOT "${MPI_CXX_HEADER_DIR}" IN_LIST _mpi_cxx_incdirs)
            set_property(
                TARGET MPI::MPI_CXX
                APPEND
                PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_HEADER_DIR}"
            )
        endif()
        unset(_mpi_cxx_incdirs)
    endif()

    check_c_source_runs(
        "\n#include <mpi.h>\nint main(int argc, char **argv) {\n    int provided;\n    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);\n    MPI_Finalize();\n    return (provided >= MPI_THREAD_SERIALIZED) ? 0 : 1;\n}\n"
        HAVE_SUFFICIENT_MPI_THREADING_SUPPORT
    )
    unset(CMAKE_REQUIRED_LIBRARIES)
else()
    set(WITH_MPI 0)
endif()
