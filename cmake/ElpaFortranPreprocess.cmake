# cmake/ElpaFortranPreprocess.cmake
#
# Replicate ELPA's "manual_cpp" two-step Fortran preprocessing.
#
# The ELPA source uses cpp macros (e.g. #define _rk _c_double) that must be
# expanded *inside* Fortran literal tokens such as 1.0_rk.  Standard Fortran
# preprocessing (-fpp) treats 1.0_rk as a single pp-number and does NOT expand
# _rk.  The autotools build uses "cpp -P -traditional" which performs
# traditional (pre-ANSI) C preprocessing where macros ARE expanded inside
# tokens.
#
# This module provides elpa_preprocess_fortran() which:
#   1. Runs a Python-based traditional preprocessor (fortran_pp.py) on each
#      .F90 source → produces a preprocessed .F90 in CMAKE_BINARY_DIR.
#   2. Returns the list of preprocessed files for use in add_library().
#
# The preprocessed files keep the .F90 extension so CMake still compiles them
# with the Fortran compiler, BUT we disable Fortran_PREPROCESS on the target
# (no -fpp) since preprocessing was already done externally.
#
# The Python preprocessor is cross-platform and eliminates the dependency on
# a GNU-compatible ``cpp`` binary (MSYS2 on Windows, gcc on Linux).
#
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(ELPA_FORTRAN_PP
    "${CMAKE_CURRENT_LIST_DIR}/python/fortran_pp.py"
    CACHE FILEPATH
    "Path to the Python-based Fortran preprocessor"
)

# elpa_preprocess_fortran(<output_list_var>
#     SOURCES <src1.F90> [<src2.F90> ...]
#     INCLUDE_DIRS <dir1> [<dir2> ...]
#     DEFINES <def1> [<def2> ...]
# )
#
# For each .F90 file in SOURCES, creates a custom command that runs
# fortran_pp.py and produces a preprocessed copy under
# ${CMAKE_BINARY_DIR}/_pp/.  Files that are NOT .F90 are passed through
# unchanged.
#
# Sets ${output_list_var} in the parent scope to the combined list.
function(elpa_preprocess_fortran output_list_var)
    cmake_parse_arguments(EPP "" "" "SOURCES;INCLUDE_DIRS;DEFINES" ${ARGN})

    set(_pp_dir "${CMAKE_BINARY_DIR}/_pp")

    # Build argument lists for the Python preprocessor
    set(_pp_inc_args "")
    foreach(_dir IN LISTS EPP_INCLUDE_DIRS)
        list(APPEND _pp_inc_args "-I${_dir}")
    endforeach()
    set(_pp_def_args "")
    foreach(_def IN LISTS EPP_DEFINES)
        list(APPEND _pp_def_args "-D${_def}")
    endforeach()

    set(_result "")
    foreach(_src IN LISTS EPP_SOURCES)
        # Only preprocess .F90 files (uppercase F = has preprocessor directives)
        get_filename_component(_ext "${_src}" EXT)
        # Skip files already in the build directory (already generated/preprocessed)
        string(FIND "${_src}" "${CMAKE_BINARY_DIR}" _in_build_dir)
        if(_ext STREQUAL ".F90" AND _in_build_dir EQUAL -1)
            # Compute a unique output path relative to the source tree
            file(RELATIVE_PATH _relpath "${CMAKE_SOURCE_DIR}" "${_src}")
            if(_relpath MATCHES "^\\.\\./")
                # File outside source tree (e.g. generated in build dir) — use
                # path relative to build dir instead
                file(RELATIVE_PATH _relpath "${CMAKE_BINARY_DIR}" "${_src}")
                set(_pp_out "${_pp_dir}/build/${_relpath}")
            else()
                set(_pp_out "${_pp_dir}/src/${_relpath}")
            endif()

            get_filename_component(_pp_out_dir "${_pp_out}" DIRECTORY)

            add_custom_command(
                OUTPUT "${_pp_out}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${_pp_out_dir}"
                COMMAND
                    ${Python3_EXECUTABLE} "${ELPA_FORTRAN_PP}" ${_pp_def_args}
                    ${_pp_inc_args} --depfile "${_pp_out}.d" -o "${_pp_out}"
                    "${_src}"
                DEPENDS "${_src}" "${ELPA_FORTRAN_PP}"
                DEPFILE "${_pp_out}.d"
                COMMENT "fortran_pp ${_relpath}"
                VERBATIM
            )
            list(APPEND _result "${_pp_out}")
        else()
            # .c, .cu, .s, etc. — pass through unchanged
            list(APPEND _result "${_src}")
        endif()
    endforeach()

    set(${output_list_var} "${_result}" PARENT_SCOPE)
endfunction()
