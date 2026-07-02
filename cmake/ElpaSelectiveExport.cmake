# ElpaSelectiveExport.cmake — Controlled symbol visibility (Windows only)
#
# On Windows, DLL symbol exports must be explicitly listed.  This module
# generates a filtered .def file so that only the public ELPA ABI is
# exported from the DLL.
#
# On Linux / macOS, all symbols remain exported (matching the upstream
# autotools build).  A GNU ld version script is intentionally NOT used
# because ifort-compiled Fortran OOP type-bound procedure dispatch (vtable
# function pointers) relies on global symbol binding; hiding internal
# module procedures via "local: *;" causes silent incorrect results at
# runtime — the dynamic linker emits a version-lookup warning and
# polymorphic calls resolve to wrong addresses.
#
# Exported symbol categories (Windows):
#   * ELPA_mp_*           — Fortran ELPA module
#   * ELPA_API_mp_*       — Fortran API module
#   * FTIMINGS_mp_TIMER_* — timing infrastructure
#   * CUDA_FUNCTIONS_mp_* — GPU memory management
#   * elpa_*              — C public API (elpa.h / elpa_generic.h)
#
# Usage:
#   include(ElpaSelectiveExport)
#   elpa_selective_export(<target>)
#
# Prerequisites:
#   - Python3_EXECUTABLE must be defined (find_package(Python3 REQUIRED))
#   - target must be a SHARED library

set(_ELPA_GENERATE_EXPORTS_SCRIPT
    "${CMAKE_CURRENT_LIST_DIR}/python/generate_exports_def.py"
    CACHE INTERNAL
    "Path to the ELPA DLL export filter script"
)

function(elpa_selective_export target)
    set(_obj_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${target}.dir")

    if(WIN32)
        set(_def_file "${CMAKE_BINARY_DIR}/${target}.def")

        # Locate dumpbin.exe — it lives alongside link.exe in the MSVC toolchain.
        # CMAKE_LINKER points to link.exe; dumpbin.exe is in the same directory.
        cmake_path(GET CMAKE_LINKER PARENT_PATH _linker_dir)
        find_program(
            _dumpbin_exe
            NAMES dumpbin
            HINTS "${_linker_dir}"
            NO_DEFAULT_PATH
        )
        if(NOT _dumpbin_exe)
            # Fallback: search PATH (succeeds when VS developer shell is active)
            find_program(_dumpbin_exe NAMES dumpbin)
        endif()
        if(NOT _dumpbin_exe)
            message(
                FATAL_ERROR
                "ELPA: dumpbin.exe not found — required for export list generation. "
                "Ensure MSVC Build Tools are installed."
            )
        endif()

        # PRE_LINK runs after all objects are compiled but before the link step.
        # The Python script scans .obj files, extracts defined external
        # symbols, filters them, and writes the .def file.
        add_custom_command(
            TARGET ${target}
            PRE_LINK
            COMMAND
                "${Python3_EXECUTABLE}" "${_ELPA_GENERATE_EXPORTS_SCRIPT}"
                "${_obj_dir}" "${_def_file}" "${_dumpbin_exe}"
            COMMENT "Generating filtered DLL export definitions for ${target}"
            VERBATIM
        )
        unset(_linker_dir)
        unset(_dumpbin_exe)

        # Tell the linker to use our filtered .def instead of exporting everything.
        #
        # The flag must be routed differently depending on how CMake invokes
        # the linker for each Intel Fortran compiler:
        #
        #   IntelLLVM (ifx)  — cmake -E vs_link_exe calls ifx.exe as the
        #     link driver; ifx then invokes link.exe internally.  Flags in
        #     LINK_FLAGS go to ifx, not link.exe, so /DEF: would be silently
        #     dropped by ifx.  Use -Qoption,link,/DEF: to tell ifx to forward
        #     the option to its underlying link.exe invocation.
        #
        #   Intel (ifort)  — cmake -E vs_link_dll calls the native linker
        #     (lld-link.exe or link.exe) DIRECTLY; LINK_FLAGS are passed
        #     straight to the linker, bypassing ifort entirely.
        #     -Qoption,link, is an Intel-compiler-only flag that lld-link and
        #     link.exe do not understand.  Pass /DEF: directly instead.
        if(CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM")
            target_link_options(
                ${target}
                PRIVATE "-Qoption,link,/DEF:${_def_file}"
            )
        else()
            target_link_options(${target} PRIVATE "/DEF:${_def_file}")
        endif()

        # CMake's vs_link_dll two-pass manifest-embedding process (link →
        # mt.exe → re-link) does not reliably regenerate the MSVC import
        # library (.lib) in the second pass.  The DLL is produced but the
        # .lib is missing, causing downstream C test executables to fail
        # at link time.  Regenerate the import library explicitly after
        # every successful build using lib.exe and the filtered DEF file.
        execute_process(
            COMMAND
                "C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
                -latest -property installationPath
            OUTPUT_VARIABLE _vswhere_install
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_vswhere_install)
            file(GLOB _vsw_msvc_vers "${_vswhere_install}/VC/Tools/MSVC/*")
            list(SORT _vsw_msvc_vers ORDER DESCENDING)
            list(GET _vsw_msvc_vers 0 _vsw_msvc_latest)
            set(_lib_exe_path "${_vsw_msvc_latest}/bin/Hostx64/x64/lib.exe")
        endif()
        if(NOT EXISTS "${_lib_exe_path}")
            # Fall back: look on PATH
            find_program(_lib_exe_path lib.exe)
        endif()
        if(_lib_exe_path)
            add_custom_command(
                TARGET ${target}
                POST_BUILD
                COMMAND
                    "${_lib_exe_path}" /nologo "/def:${_def_file}"
                    "/out:$<TARGET_LINKER_FILE:${target}>"
                    "/name:$<TARGET_FILE_NAME:${target}>" /machine:x64
                COMMENT
                    "Regenerating MSVC import library for ${target} from filtered DEF"
                VERBATIM
            )
        else()
            message(
                WARNING
                "ELPA: lib.exe not found — import library for ${target} may be missing."
                "Set MSVC on PATH or ensure Visual Studio Build Tools is installed."
            )
        endif()
        unset(_vswhere_install)
        unset(_vsw_msvc_vers)
        unset(_vsw_msvc_latest)
        unset(_lib_exe_path)
    else()
        # Linux / macOS: no version script — all symbols remain exported.
        # See header comment for rationale.
    endif()
endfunction()
