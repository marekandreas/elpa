find_package(Python3 REQUIRED COMPONENTS Interpreter)

function(_elpa_extract_interfaces output_path)
    cmake_parse_arguments(EI "" "" "MARKERS;SOURCES" ${ARGN})

    get_filename_component(_output_dir "${output_path}" DIRECTORY)
    set(_extract_script
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/python/extract_interfaces.py"
    )

    add_custom_command(
        OUTPUT "${output_path}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_output_dir}"
        COMMAND
            ${Python3_EXECUTABLE} "${_extract_script}" ${EI_MARKERS} -o
            "${output_path}" ${EI_SOURCES}
        DEPENDS "${_extract_script}" ${EI_SOURCES}
        COMMENT "Extracting interfaces -> ${output_path}"
        VERBATIM
    )
endfunction()

function(elpa_generate_build_artifacts)
    set(_elpa_cmake_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    set(_elpa_python_dir "${_elpa_cmake_dir}/python")
    set(_elpa_fortran_constants_script
        "${_elpa_python_dir}/process_fortran_constants.py"
    )

    file(
        MAKE_DIRECTORY
            "${PROJECT_BINARY_DIR}/elpa"
            "${PROJECT_BINARY_DIR}/src"
            "${PROJECT_BINARY_DIR}/test/shared"
    )

    configure_file(
        "${_elpa_cmake_dir}/config.h.cmake.in"
        "${PROJECT_BINARY_DIR}/config.h"
        @ONLY
    )

    file(
        STRINGS "${PROJECT_BINARY_DIR}/config.h"
        _config_defines
        REGEX "^#define"
    )
    list(JOIN _config_defines "\n" _config_defines_text)
    file(WRITE "${PROJECT_BINARY_DIR}/config-f90.h" "${_config_defines_text}\n")

    configure_file(
        "${PROJECT_SOURCE_DIR}/elpa/elpa_constants.h.in"
        "${PROJECT_BINARY_DIR}/elpa/elpa_constants.h"
        @ONLY
    )
    configure_file(
        "${PROJECT_SOURCE_DIR}/elpa/elpa_version.h.in"
        "${PROJECT_BINARY_DIR}/elpa/elpa_version.h"
        @ONLY
    )
    configure_file(
        "${PROJECT_SOURCE_DIR}/elpa/elpa_configured_options.h.in"
        "${PROJECT_BINARY_DIR}/elpa/elpa_configured_options.h"
        @ONLY
    )

    file(
        WRITE "${PROJECT_BINARY_DIR}/elpa/elpa_build_config.h"
        "// The stored build config\n"
    )

    if(ELPA_OPTIONAL_C_ERROR_ARGUMENT)
        file(
            WRITE "${PROJECT_BINARY_DIR}/elpa/elpa_generated_c_api.h"
            "#define OPTIONAL_C_ERROR_ARGUMENT 1\n"
        )
    else()
        file(
            WRITE "${PROJECT_BINARY_DIR}/elpa/elpa_generated_c_api.h"
            "#undef OPTIONAL_C_ERROR_ARGUMENT\n"
        )
    endif()

    # fortran_constants.h relies on a real C preprocessor.
    if(WIN32)
        add_custom_command(
            OUTPUT "${PROJECT_BINARY_DIR}/src/fortran_constants.F90"
            COMMAND
                ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/src"
            COMMAND
                ${CMAKE_C_COMPILER} /EP /P "/I${PROJECT_BINARY_DIR}"
                "/I${PROJECT_SOURCE_DIR}"
                "/Fi${PROJECT_BINARY_DIR}/src/fortran_constants.F90_"
                "${PROJECT_SOURCE_DIR}/src/fortran_constants.h"
            COMMAND
                ${Python3_EXECUTABLE} "${_elpa_fortran_constants_script}"
                "${PROJECT_BINARY_DIR}/src/fortran_constants.F90_"
                "${PROJECT_BINARY_DIR}/src/fortran_constants.F90"
            COMMAND
                ${CMAKE_COMMAND} -E remove
                "${PROJECT_BINARY_DIR}/src/fortran_constants.F90_"
            DEPENDS
                "${PROJECT_SOURCE_DIR}/src/fortran_constants.h"
                "${PROJECT_BINARY_DIR}/config.h"
                "${_elpa_fortran_constants_script}"
            COMMENT "Generating Fortran constants"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT "${PROJECT_BINARY_DIR}/src/fortran_constants.F90"
            COMMAND
                ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/src"
            COMMAND
                ${CMAKE_C_COMPILER} -E -P -I${PROJECT_BINARY_DIR}
                -I${PROJECT_SOURCE_DIR} -o
                "${PROJECT_BINARY_DIR}/src/fortran_constants.F90_"
                "${PROJECT_SOURCE_DIR}/src/fortran_constants.h"
            COMMAND
                ${Python3_EXECUTABLE} "${_elpa_fortran_constants_script}"
                "${PROJECT_BINARY_DIR}/src/fortran_constants.F90_"
                "${PROJECT_BINARY_DIR}/src/fortran_constants.F90"
            COMMAND
                ${CMAKE_COMMAND} -E remove
                "${PROJECT_BINARY_DIR}/src/fortran_constants.F90_"
            DEPENDS
                "${PROJECT_SOURCE_DIR}/src/fortran_constants.h"
                "${PROJECT_BINARY_DIR}/config.h"
                "${_elpa_fortran_constants_script}"
            COMMENT "Generating Fortran constants"
            VERBATIM
        )
    endif()
    add_custom_target(
        elpa_fortran_constants
        DEPENDS "${PROJECT_BINARY_DIR}/src/fortran_constants.F90"
    )

    set(_c_interface_sources
        "${PROJECT_SOURCE_DIR}/src/elpa_impl.F90"
        "${PROJECT_SOURCE_DIR}/src/elpa_impl_math_template.F90"
        "${PROJECT_SOURCE_DIR}/src/elpa_impl_math_solvers_template.F90"
        "${PROJECT_SOURCE_DIR}/src/elpa_impl_math_generalized_template.F90"
        "${PROJECT_SOURCE_DIR}/src/elpa_api.F90"
    )
    _elpa_extract_interfaces(
        "${PROJECT_BINARY_DIR}/elpa/elpa_generated.h"
        MARKERS -m "!c>" -m "!c_o>" -m "!c_no>"
        SOURCES ${_c_interface_sources}
    )

    set(_test_interface_sources
        "${PROJECT_SOURCE_DIR}/test/shared/test_prepare_matrix_template.F90"
        "${PROJECT_SOURCE_DIR}/test/shared/test_check_correctness_template.F90"
        "${PROJECT_SOURCE_DIR}/test/shared/test_analytic_template.F90"
        "${PROJECT_SOURCE_DIR}/test/shared/test_blacs_infrastructure.F90"
    )
    _elpa_extract_interfaces(
        "${PROJECT_BINARY_DIR}/test/shared/generated.h"
        MARKERS -m "!c>"
        SOURCES ${_test_interface_sources}
    )

    file(
        GLOB _fortran_intf_c_sources
        CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/helpers/*.c"
        "${PROJECT_SOURCE_DIR}/src/elpa2/kernels/*.c"
        "${PROJECT_SOURCE_DIR}/src/elpa2/kernels/*.s"
        "${PROJECT_SOURCE_DIR}/src/*.[ch]"
        "${PROJECT_SOURCE_DIR}/src/elpa_generalized/*.[ch]"
    )
    list(FILTER _fortran_intf_c_sources EXCLUDE REGEX "generated")
    _elpa_extract_interfaces(
        "${PROJECT_BINARY_DIR}/src/elpa_generated_fortran_interfaces.h"
        MARKERS -m "!f>" -m "#!f>"
        SOURCES ${_fortran_intf_c_sources}
    )

    file(
        GLOB _public_fortran_intf_sources
        CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/*.[ch]"
    )
    list(FILTER _public_fortran_intf_sources EXCLUDE REGEX "generated")
    _elpa_extract_interfaces(
        "${PROJECT_BINARY_DIR}/src/elpa_generated_public_fortran_interfaces.h"
        MARKERS -m "!pf>"
        SOURCES ${_public_fortran_intf_sources}
    )

    add_custom_target(
        elpa_generated_headers
        DEPENDS
            "${PROJECT_BINARY_DIR}/elpa/elpa_generated.h"
            "${PROJECT_BINARY_DIR}/elpa/elpa_constants.h"
            "${PROJECT_BINARY_DIR}/test/shared/generated.h"
            "${PROJECT_BINARY_DIR}/src/elpa_generated_fortran_interfaces.h"
            "${PROJECT_BINARY_DIR}/src/elpa_generated_public_fortran_interfaces.h"
    )
endfunction()
