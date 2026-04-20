# ElpaInstall.cmake — Installation rules for ELPA

include(GNUInstallDirs)

set(_libname "elpa${ELPA_SUFFIX}")
set(_version_dir "elpa${ELPA_SUFFIX}-${PROJECT_VERSION}")

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
install(
    TARGETS ${_libname}
    EXPORT ElpaTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

# ---------------------------------------------------------------------------
# Fortran modules
# ---------------------------------------------------------------------------
install(
    DIRECTORY "${CMAKE_BINARY_DIR}/modules/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${_version_dir}/modules"
    FILES_MATCHING
    PATTERN "*.mod"
)

# ---------------------------------------------------------------------------
# C/C++ headers
# ---------------------------------------------------------------------------
# Public headers from source tree (matches autotools nobase_elpa_include_HEADERS)
install(
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/elpa/elpa.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/elpa/elpa_generic.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/elpa/elpa_explicit_name.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/elpa/elpa_simd_constants.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${_version_dir}/elpa"
)

install(
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/helpers/lapack_interfaces.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/helpers/scalapack_interfaces.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${_version_dir}/src/helpers"
)

# Generated headers (matches autotools nobase_nodist_elpa_include_HEADERS)
install(
    FILES
        "${CMAKE_BINARY_DIR}/elpa/elpa_version.h"
        "${CMAKE_BINARY_DIR}/elpa/elpa_constants.h"
        "${CMAKE_BINARY_DIR}/elpa/elpa_generated.h"
        "${CMAKE_BINARY_DIR}/elpa/elpa_generated_c_api.h"
        "${CMAKE_BINARY_DIR}/elpa/elpa_configured_options.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${_version_dir}/elpa"
)

# ---------------------------------------------------------------------------
# pkg-config
# ---------------------------------------------------------------------------
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/elpa.pc.cmake.in"
    "${CMAKE_BINARY_DIR}/${_libname}.pc"
    @ONLY
)
install(
    FILES "${CMAKE_BINARY_DIR}/${_libname}.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)

# ---------------------------------------------------------------------------
# CMake package config
# ---------------------------------------------------------------------------
if(ELPA_INSTALL_CMAKE_PACKAGE)
    include(CMakePackageConfigHelpers)

    install(
        EXPORT ElpaTargets
        FILE ElpaTargets.cmake
        NAMESPACE ELPA::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/elpa"
    )

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ElpaConfigPackage.cmake.in"
        "${CMAKE_BINARY_DIR}/ElpaConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/elpa"
    )

    write_basic_package_version_file(
        "${CMAKE_BINARY_DIR}/ElpaConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY AnyNewerVersion
    )

    install(
        FILES
            "${CMAKE_BINARY_DIR}/ElpaConfig.cmake"
            "${CMAKE_BINARY_DIR}/ElpaConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/elpa"
    )
endif()
