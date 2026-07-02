if(ELPA_OPENMP)
    # When MKL's intel_thread layer is in use with non-Intel compilers, the
    # OpenMP runtime must still be Intel's libiomp5 to avoid a dual-runtime
    # conflict. Clang can select it directly with -fopenmp=libiomp5. GNU
    # compilers still need -fopenmp for OpenMP code generation, but their link
    # step must be rewritten to use explicit libiomp5 instead of libgomp.
    # Trigger when: MKL intel_thread is selected AND iomp5 has not already
    # been chosen by the caller AND the Fortran compiler is not Intel
    # (the existing branch below handles that case).
    if(
        NOT CMAKE_Fortran_COMPILER_ID MATCHES "Intel"
        AND NOT DEFINED CACHE{OpenMP_C_LIB_NAMES}
        AND MKL_THREADING STREQUAL "intel_thread"
    )
        if(WIN32)
            set(_iomp_name "libiomp5md")
        else()
            set(_iomp_name "iomp5")
        endif()
        if(OMP_LIBRARY)
            set(_iomp5_lib "${OMP_LIBRARY}")
        else()
            find_library(_iomp5_lib NAMES ${_iomp_name})
        endif()
        if(_iomp5_lib)
            if(
                CMAKE_C_COMPILER_ID STREQUAL "GNU"
                OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
                OR CMAKE_Fortran_COMPILER_ID STREQUAL "GNU"
            )
                message(
                    STATUS
                    "ELPA: MKL intel_thread + GNU compilers: "
                    "compile with -fopenmp, link directly to libiomp5"
                )
                set(OpenMP_C_FLAGS "-fopenmp" CACHE STRING "" FORCE)
                set(OpenMP_CXX_FLAGS "-fopenmp" CACHE STRING "" FORCE)
                set(OpenMP_Fortran_FLAGS "-fopenmp" CACHE STRING "" FORCE)
                set(_elpa_openmp_link_flag_to_strip "-fopenmp")
                set(_elpa_openmp_link_strip_langs "C;CXX;Fortran")
            else()
                message(
                    STATUS
                    "ELPA: MKL intel_thread + non-Intel compilers: "
                    "using -fopenmp=libiomp5 (clang/flang-new native Intel OMP selection)"
                )
                # -fopenmp=libiomp5 is clang's explicit way to select Intel's
                # OpenMP runtime; omp.h is provided by the system libomp-dev
                # package (same standard API, ABI-compatible with libiomp5).
                set(OpenMP_C_FLAGS "-fopenmp=libiomp5" CACHE STRING "" FORCE)
                set(OpenMP_CXX_FLAGS "-fopenmp=libiomp5" CACHE STRING "" FORCE)
                set(OpenMP_Fortran_FLAGS "-fopenmp=libiomp5" CACHE STRING "" FORCE)
            endif()
            set(OpenMP_C_LIB_NAMES "${_iomp_name}" CACHE STRING "" FORCE)
            set(OpenMP_CXX_LIB_NAMES "${_iomp_name}" CACHE STRING "" FORCE)
            set(OpenMP_Fortran_LIB_NAMES "${_iomp_name}" CACHE STRING "" FORCE)
            set(OpenMP_${_iomp_name}_LIBRARY
                "${_iomp5_lib}"
                CACHE FILEPATH
                ""
                FORCE
            )
        endif()
        unset(_iomp5_lib)
        unset(_iomp_name)
    endif()

    if(
        CMAKE_Fortran_COMPILER_ID MATCHES "Intel"
        AND NOT DEFINED CACHE{OpenMP_C_LIB_NAMES}
    )
        if(WIN32)
            set(_iomp_name "libiomp5md")
        else()
            set(_iomp_name "iomp5")
        endif()
        if(OMP_LIBRARY)
            set(_iomp5_lib "${OMP_LIBRARY}")
        else()
            find_library(_iomp5_lib NAMES ${_iomp_name})
        endif()
        if(_iomp5_lib)
            message(
                STATUS
                "ELPA: Pre-setting OpenMP library to ${_iomp_name} (avoid dual-runtime)"
            )
            if(WIN32)
                set(OpenMP_C_FLAGS "-Xclang -fopenmp" CACHE STRING "" FORCE)
                set(OpenMP_CXX_FLAGS "-Xclang -fopenmp" CACHE STRING "" FORCE)
                set(OpenMP_Fortran_FLAGS "/Qopenmp" CACHE STRING "" FORCE)
            else()
                # ifx always uses -qopenmp (not -fopenmp) for correct iomp5 linkage.
                set(OpenMP_Fortran_FLAGS "-qopenmp" CACHE STRING "" FORCE)
                # C/CXX flags depend on which C compiler is used with ifx.
                if(CMAKE_C_COMPILER_ID MATCHES "IntelLLVM|Intel")
                    # Full Intel stack: use -qopenmp for all languages.
                    set(OpenMP_C_FLAGS "-qopenmp" CACHE STRING "" FORCE)
                    set(OpenMP_CXX_FLAGS "-qopenmp" CACHE STRING "" FORCE)
                elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
                    # gcc C/CXX + ifx Fortran: compile with -fopenmp, but strip
                    # from C/CXX link targets so only explicit libiomp5 is used.
                    # Fortran target (-qopenmp) is left intact.
                    message(
                        STATUS
                        "ELPA: gcc C/CXX + ifx Fortran: "
                        "compile -fopenmp, link explicit libiomp5 (C/CXX only)"
                    )
                    set(OpenMP_C_FLAGS "-fopenmp" CACHE STRING "" FORCE)
                    set(OpenMP_CXX_FLAGS "-fopenmp" CACHE STRING "" FORCE)
                    set(_elpa_openmp_link_flag_to_strip "-fopenmp")
                    set(_elpa_openmp_link_strip_langs "C;CXX")
                elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
                    # clang C/CXX + ifx Fortran: clang selects iomp5 directly.
                    message(
                        STATUS
                        "ELPA: clang C/CXX + ifx Fortran: "
                        "using -fopenmp=libiomp5 for C/CXX"
                    )
                    set(OpenMP_C_FLAGS "-fopenmp=libiomp5" CACHE STRING "" FORCE)
                    set(OpenMP_CXX_FLAGS "-fopenmp=libiomp5" CACHE STRING "" FORCE)
                else()
                    set(OpenMP_C_FLAGS "-fopenmp" CACHE STRING "" FORCE)
                    set(OpenMP_CXX_FLAGS "-fopenmp" CACHE STRING "" FORCE)
                endif()
            endif()
            set(OpenMP_C_LIB_NAMES "${_iomp_name}" CACHE STRING "" FORCE)
            set(OpenMP_CXX_LIB_NAMES "${_iomp_name}" CACHE STRING "" FORCE)
            set(OpenMP_Fortran_LIB_NAMES "${_iomp_name}" CACHE STRING "" FORCE)
            set(OpenMP_${_iomp_name}_LIBRARY
                "${_iomp5_lib}"
                CACHE FILEPATH
                ""
                FORCE
            )
        endif()
        unset(_iomp5_lib)
        unset(_iomp_name)
    endif()

    find_package(OpenMP REQUIRED COMPONENTS C CXX Fortran)

    # GNU compilers still need -fopenmp during compilation, but that same flag
    # would pull libgomp back into link commands. Strip it from the imported
    # targets and keep the explicit libiomp5 library selected above.
    if(NOT WIN32 AND DEFINED _elpa_openmp_link_flag_to_strip)
        foreach(_lang ${_elpa_openmp_link_strip_langs})
            set(_tgt "OpenMP::OpenMP_${_lang}")
            if(NOT TARGET ${_tgt})
                continue()
            endif()
            get_target_property(_libs ${_tgt} INTERFACE_LINK_LIBRARIES)
            if(_libs)
                list(REMOVE_ITEM _libs "${_elpa_openmp_link_flag_to_strip}")
                set_target_properties(${_tgt} PROPERTIES INTERFACE_LINK_LIBRARIES "${_libs}")
            endif()
            get_target_property(_opts ${_tgt} INTERFACE_LINK_OPTIONS)
            if(_opts)
                list(REMOVE_ITEM _opts "SHELL:${_elpa_openmp_link_flag_to_strip}")
                list(REMOVE_ITEM _opts "${_elpa_openmp_link_flag_to_strip}")
                set_target_properties(${_tgt} PROPERTIES INTERFACE_LINK_OPTIONS "${_opts}")
            endif()
        endforeach()
        unset(_libs)
        unset(_opts)
        unset(_tgt)
        unset(_lang)
        message(
            STATUS
            "ELPA: stripped -fopenmp from OpenMP imported targets "
            "(GNU link drivers would otherwise pull in libgomp; explicit libiomp5 is sufficient)"
        )
        unset(_elpa_openmp_link_flag_to_strip)
        unset(_elpa_openmp_link_strip_langs)
    endif()

    # CMake's FindOpenMP puts /Qopenmp in OpenMP::OpenMP_Fortran's
    # INTERFACE_LINK_LIBRARIES so that ifx.exe can locate the OpenMP runtime
    # when it acts as the linker driver.  That flag transitively reaches C/CXX
    # test executables which are linked by lld-link.exe (clang-cl's linker);
    # lld-link rejects /Qopenmp as an unknown file.  Strip it: libiomp5md.lib
    # (also in the interface) is sufficient for the runtime at link time.
    if(WIN32 AND CMAKE_Fortran_COMPILER_ID MATCHES "IntelLLVM")
        # CMake 4.x FindOpenMP puts "SHELL:/Qopenmp" in INTERFACE_LINK_OPTIONS
        # and the bare flag in INTERFACE_LINK_LIBRARIES.  Strip from both so
        # it does not leak transitively to C/CXX test executables linked by
        # lld-link (which rejects /Qopenmp as an unknown file).  The explicit
        # libiomp5md.lib in INTERFACE_LINK_LIBRARIES is sufficient.
        get_target_property(
            _omp_fortran_link_libs
            OpenMP::OpenMP_Fortran
            INTERFACE_LINK_LIBRARIES
        )
        if(_omp_fortran_link_libs)
            list(REMOVE_ITEM _omp_fortran_link_libs "/Qopenmp")
            list(REMOVE_ITEM _omp_fortran_link_libs "-Qopenmp")
            set_target_properties(
                OpenMP::OpenMP_Fortran
                PROPERTIES INTERFACE_LINK_LIBRARIES "${_omp_fortran_link_libs}"
            )
        endif()
        unset(_omp_fortran_link_libs)

        get_target_property(
            _omp_fortran_link_opts
            OpenMP::OpenMP_Fortran
            INTERFACE_LINK_OPTIONS
        )
        if(_omp_fortran_link_opts)
            list(REMOVE_ITEM _omp_fortran_link_opts "SHELL:/Qopenmp")
            list(REMOVE_ITEM _omp_fortran_link_opts "SHELL:-Qopenmp")
            list(REMOVE_ITEM _omp_fortran_link_opts "/Qopenmp")
            list(REMOVE_ITEM _omp_fortran_link_opts "-Qopenmp")
            set_target_properties(
                OpenMP::OpenMP_Fortran
                PROPERTIES INTERFACE_LINK_OPTIONS "${_omp_fortran_link_opts}"
            )
        endif()
        unset(_omp_fortran_link_opts)

        message(
            STATUS
            "ELPA: stripped /Qopenmp from OpenMP::OpenMP_Fortran "
            "INTERFACE_LINK_LIBRARIES and INTERFACE_LINK_OPTIONS "
            "(incompatible with lld-link; libiomp5md.lib is sufficient)"
        )
    endif()

    # Linux + CUDA: same problem in a different guise.  nvcc delegates its
    # link step to the CUDA host compiler (gcc/g++), which does not understand
    # -qopenmp.  The explicit libiomp5.so already in INTERFACE_LINK_LIBRARIES
    # is sufficient for runtime resolution, so strip the flag from all three
    # OpenMP imported targets.
    # The condition fires when ANY compiler in the mix is Intel-based, because
    # ifx sets -qopenmp on the Fortran target and cmake propagates it
    # transitively to C/CXX link lines via PUBLIC dependencies.
    if(NOT WIN32 AND ELPA_CUDA
       AND (CMAKE_C_COMPILER_ID MATCHES "IntelLLVM|Intel"
            OR CMAKE_Fortran_COMPILER_ID MATCHES "IntelLLVM|Intel"))
        foreach(_lang C CXX Fortran)
            set(_tgt "OpenMP::OpenMP_${_lang}")
            if(NOT TARGET ${_tgt})
                continue()
            endif()
            get_target_property(_libs ${_tgt} INTERFACE_LINK_LIBRARIES)
            if(_libs)
                list(REMOVE_ITEM _libs "-qopenmp")
                set_target_properties(${_tgt} PROPERTIES INTERFACE_LINK_LIBRARIES "${_libs}")
            endif()
            get_target_property(_opts ${_tgt} INTERFACE_LINK_OPTIONS)
            if(_opts)
                list(REMOVE_ITEM _opts "SHELL:-qopenmp")
                list(REMOVE_ITEM _opts "-qopenmp")
                set_target_properties(${_tgt} PROPERTIES INTERFACE_LINK_OPTIONS "${_opts}")
            endif()
        endforeach()
        unset(_libs)
        unset(_opts)
        unset(_tgt)
        unset(_lang)
        message(
            STATUS
            "ELPA: stripped -qopenmp from OpenMP imported targets "
            "(incompatible with gcc/g++ CUDA host linker; libiomp5.so is sufficient)"
        )
    endif()

    set(WITH_OPENMP_TRADITIONAL 1)

    # Check whether the Fortran compiler supports !$omp masked (OpenMP 5.1).
    # flang-new reports _OPENMP=199911 even though it handles the directive,
    # so a compile test is the only reliable detection method.
    # IMPORTANT: a compile-only check is insufficient here — AOCC classic flang
    # compiles successfully but only emits a warning for !$omp masked, silently
    # ignoring the directive at runtime (all threads enter the block).  A run
    # test catches this by verifying that only the master thread executes.
    include(CheckFortranSourceRuns)
    set(CMAKE_REQUIRED_FLAGS "${OpenMP_Fortran_FLAGS}")
    set(CMAKE_REQUIRED_LIBRARIES "${OpenMP_Fortran_LIBRARIES}")
    check_fortran_source_runs(
        "
        program test_masked
        use omp_lib
        implicit none
        integer :: n
        n = 0
        !$ call omp_set_num_threads(4)
        !$omp parallel shared(n)
        !$omp masked
        !$omp atomic
        n = n + 1
        !$omp end masked
        !$omp end parallel
        if (n /= 1) stop 1
        end program
        "
        _omp_masked_ok
        SRC_EXT F90
    )
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_LIBRARIES)
    if(_omp_masked_ok)
        set(HAVE_OMP_MASKED 1)
    endif()
else()
    set(WITH_OPENMP_TRADITIONAL 0)
endif()
