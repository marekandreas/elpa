include(CheckFortranSourceCompiles)
include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckTypeSize)
include(CMakePushCheckState)

set(USE_FORTRAN2008 1)
set(USE_ASSUMED_SIZE 1)

cmake_push_check_state(RESET)
if(TARGET MPI::MPI_Fortran)
    set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_Fortran)
else()
    if(MPI_Fortran_COMPILE_FLAGS)
        set(CMAKE_REQUIRED_FLAGS "${MPI_Fortran_COMPILE_FLAGS}")
    endif()
    if(MPI_Fortran_LINK_FLAGS)
        string(STRIP "${MPI_Fortran_LINK_FLAGS}" _mpi_link_stripped2)
        separate_arguments(
            _mpi_link_list2
            NATIVE_COMMAND
            "${_mpi_link_stripped2}"
        )
        list(APPEND CMAKE_REQUIRED_LIBRARIES ${_mpi_link_list2})
    endif()
    if(MPI_Fortran_LIBRARIES)
        list(APPEND CMAKE_REQUIRED_LIBRARIES ${MPI_Fortran_LIBRARIES})
    endif()
endif()

check_fortran_source_compiles(
    "\n  program test_iso\n    use iso_fortran_env, only : error_unit\n    implicit none\n    write(error_unit,*) 'ok'\n  end program\n"
    HAVE_ISO_FORTRAN_ENV
    SRC_EXT F90
)

check_fortran_source_compiles(
    "\n  program test_env\n    character(len=256) :: homedir\n    call get_environment_variable('HOME', homedir)\n  end program\n"
    HAVE_ENVIRONMENT_CHECKING
    SRC_EXT F90
)

if(ELPA_MPI)
    check_fortran_source_compiles(
        "\n      program test_mpi_module\n        use mpi\n        implicit none\n        integer :: ierr, rank\n        real(8) :: a(2)\n        complex(8) :: b(2)\n        call MPI_Init(ierr)\n        call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)\n        call MPI_Bcast(a, 2, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)\n        call MPI_Bcast(b, 2, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD, ierr)\n        call MPI_Finalize(ierr)\n      end program\n    "
        _MPI_MODULE_FULL_INTERFACES
        SRC_EXT F90
    )

    if(_MPI_MODULE_FULL_INTERFACES)
        set(HAVE_MPI_MODULE 1)
    else()
        set(HAVE_MPI_MODULE 0)
        message(
            STATUS
            "ELPA: MPI Fortran module does not provide all interfaces; using mpif.h"
        )
    endif()
endif()
cmake_pop_check_state()

if(ELPA_SINGLE_PRECISION)
    set(WANT_SINGLE_PRECISION_REAL 1)
    set(WANT_SINGLE_PRECISION_COMPLEX 1)
endif()

if(ELPA_SKEWSYMMETRIC)
    set(HAVE_SKEWSYMMETRIC 1)
endif()

if(ELPA_TIMINGS)
    set(HAVE_DETAILED_TIMINGS 1)
endif()

if(ELPA_AUTOTUNE)
    set(ENABLE_AUTOTUNING 1)
endif()

set(ENABLE_C_TESTS 1)
set(ENABLE_CPP_TESTS 1)
set(ENABLE_FORTRAN_TESTS 1)

check_include_file(stdint.h HAVE_STDINT_H)
check_include_file(inttypes.h HAVE_INTTYPES_H)
check_include_file(stdio.h HAVE_STDIO_H)
check_include_file(stdlib.h HAVE_STDLIB_H)
check_include_file(string.h HAVE_STRING_H)
check_include_file(strings.h HAVE_STRINGS_H)
check_include_file(sys/stat.h HAVE_SYS_STAT_H)
check_include_file(sys/types.h HAVE_SYS_TYPES_H)
check_include_file(unistd.h HAVE_UNISTD_H)

check_type_size("long int" SIZEOF_LONG_INT)

cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)
check_symbol_exists(sched_setaffinity "sched.h" _HAVE_SCHED_SETAFFINITY)
cmake_pop_check_state()
if(_HAVE_SCHED_SETAFFINITY)
    set(HAVE_AFFINITY_CHECKING 1)
else()
    set(HAVE_AFFINITY_CHECKING 0)
endif()

if(ELPA_64BIT_INTEGER_MATH)
    set(HAVE_64BIT_INTEGER_MATH_SUPPORT 1)
endif()
if(ELPA_64BIT_INTEGER_MPI)
    set(HAVE_64BIT_INTEGER_MPI_SUPPORT 1)
endif()

if(ELPA_MPI AND ELPA_OPENMP)
    set(THREADING_SUPPORT_CHECK 1)
    set(ALLOW_THREAD_LIMITING 1)
endif()

if(ELPA_MPI AND HAVE_MPI_MODULE)
    set(PACK_REAL_TO_COMPLEX 1)
else()
    set(PACK_REAL_TO_COMPLEX 0)
endif()

set(BAND_TO_FULL_BLOCKING 1)
set(STDC_HEADERS 1)

if(ELPA_STORE_BUILD_CONFIG)
    find_program(XXD xxd REQUIRED)
    set(STORE_BUILD_CONFIG 1)
endif()
