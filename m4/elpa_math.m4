# SPDX-License-Identifier: LGPL-3.0-only
# ELPA's math-provider contract. Discovery is link-only and does not change
# user CFLAGS, CPPFLAGS, FCFLAGS, LDFLAGS or LIBS.

AC_DEFUN([ELPA_WITH_CHOICE], [
  AC_ARG_WITH([$1], [AS_HELP_STRING([--with-$1=VALUE], [$4 ($3; default $2)])],
    [], [with_[]AS_TR_SH([$1])=$2])
  AS_CASE([$with_[]AS_TR_SH([$1])], [$3], [],
    [AC_MSG_ERROR([Invalid --with-$1=$with_[]AS_TR_SH([$1]); expected $3])])
])

AC_DEFUN([ELPA_MATH_OPTIONS], [
  ELPA_WITH_CHOICE([blas], [auto], [auto|mkl|openblas|flexiblas|generic|custom], [BLAS/LAPACK provider])
  ELPA_WITH_CHOICE([scalapack], [auto], [auto|mkl|generic|custom], [ScaLAPACK provider])
  ELPA_WITH_CHOICE([mkl-threading], [sequential], [sequential|openmp|gnu|intel|tbb], [MKL threading layer])
  ELPA_WITH_CHOICE([mkl-link], [dynamic], [dynamic|static], [MKL library linkage, independent of ELPA linkage])
  ELPA_WITH_CHOICE([mkl-blacs], [auto], [auto|intelmpi|mpich|openmpi], [MKL BLACS MPI ABI])
  AC_ARG_VAR([MKLROOT], [oneMKL installation root])
  AC_ARG_WITH([mkl-root], [AS_HELP_STRING([--with-mkl-root=DIR], [oneMKL installation root (otherwise MKLROOT)])],
    [], [with_mkl_root=$MKLROOT])
  AS_CASE([$with_mkl_root], [yes|no], [AC_MSG_ERROR([--with-mkl-root requires a directory])])
  AC_ARG_VAR([BLAS_LIBS], [Custom BLAS libraries; implies --with-blas=custom for auto])
  AC_ARG_VAR([LAPACK_LIBS], [Custom LAPACK libraries; empty if BLAS_LIBS already supplies LAPACK])
  AC_ARG_VAR([SCALAPACK_LIBS], [Custom ScaLAPACK/BLACS libraries; implies --with-scalapack=custom for auto])
  AC_ARG_VAR([MATH_CFLAGS], [Optional dependency-local include/preprocessor flags for a custom math stack])
  AC_ARG_VAR([SCALAPACK_LDFLAGS], [Extra link flags for the math libraries (legacy interface)])
  AC_ARG_VAR([SCALAPACK_FCFLAGS], [Extra Fortran flags for the math libraries (legacy interface)])

])

# Both integer switches select the existing ELPA integer-math implementation.
# Keep the old option, with its LP64 default; reject only conflicting requests.
AC_DEFUN([ELPA_MATH_INTEGER_INTERFACE], [
  AC_ARG_ENABLE([64bit-integer-math-support],
    [AS_HELP_STRING([--enable-64bit-integer-math-support],
      [link against 64-bit integer BLAS/LAPACK/ScaLAPACK (default: no)])],
    [elpa_integer_option_set=yes],
    [elpa_integer_option_set=no; enable_64bit_integer_math_support=no])
  AS_IF([test "x$enable_64bit_integer_math_support" = xyes],
    [elpa_default_interface=ilp64], [elpa_default_interface=lp64])
  ELPA_WITH_CHOICE([blas-interface], [$elpa_default_interface], [lp64|ilp64], [Math-library integer interface])
  AS_IF([test "$elpa_integer_option_set" = yes &&
         test "$with_blas_interface" != "$elpa_default_interface"],
    [AC_MSG_ERROR([--with-blas-interface conflicts with --enable/disable-64bit-integer-math-support])])
  AS_IF([test "$with_blas_interface" = ilp64],
    [enable_64bit_integer_math_support=yes], [enable_64bit_integer_math_support=no])
])

# The program only links: it must never be executed. In particular, the
# no-argument calls to auxiliary routines below check symbol availability.
AC_DEFUN([_ELPA_LINK_MATH], [
  elpa_probe_save_LIBS=$LIBS
  elpa_probe_save_FCFLAGS=$FCFLAGS
  LIBS="$1 $SCALAPACK_LDFLAGS $elpa_user_LIBS"
  FCFLAGS="$FCFLAGS $SCALAPACK_FCFLAGS $ELPA_MATH_CFLAGS"
  AC_LANG_PUSH([Fortran])
  AC_LINK_IFELSE([AC_LANG_SOURCE([
program elpa_math_link
  implicit none
  integer, parameter :: ik = $elpa_math_integer_kind
  integer(ik) :: n, info, lwork
  double precision :: a(1,1), b(1,1), c(1,1), w(1), work(16)
  n = 1
  lwork = 16
  a = 1.0d0
  b = 1.0d0
  c = 0.0d0
  call dgemm('N','N',n,n,n,1.0d0,a,n,b,n,0.0d0,c,n)
  m4_if([$3], [blas-only], [], [call dsyev('V','U',n,c,n,w,work,lwork,info)
  call dlarrv()])
  $2
end program
  ])], [elpa_math_link_ok=yes], [elpa_math_link_ok=no])
  AC_LANG_POP([Fortran])
  LIBS=$elpa_probe_save_LIBS
  FCFLAGS=$elpa_probe_save_FCFLAGS
])

AC_DEFUN([ELPA_DETECT_MPI_VENDOR], [
  ELPA_MPI_VENDOR=none
  AS_IF([test "$with_mpi" = yes], [
    ELPA_MPI_VENDOR=unknown
    AC_LANG_PUSH([C])
    AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
#include <mpi.h>
#ifndef I_MPI_VERSION
#error not Intel MPI
#endif
int main(void) { return 0; }
    ]])], [ELPA_MPI_VENDOR=intelmpi])
    AS_IF([test "$ELPA_MPI_VENDOR" = unknown], [
      AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
#include <mpi.h>
#ifndef OMPI_MAJOR_VERSION
#error not Open MPI
#endif
int main(void) { return 0; }
      ]])], [ELPA_MPI_VENDOR=openmpi])
    ])
    AS_IF([test "$ELPA_MPI_VENDOR" = unknown], [
      AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
#include <mpi.h>
#if !defined(MPICH_NAME) && !defined(MPICH_VERSION)
#error not MPICH
#endif
int main(void) { return 0; }
      ]])], [ELPA_MPI_VENDOR=mpich])
    ])
    AC_LANG_POP([C])
  ])
  AC_SUBST([ELPA_MPI_VENDOR])
])

AC_DEFUN([ELPA_FIND_MATH], [
  AC_PATH_TOOL([PKG_CONFIG], [pkg-config], [false])
  elpa_pc_libs='--libs'
  AS_IF([test "x$enable_shared" = xno], [elpa_pc_libs='--libs --static'])
  elpa_user_LIBS=$LIBS
  ELPA_MATH_CFLAGS=$MATH_CFLAGS
  ELPA_MATH_LIBS=
  ELPA_BLAS_PROVIDER=
  ELPA_SCALAPACK_PROVIDER=none
  elpa_math_integer_kind=4
  AS_IF([test "$with_blas_interface" = ilp64], [elpa_math_integer_kind=8])
  ELPA_DETECT_MPI_VENDOR

  AS_IF([test -n "$BLAS_LIBS$LAPACK_LIBS"], [
    AS_CASE([$with_blas], [auto], [with_blas=custom], [custom], [],
      [AC_MSG_ERROR([BLAS_LIBS/LAPACK_LIBS cannot be combined with --with-blas=$with_blas])])
  ])
  AS_IF([test -n "$SCALAPACK_LIBS"], [
    AS_CASE([$with_scalapack], [auto], [with_scalapack=custom], [custom], [],
      [AC_MSG_ERROR([SCALAPACK_LIBS cannot be combined with --with-scalapack=$with_scalapack])])
  ])

  AS_IF([test "$with_blas" = auto], [
    AC_MSG_CHECKING([for BLAS/LAPACK in the supplied compiler and link flags])
    _ELPA_LINK_MATH([], [])
    AC_MSG_RESULT([$elpa_math_link_ok])
    AS_IF([test "$elpa_math_link_ok" = yes], [ELPA_BLAS_PROVIDER=user], [
      # As before, a supplied BLAS without LAPACK may use a separate -llapack;
      # do not choose a second BLAS provider in front of the user's library.
      _ELPA_LINK_MATH([], [], [blas-only])
      AS_IF([test "$elpa_math_link_ok" = yes], [
        _ELPA_LINK_MATH([-llapack], [])
        AS_IF([test "$elpa_math_link_ok" = yes], [
          ELPA_BLAS_PROVIDER=user
          ELPA_MATH_LIBS=-llapack
        ])
      ])
    ])
  ])

  AS_IF([test "$with_blas" = custom], [
    AS_IF([test -z "$BLAS_LIBS"], [AC_MSG_ERROR([--with-blas=custom requires BLAS_LIBS])])
    ELPA_MATH_LIBS="$LAPACK_LIBS $BLAS_LIBS"
    _ELPA_LINK_MATH([$ELPA_MATH_LIBS], [])
    AS_IF([test "$elpa_math_link_ok" != yes], [AC_MSG_ERROR([Custom BLAS/LAPACK do not link; see config.log])])
    ELPA_BLAS_PROVIDER=custom
  ])

  # Auto considers MKL only when explicitly made discoverable. It does not
  # scan arbitrary /opt installations or silently override a system provider.
  AS_IF([test "$with_blas" = mkl || { test -z "$ELPA_BLAS_PROVIDER" && test "$with_blas" = auto && test -d "$with_mkl_root"; }], [
    ELPA_FIND_MKL
    AS_IF([test "$elpa_mkl_found" = yes], [
      ELPA_MATH_LIBS=$elpa_mkl_base_libs
      ELPA_BLAS_PROVIDER=mkl
    ], [
      AS_IF([test "$with_blas" = mkl], [AC_MSG_ERROR([Requested MKL stack does not link. Check --with-mkl-root/MKLROOT, interface and threading; see config.log.])])
      ELPA_MATH_CFLAGS=$MATH_CFLAGS
    ])
  ])

  AS_IF([test -z "$ELPA_BLAS_PROVIDER"], [
    # Unversioned Fortran symbols do not reveal the integer ABI. Do not infer
    # ILP64 from a successful dgemm_ link or silently select an LP64 library.
    AS_IF([test "$with_blas_interface" = ilp64], [
      AC_MSG_ERROR([Automatic ILP64 discovery requires MKL. For another ILP64 implementation select --with-blas=custom and provide its BLAS_LIBS/LAPACK_LIBS explicitly.])
    ])
    AS_CASE([$with_blas], [auto], [elpa_providers='flexiblas openblas satlas generic'],
      [elpa_providers=$with_blas])
    for elpa_provider in $elpa_providers; do
      AS_CASE([$elpa_provider],
        [generic], [elpa_pc_modules='lapack blas'; elpa_fallback='-llapack -lblas'],
        [elpa_pc_modules=$elpa_provider; elpa_fallback="-l$elpa_provider"])
      elpa_candidate=$elpa_fallback
      ELPA_MATH_CFLAGS=$MATH_CFLAGS
      AS_IF([$PKG_CONFIG --exists $elpa_pc_modules], [
        elpa_candidate=`$PKG_CONFIG $elpa_pc_libs $elpa_pc_modules`
        ELPA_MATH_CFLAGS="$MATH_CFLAGS `$PKG_CONFIG --cflags $elpa_pc_modules`"
      ])
      AC_MSG_CHECKING([for $elpa_provider BLAS/LAPACK])
      _ELPA_LINK_MATH([$elpa_candidate], [])
      # OpenBLAS can be built without LAPACK; use a separate LAPACK if needed.
      AS_IF([test "$elpa_math_link_ok" != yes && test "$elpa_provider" != generic], [
        elpa_lapack='-llapack'
        AS_IF([$PKG_CONFIG --exists lapack], [elpa_lapack=`$PKG_CONFIG $elpa_pc_libs lapack`])
        elpa_candidate="$elpa_lapack $elpa_candidate"
        _ELPA_LINK_MATH([$elpa_candidate], [])
      ])
      AC_MSG_RESULT([$elpa_math_link_ok])
      AS_IF([test "$elpa_math_link_ok" = yes], [
        ELPA_BLAS_PROVIDER=$elpa_provider
        ELPA_MATH_LIBS=$elpa_candidate
        break
      ])
    done
    AS_IF([test -z "$ELPA_BLAS_PROVIDER"], [AC_MSG_ERROR([No usable $with_blas BLAS/LAPACK. Expose the provider through PKG_CONFIG_PATH, or select a custom stack with BLAS_LIBS/LAPACK_LIBS.])])
  ])

  AS_IF([test "$with_mpi" = yes], [
    AS_IF([test "$with_scalapack" = auto], [
      AC_MSG_CHECKING([for ScaLAPACK/BLACS in the supplied compiler and link flags])
      _ELPA_LINK_MATH([$ELPA_MATH_LIBS], [call blacs_gridinit(); call pdtran()])
      AC_MSG_RESULT([$elpa_math_link_ok])
      AS_IF([test "$elpa_math_link_ok" = yes], [ELPA_SCALAPACK_PROVIDER=user])
    ])
    AS_IF([test "$with_scalapack" = mkl && test "$ELPA_BLAS_PROVIDER" != mkl],
      [AC_MSG_ERROR([--with-scalapack=mkl requires MKL BLAS/LAPACK])])
    AS_IF([test "$with_scalapack" = mkl || { test "$with_scalapack" = auto && test "$ELPA_SCALAPACK_PROVIDER" = none && test "$ELPA_BLAS_PROVIDER" = mkl; }], [
      ELPA_FIND_MKL_SCALAPACK
      AS_IF([test "$elpa_math_link_ok" = yes], [
        ELPA_SCALAPACK_PROVIDER=mkl
        ELPA_MATH_LIBS=$elpa_mkl_cluster_libs
      ], [
        AS_IF([test "$with_scalapack" = mkl], [AC_MSG_ERROR([Requested MKL ScaLAPACK/BLACS stack does not link; check --with-mkl-blacs and config.log])])
      ])
    ])
    AS_IF([test "$ELPA_SCALAPACK_PROVIDER" = none], [
      AS_IF([test "$with_scalapack" = custom], [
        AS_IF([test -z "$SCALAPACK_LIBS"], [AC_MSG_ERROR([--with-scalapack=custom requires SCALAPACK_LIBS])])
        elpa_scalapack_candidates=custom
      ], [
        AS_IF([test "$with_blas_interface" = ilp64], [AC_MSG_ERROR([Non-MKL ILP64 ScaLAPACK requires --with-scalapack=custom and SCALAPACK_LIBS])])
        # An MPI-specific package comes first; never try the other MPI's name.
        AS_CASE([$ELPA_MPI_VENDOR], [openmpi|mpich],
          [elpa_scalapack_candidates="scalapack-$ELPA_MPI_VENDOR scalapack mpiscalapack"],
          [elpa_scalapack_candidates="mpiscalapack scalapack"])
      ])
      elpa_base_cflags=$ELPA_MATH_CFLAGS
      for elpa_scalapack in $elpa_scalapack_candidates; do
        ELPA_MATH_CFLAGS=$elpa_base_cflags
        AS_IF([test "$elpa_scalapack" = custom], [elpa_candidate=$SCALAPACK_LIBS], [
          elpa_candidate="-l$elpa_scalapack"
          AS_IF([$PKG_CONFIG --exists "$elpa_scalapack"], [
            elpa_candidate=`$PKG_CONFIG $elpa_pc_libs "$elpa_scalapack"`
            ELPA_MATH_CFLAGS="$ELPA_MATH_CFLAGS `$PKG_CONFIG --cflags "$elpa_scalapack"`"
          ])
        ])
        AC_MSG_CHECKING([for $elpa_scalapack ScaLAPACK and BLACS])
        _ELPA_LINK_MATH([$elpa_candidate $ELPA_MATH_LIBS], [call blacs_gridinit(); call pdtran()])
        # Retain legacy installations with a separately packaged BLACS. Do not
        # try Open MPI's names with a different known MPI implementation.
        AS_IF([test "$elpa_math_link_ok" != yes && test "$elpa_scalapack" != custom], [
          elpa_blacs_candidates='mpiblacs blacs mpiblacsF77init'
          AS_IF([test "$ELPA_MPI_VENDOR" = openmpi],
            [elpa_blacs_candidates="blacs-openmpi $elpa_blacs_candidates"])
          for elpa_blacs in $elpa_blacs_candidates; do
            AS_CASE([$elpa_blacs],
              [mpiblacsF77init], [elpa_blacs_libs='-lmpiblacsF77init -lmpiblacs'],
              [blacs-openmpi], [elpa_blacs_libs='-lblacs-openmpi -lblacsCinit-openmpi'],
              [elpa_blacs_libs="-l$elpa_blacs"])
            _ELPA_LINK_MATH([$elpa_candidate $elpa_blacs_libs $ELPA_MATH_LIBS], [call blacs_gridinit(); call pdtran()])
            AS_IF([test "$elpa_math_link_ok" = yes], [
              elpa_candidate="$elpa_candidate $elpa_blacs_libs"
              break
            ])
          done
        ])
        AC_MSG_RESULT([$elpa_math_link_ok])
        AS_IF([test "$elpa_math_link_ok" = yes], [
          ELPA_SCALAPACK_PROVIDER=$elpa_scalapack
          ELPA_MATH_LIBS="$elpa_candidate $ELPA_MATH_LIBS"
          break
        ])
      done
      AS_IF([test "$ELPA_SCALAPACK_PROVIDER" = none], [AC_MSG_ERROR([No ScaLAPACK/BLACS matching the selected MPI. Set PKG_CONFIG_PATH for that MPI stack, or SCALAPACK_LIBS for a custom implementation.])])
    ])
  ], [
    AS_IF([test "$with_scalapack" != auto], [AC_MSG_ERROR([--with-scalapack requires --with-mpi=yes])])
  ])

  # Legacy flags stay in their original variables for callers inspecting them.
  # Append them once to the ordered link line, without changing global flags.
  ELPA_MATH_LIBS="$ELPA_MATH_LIBS $SCALAPACK_LDFLAGS"
  WITH_MKL=
  WITH_BLACS=1
  AS_IF([test "$ELPA_BLAS_PROVIDER" = mkl], [WITH_MKL=1; WITH_BLACS=])
  AC_MSG_NOTICE([Math libraries: BLAS/LAPACK=$ELPA_BLAS_PROVIDER ($with_blas_interface), ScaLAPACK=$ELPA_SCALAPACK_PROVIDER, MPI=$ELPA_MPI_VENDOR])
  AC_SUBST([ELPA_MATH_LIBS])
  AC_SUBST([ELPA_MATH_CFLAGS])
  AC_SUBST([ELPA_BLAS_PROVIDER])
  AC_SUBST([ELPA_SCALAPACK_PROVIDER])
])
