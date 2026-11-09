# SPDX-License-Identifier: LGPL-3.0-only
# Resolve oneMKL as one coherent stack, including its MPI-specific BLACS.

AC_DEFUN([ELPA_FIND_MKL], [
  elpa_mkl_found=no
  elpa_mkl_blacs=none
  elpa_mkl_threading=$with_mkl_threading
  elpa_mkl_fortran=intel
  AS_IF([test "$ac_cv_fc_compiler_gnu" = yes], [elpa_mkl_fortran=gf])
  AS_CASE([$host_os], [darwin*], [elpa_mkl_fortran=intel])
  elpa_mkl_intel_compiler=no
  AX_FC_CHECK_DEFINE([__INTEL_COMPILER], [elpa_mkl_intel_compiler=yes])
  AX_FC_CHECK_DEFINE([__INTEL_LLVM_COMPILER], [elpa_mkl_intel_compiler=yes])
  AS_IF([test "$elpa_mkl_threading" = openmp], [
    AS_IF([test "$ac_cv_fc_compiler_gnu" = yes], [elpa_mkl_threading=gnu], [
      AS_IF([test "$elpa_mkl_intel_compiler" = yes], [elpa_mkl_threading=intel],
        [AC_MSG_ERROR([Cannot infer MKL OpenMP runtime for this Fortran compiler. Select sequential, gnu, intel or tbb explicitly.])])
    ])
  ])
  AS_CASE([$elpa_mkl_threading],
    [sequential], [elpa_mkl_thread=mkl_sequential; elpa_mkl_runtime=],
    [gnu], [
      AS_IF([test "$ac_cv_fc_compiler_gnu" != yes], [AC_MSG_ERROR([MKL GNU threading requires GNU Fortran; do not mix OpenMP runtimes])])
      elpa_mkl_thread=mkl_gnu_thread; elpa_mkl_runtime=-lgomp
    ],
    [intel], [
      AS_IF([test "$elpa_mkl_intel_compiler" != yes], [AC_MSG_ERROR([MKL Intel threading requires an Intel Fortran compiler; do not mix OpenMP runtimes])])
      elpa_mkl_thread=mkl_intel_thread; elpa_mkl_runtime=-liomp5
    ],
    [tbb], [elpa_mkl_thread=mkl_tbb_thread; elpa_mkl_runtime=-ltbb])
  elpa_mkl_interface=mkl_${elpa_mkl_fortran}_${with_blas_interface}
  AS_IF([test "$with_mkl_link" = static], [elpa_mkl_ext=a], [
    AS_CASE([$host_os], [darwin*], [elpa_mkl_ext=dylib], [elpa_mkl_ext=so])
  ])
  elpa_mkl_libdir=
  AS_IF([test -n "$with_mkl_root"], [
    AS_IF([test ! -d "$with_mkl_root"], [AC_MSG_ERROR([MKL root is not a directory: $with_mkl_root])])
    for elpa_dir in "$with_mkl_root/lib" "$with_mkl_root/lib/intel64" "$with_mkl_root/lib/intel64_lin" "$with_mkl_root/lib/intel64_mac"; do
      AS_IF([test -f "$elpa_dir/lib${elpa_mkl_interface}.$elpa_mkl_ext" &&
             test -f "$elpa_dir/lib${elpa_mkl_thread}.$elpa_mkl_ext" &&
             test -f "$elpa_dir/libmkl_core.$elpa_mkl_ext"], [elpa_mkl_libdir=$elpa_dir; break])
    done
    ELPA_MATH_CFLAGS="$MATH_CFLAGS -I$with_mkl_root/include"
  ])
  # Dynamic builds may use the compiler/system search path. Static builds need
  # concrete archives so that -lmkl_* cannot accidentally select shared objects.
  AS_IF([test "$with_mkl_link" = static && test -z "$elpa_mkl_libdir"], [
    AC_MSG_ERROR([Static MKL discovery requires --with-mkl-root or MKLROOT with the selected archives])
  ])
  elpa_mkl_system='-lpthread -lm'
  AS_CASE([$host_os], [linux*], [elpa_mkl_system="$elpa_mkl_system -ldl"])
  elpa_mkl_names="$elpa_mkl_interface $elpa_mkl_thread mkl_core"
  _ELPA_MKL_LINK_LINE
  elpa_mkl_base_libs=$elpa_mkl_link_line
  AC_MSG_CHECKING([for MKL $with_blas_interface, $elpa_mkl_threading, $with_mkl_link])
  AS_IF([test -z "$with_mkl_root" || test -n "$elpa_mkl_libdir"], [
    _ELPA_LINK_MATH([$elpa_mkl_base_libs], [])
    elpa_mkl_found=$elpa_math_link_ok
  ])
  AC_MSG_RESULT([$elpa_mkl_found])
])

AC_DEFUN([_ELPA_MKL_LINK_LINE], [
  elpa_mkl_link_line=
  for elpa_name in $elpa_mkl_names; do
    AS_IF([test "$with_mkl_link" = static],
      [elpa_mkl_link_line="$elpa_mkl_link_line $elpa_mkl_libdir/lib$elpa_name.a"],
      [elpa_mkl_link_line="$elpa_mkl_link_line -l$elpa_name"])
  done
  AS_IF([test "$with_mkl_link" = static], [
    AS_CASE([$host_os], [linux*],
      [elpa_mkl_link_line="-Wl,--start-group $elpa_mkl_link_line -Wl,--end-group"],
      [elpa_mkl_link_line="$elpa_mkl_link_line $elpa_mkl_link_line $elpa_mkl_link_line"])
  ], [
    AS_IF([test -n "$elpa_mkl_libdir"], [elpa_mkl_link_line="-L$elpa_mkl_libdir $elpa_mkl_link_line"])
  ])
  elpa_mkl_link_line="$elpa_mkl_link_line $elpa_mkl_runtime $elpa_mkl_system"
])

AC_DEFUN([ELPA_FIND_MKL_SCALAPACK], [
  elpa_mkl_blacs=$with_mkl_blacs
  AS_IF([test "$elpa_mkl_blacs" = auto], [elpa_mkl_blacs=$ELPA_MPI_VENDOR])
  AS_CASE([$elpa_mkl_blacs], [intelmpi|mpich|openmpi], [], [
    AC_MSG_ERROR([Cannot determine MKL BLACS ABI from the active MPI headers. Set --with-mkl-blacs explicitly; a successful link alone does not establish MPI ABI compatibility.])
  ])
  AS_CASE([$ELPA_MPI_VENDOR:$elpa_mkl_blacs],
    [openmpi:intelmpi|openmpi:mpich|mpich:openmpi|intelmpi:openmpi],
    [AC_MSG_ERROR([Requested MKL BLACS $elpa_mkl_blacs conflicts with MPI $ELPA_MPI_VENDOR])])
  AS_CASE([$elpa_mkl_blacs],
    [openmpi], [elpa_mkl_blacs_name=mkl_blacs_openmpi_$with_blas_interface],
    [mpich], [
      AS_CASE([$host_os], [darwin*], [elpa_mkl_blacs_name=mkl_blacs_mpich_$with_blas_interface],
        [elpa_mkl_blacs_name=mkl_blacs_intelmpi_$with_blas_interface])
    ],
    [elpa_mkl_blacs_name=mkl_blacs_intelmpi_$with_blas_interface])
  elpa_mkl_names="mkl_scalapack_$with_blas_interface $elpa_mkl_interface $elpa_mkl_thread mkl_core $elpa_mkl_blacs_name"
  _ELPA_MKL_LINK_LINE
  elpa_mkl_cluster_libs=$elpa_mkl_link_line
  AC_MSG_CHECKING([for MKL ScaLAPACK with $elpa_mkl_blacs BLACS])
  _ELPA_LINK_MATH([$elpa_mkl_cluster_libs], [call blacs_gridinit(); call pdtran()])
  AC_MSG_RESULT([$elpa_math_link_ok])
])
