dnl
dnl AX_FC_CHECK_DEFINE(MACRONAME, [ACTION_IF_DEFINED], [ACTION_IF_NOT_DEFINED])
dnl
AC_DEFUN([AX_FC_CHECK_DEFINE], [
        AC_LANG_PUSH([Fortran])
        AC_COMPILE_IFELSE([AC_LANG_SOURCE([
program test_define
#ifndef $1
  choke me
#endif
end program
        ])],
        [$2],
        [$3])
        AC_LANG_POP([Fortran])
])
