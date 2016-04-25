
dnl macro for testing whether the user wanted to compile only with one
dnl specific real kernel

dnl usage: DEFINE_OPTION([real-generic-kernel-only],[generic-kernel],[with_real_generic_kernel],[install_real_generic])

AC_DEFUN([DEFINE_OPTION_SPECIFIC_REAL_KERNEL],[
  AC_ARG_WITH([$1],
               AS_HELP_STRING([--with-$1],
                              [only compile $2 for real case]),
              [with_option=yes],[with_option=no])

   if test x"${with_option}" = x"yes" ; then
    if test x"${use_specific_real_kernel}" = x"no" ; then

    dnl make sure that all the other kernels are unset
    install_real_generic=no
    install_real_generic_simple=no
    install_real_sse_assembly=no
    install_real_bgp=no
    install_real_bgq=no
    install_real_sse_block2=no
    install_real_sse_block4=no
    install_real_sse_block6=no
    install_real_avx_block2=no
    install_real_avx_block4=no
    install_real_avx_block6=no
    want_sse=no
    want_avx=no
    want_avx2=no
    install_gpu=no

    use_specific_real_kernel=yes
    dnl now set the specific kernel
    $3=yes
    dnl take care of some dependencies
    if test x"${install_real_sse_block4}" = x"yes" ; then
      AC_MSG_NOTICE([$1 set. Also sse_block2 is needed])
      install_real_sse_block2=yes
    fi
    if test x"${install_real_avx_block4}" = x"yes" ; then
      AC_MSG_NOTICE([$1 set. Also avx_block2 is needed])
      install_real_avx_block2=yes
    fi
    if test x"${install_real_sse_block6}" = x"yes" ; then
      AC_MSG_NOTICE([$1 set. Also sse_block2 is needed])
      AC_MSG_NOTICE([$1 set. Also sse_block4 is needed])
      install_real_sse_block4=yes
      install_real_sse_block2=yes
    fi
    if test x"${install_real_avx_block6}" = x"yes" ; then
      AC_MSG_NOTICE([$1 set. Also avx_block2 is needed])
      AC_MSG_NOTICE([$1 set. Also avx_block4 is needed])
      install_real_avx_block4=yes
      install_real_avx_block2=yes
    fi

    dnl in case of SSE or AVX make sure that we can compile the choosen kernel
    if test x"${install_real_sse_assembly}" = x"yes" ; then
     if test x"${can_compile_sse_assembly}" = x"no" ; then
       AC_MSG_ERROR([$2 kernel was set, but cannot be compiled!])
     fi
    fi

    if test x"${install_real_sse_block2}" = x"yes" ; then
     if test x"${can_compile_sse_intrinsics}" = x"no" ; then
       AC_MSG_ERROR([$2 kernel was set, but cannot be compiled!])
     else
       want_sse=yes
     fi
    fi

    if test x"${install_real_sse_block4}" = x"yes" ; then
     if test x"${can_compile_sse_intrinsics}" = x"no" ; then
       AC_MSG_ERROR([$2 kernel was set, but cannot be compiled!])
     else
       want_sse=yes
     fi
    fi

    if test x"${install_real_sse_block6}" = x"yes" ; then
     if test x"${can_compile_sse_inrinsics}" = x"no" ; then
       AC_MSG_ERROR([$2 kernel was set, but cannot be compiled!])
     else
       want_sse=yes
     fi
    fi

    if test x"${install_real_sse_block2}" = x"yes" ; then
     if test x"${can_compile_sse_intrinsics}" = x"no" ; then
       AC_MSG_ERROR([$2 kernel was set, but cannot be compiled!])
     else
       want_sse=yes
     fi
    fi

    if test x"${install_real_sse_block4}" = x"yes" ; then
     if test x"${can_compile_sse_intrinsics}" = x"no" ; then
       AC_MSG_ERROR([$2 kernel was set, but cannot be compiled!])
     else
       want_sse=yes
     fi
    fi

    if test x"${install_real_sse_block6}" = x"yes" ; then
     if test x"${can_compile_sse_inrinsics}" = x"no" ; then
       AC_MSG_ERROR([$2 kernel was set, but cannot be compiled!])
     else
       want_sse=yes
     fi
    fi

    if test x"${install_real_avx_block2}" = x"yes" ; then
     if test x"${can_compile_avx}" = x"no" ; then
       AC_MSG_ERROR([$2 kernel was set, but cannot be compiled!])
     else
       want_avx=yes
     fi
    fi

    if test x"${install_real_avx_block4}" = x"yes" ; then
     if test x"${can_compile_avx}" = x"no" ; then
       AC_MSG_ERROR([$2 kernel was set, but cannot be compiled!])
     else
       want_avx=yes
     fi
    fi

    if test x"${install_real_avx_block6}" = x"yes" ; then
     if test x"${can_compile_avx}" = x"no" ; then
       AC_MSG_ERROR([$2 kernel was set, but cannot be compiled!])
     else
       want_avx=yes
     fi
    fi

    AC_MSG_NOTICE([$1 will be the only compiled kernel for real case])
#    if test x"${want_gpu}" = x"yes" ; then
#      AC_MSG_WARN([At the moment this disables GPU support!])
#      AC_MSG_WARN([IF GPU support is wanted do NOT specify a specific real kernel])
#    fi
   else
    AC_MSG_FAILURE([$1 failed; A specific kernel for real case has already been defined before!])
   fi
  fi
])


AC_DEFUN([DEFINE_OPTION_SPECIFIC_COMPLEX_KERNEL],[
  AC_ARG_WITH([$1],
                 AS_HELP_STRING([--with-$1],
                                [only compile $2 for complex case]),
              [with_option=yes],[with_option=no])

   if test x"${with_option}" = x"yes" ; then
    if test x"${use_specific_complex_kernel}" = x"no" ; then

    dnl make sure that all the other kernels are unset
    install_complex_generic=no
    install_complex_generic_simple=no
    install_complex_sse_assembly=no
    install_complex_bgp=no
    install_complex_bgq=no
    install_complex_sse_block1=no
    install_complex_sse_block2=no
    install_complex_avx_block1=no
    install_complex_avx_block2=no
    want_sse=no
    want_avx=no
    want_avx2=no

#    install_gpu=no
    use_specific_complex_kernel=yes
    dnl now set the specific kernel
    $3=yes
    dnl take care of some dependencies
    if test x"${install_complex_sse_block2}" = x"yes" ; then
      install_complex_sse_block1=yes
    fi
    if test x"${install_complex_avx_block2}" = x"yes" ; then
      install_complex_avx_block1=yes
    fi

    dnl in case of SSE or AVX make sure that we can compile the choosen kernel
    if test x"${install_complex_sse_assembly}" = x"yes" ; then
     if test x"${can_compile_sse_assembly}" = x"no" ; then
       AC_MSG_ERROR([$2 kernel was set, but cannot be compiled!])
     else
       want_sse=yes
     fi
    fi

    if test x"${install_complex_sse_block1}" = x"yes" ; then
     if test x"${can_compile_sse_intrinsics}" = x"no" ; then
       AC_MSG_ERROR([$2 kernel was set, but cannot be compiled!])
     else
       want_sse=yes
     fi
    fi

    if test x"${install_complex_sse_block2}" = x"yes" ; then
     if test x"${can_compile_sse_intrinsics}" = x"no" ; then
       AC_MSG_ERROR([$2 kernel was set, but cannot be compiled!])
     else
       want_sse=yes
     fi
    fi
    if test x"${install_complex_avx_block1}" = x"yes" ; then
     if test x"${can_compile_avx}" = x"no" ; then
       AC_MSG_ERROR([$2 kernel was set, but cannot be compiled!])
     else
       want_avx=yes
     fi
    fi

    if test x"${install_complex_avx_block2}" = x"yes" ; then
     if test x"${can_compile_avx}" = x"no" ; then
       AC_MSG_ERROR([$2 kernel was set, but cannot be compiled!])
     else
       want_avx=yes
     fi
    fi

    AC_MSG_NOTICE([$1 will be the only compiled kernel for complex case])
    if test x"${want_gpu}" = x"yes" ; then
      AC_MSG_WARN([At the moment this disables GPU support!])
      AC_MSG_WARN([IF GPU support is wanted do NOT specify a specific complex kernel])
    fi
   else
    AC_MSG_FAILURE([$1 failed; A specific kernel for complex case has already been defined before!])
   fi
  fi
])

