
dnl macro for testing whether the user wanted to compile only with the GPU version

dnl usage: DEFINE_OPTION([gpu-support-only],[gpu-support],[with_gpu_support],[install_gpu])

AC_DEFUN([DEFINE_OPTION_GPU_SUPPORT_ONLY],[
  AC_ARG_WITH([$1],
               AS_HELP_STRING([--with-$1],
                              [only compile $2 ]),
              [with_option=yes],[with_option=no])

  if test x"${with_option}" = x"yes" ; then
    dnl make sure that all the other kernels are unset
    install_real_generic=no
    install_real_generic_simple=no
    install_real_sse=no
    install_real_bgp=no
    install_real_bgq=no
    install_real_avx_block2=no
    install_real_avx_block4=no
    install_real_avx_block6=no


    install_complex_generic=no
    install_complex_generic_simple=no
    install_complex_sse=no
    install_complex_bgp=no
    install_complex_bgq=no
    install_complex_avx_block1=no
    install_complex_avx_block2=no


    install_gpu=yes

    want_avx=no

    build_with_gpu_support_only=yes
    use_specific_complex_kernel=yes
    use_specific_real_kernel=yes
    dnl now set the specific kernel
    $3=yes

    AC_MSG_NOTICE([ELPA will be build only with $1])
  else
    build_with_gpu_support_only=no
  fi
])

