ELPA generally uses BLAS-Routines for all compute intensive work
so that the performance of ELPA mainly depends on the quality of
the BLAS implementation used when linking.

The only exception is the backtransformation of the eigenvectors
for the 2-stage solver (ELPA2). In this case BLAS routines cannot
be used effectively due to the nature of the problem.

The compute intensive part of the backtransformation of ELPA2
has been put to a file of its own (elpa2_kernels.f90) so that
this can be replaced by hand tailored, optimized code for
specific platforms.

Currently we offer the following alternatives for the ELPA2 kernels:

* elpa2_kernels.f90          - The generic FORTRAN version of the ELPA2 kernels
                               which should be useable on every platform.
                               It contains some hand optimizations (loop unrolling)
                               in the hope to get optimal code from most FORTRAN
                               compilers.

* elpa2_kernels_simple.f90   - Plain and simple version of elpa2_kernels.f90.
                               Please note that we observed that some compilers get
                               get confused by the hand optimizations done in
                               elpa2_kernels.f90 and give better performance
                               with this version - so it is worth to try both!

* elpa2_kernels_bg.f90       - Fortran code enhanced with assembler calls
                               for the IBM BlueGene/P

* elpa2_kernels_asm_x86_64.s - Optimized assembler code for x86_64
                               systems (i.e. Intel/AMD architecture)
                               using SSE2/SSE3 operations.
                               (Use GNU assembler for assembling!)


So which version should be used?
================================

* On x86_64 systems (i.e. almost all Intel/AMD systems) or on the IBM BlueGene/P
  you should get the optimal performance using the optimized assembler versions
  in elpa2_kernels_asm_x86_64.s or elpa2_kernels_bg.f90 respectively.

* If you don't compile for one of these systems or you don't like to use assembler
  for any reason, it is likely that you are best off using elpa2_kernels.f90.
  Make a perfomance test with elpa2_kernels_simple.f90, however, to check if
  your compiler doesn't get confused by the hand optimizations.

* If you want to develop your own optimized kernels for you platform, it is
  easier to start with elpa2_kernels_simple.f90.
  Don't let you confuse from the huge code in elpa2_kernels.f90, the mathemathics
  done in the kernels is relatively trivial.
