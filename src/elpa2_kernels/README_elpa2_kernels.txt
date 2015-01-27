This file is intended as guideline for choosing one appropiate
ELPA2-kernel for your installation.

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

However, we cannot choose for you the best kernels, you should read
these hints, and maybe try which kernel works best for you.

Currently we offer the following alternatives for the ELPA2 kernels:

* elpa2_kernels_{real|complex}.f90       

                             - The generic FORTRAN version of the ELPA2 kernels
                               which should be useable on every platform.
                               It contains some hand optimizations (loop unrolling)
                               in the hope to get optimal code from most FORTRAN
                               compilers. The configure option "--with-generic"
                               uses these kernels. They are propably a good
                               default if you do not know which kernel
                               to use. Note that in the real version,
                               there is used a complex variable in
                               order to enforce better compiler
                               optimizations. This produces correct
                               code, however, some compilers might
                               produce a warning. 



* elpa2_kernels_{real|complex}_simple.f90  
                           
                             - Plain and simple version of elpa2_kernels.f90.
                               Please note that we observed that some compilers get
                               get confused by the hand optimizations done in
                               elpa2_kernels_{real|complex}.f90 and
                               give better performance with this
                               version - so it is worth to try both!
                               The configure option "--with-generic-simple"
                               uses these kernels. 

* elpa2_kernels_real_bgp.f90 
                             - Fortran code enhanced with assembler calls
                               for the IBM BlueGene/P. For the complex 
                               eigenvalue problem the "elpa2_kernels_complex.f90"
                               is recommended. The configure option 
                               "--with-generic-bgp" uses these
			       kernels. Note that the OpenMP functionality of
			       this kernel is not yet tested and thus an
			       preprocessor error is thrown in the combination
			       of this kernel with OpenMP. By manually editing
			       the file src/elpa2.F90 one can avoid this and 
			       test the OpenMP functionality. The ELPA
			       developers would welcome every feedback to this
			       subject.

* elpa2_kernels_real_bgq.f90 
                             - Fortran code enhanced with assembler calls
                               for the IBM BlueGene/Q. For the complex 
                               eigenvalue problem the "elpa2_kernels_complex.f90"
                               is recommended. The configure option 
                               "--with-generic-bgq" uses these
			       kernels. Note that the OpenMP functionality of
			       this kernel is not yet tested and thus an
			       preprocessor error is thrown in the combination
			       of this kernel with OpenMP. By manually editing
			       the file src/elpa2.F90 one can avoid this and
       			       test the OpenMP functionality. The ELPA 
                               developers would welcome every feedback
			       to this subject.

* elpa2_kernels_asm_x86_64.s
                             - Fortran code enhanced with assembler 
                               for the SSE vectorization. The configure option 
                               "--with-sse-assembler" uses these kernels. 
                               They are worth trying on x86_64 without AVX,
        		       e.g. Intel Nehalem. 

 

Several

* elpa2_kernels_{real|complex}_sse-avx_*.c(pp)     
                             - Optimized intrinisic code for x86_64
                               systems (i.e. Intel/AMD architecture)
                               using SSE2/SSE3 operations.
                               (Use gcc for compiling as Intel
			       compiler generates slower code!)

			       Note that you have to specify with
                               configure the flags 
         		       CFLAGS="-O3 -mavx -funsafe-loop-optimizations \
			       -funsafe-math-optimizations -ftree-vect-loop-version \
			       -ftree-vectorize"
			       and 
			       CXXFLAGS="-O3 -mavx -funsafe-loop-optimizations \
			       -funsafe-math-optimizations -ftree-vect-loop-version \
			       -ftree-vectorize"
			       for best performace results.

                               For convenience the flag
                               "--with-avx-optimization" sets these
                               CFLAGS and CXXFLAGS automatically.

                               On Intel Sandybridge architectures the
                               configure option "--with-avx-sandybride" 
			       uses the best combination, which is a 
                               combination of block2 for real matrices
                               and block1 for complex matrices.

                               On AMD Bulldozer architectures the
                               configure option "--with-amd-bulldozer" 
			       uses the best combination, which is a
			       combination of block4 for real matrices 
			       and block1 for complex matrices.

			       Otherwise, you can try out your own
			       combinations with the configure options 
                               "--with-avx-complex-block{1|2}" and
                               "--with-avx-real-block{2|4|6}".




So which version should be used?
================================

* On the IBM BlueGene/P, BlueGene/Q,
  you should get the optimal performance using the optimized intrinsics/assembler versions
  elpa2_kernels_{real|complex}_bg{p|q}.f90, respectively.
  

* On x86_64 systems (i.e. almost all Intel/AMD systems) you should get
  the optimal performance using the optimized intrinsics/assembler versions
  in elpa2_kernels_*.c or elpa2_kernels_{real|complex}_bg{p|q}.f90
  respectively. However, here you have quite some choice to find your
  optimal kernel.

* If you don't compile for one of these systems or you don't like to use assembler
  for any reason, it is likely that you are best off using elpa2_kernels.f90.
  Make a perfomance test with elpa2_kernels_simple.f90, however, to check if
  your compiler doesn't get confused by the hand optimizations.

* If you want to develop your own optimized kernels for you platform, it is
  easier to start with elpa2_kernels_simple.f90.
  Don't let you confuse from the huge code in elpa2_kernels.f90, the mathemathics
  done in the kernels is relatively trivial.
