## Performance tuning og the *ELPA*-library ##

This document should help the user to obtain the maximum performance of the *ELPA*-library and to avoid some pitfals which can influence the performance quite dramatically.

### Optimal build of the library ###
Please make sure that *ELPA* is build with compiler optimization flags and that the SIMD vectorization for your hardware is enabled. During the configure step should warn you, if the SIMD vectorization is not set for the CPU you are building ELPA on. However, if you want to cross-compile *ELPA* you have to take care yourself. Please make sure that at the end of the configure output, the appropiate kernels for your hardware are activated. For example on the latest Intel hardware, which supports AVX-512 instructions you should see an output like this:

```Fortran
The following ELPA2 kernels will be build:

  real_generic
  real_generic_simple
  real_generic_simple_block4
  real_generic_simple_block6
  real_sse_block2
  real_sse_block4
  real_sse_block6
  real_sse_assembly
  real_avx_block2
  real_avx_block4
  real_avx_block6
  real_avx2_block2
  real_avx2_block4
  real_avx2_block6
  real_avx512_block2 (default)
  real_avx512_block4
  real_avx512_block6
  complex_generic
  complex_generic_simple
  complex_sse_block1
  complex_sse_block2
  complex_sse_assembly
  complex_avx_block1
  complex_avx_block2
  complex_avx2_block1
  complex_avx2_block2
  complex_avx512_block1 (default)
  complex_avx512_block2
```

If you build the GPU version of ELPA please make sure that you set during the configure step the compute capability to the highest level your Nvidia GPU cards support.Please also make sure that at the end of the configure steps the GPU kernels are listed.

### Runtime pitfalls ###

#### MPI-only runs ####
The performance of the ELPA library is best, if the 2D-MPI-grid is quadratic, or at least as "quadratic" as possible. For example, using ELPA with 16 MPI tasks the setup (MPI-rows, MPI-columns)

- 4,4

works best, the setups

- 8,2
- 2,8
- 16,1
- 1,16

do work, but with less optimal performance. In case you do have the free choice of the number of MPI-tasks which you want to use, try to use a setup which can be split up in a "quadratic" way. If this is not possible, you might want to use less MPI tasks in ELPA than in your calling application and try the internal redistribution of ELPA.



