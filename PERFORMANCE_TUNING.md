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
#### Builds with OpenMP enabled ####
If you enable OpenMP support in your ELPA build -- independent wheter MPI is enabled or disabled -- please ensure that you link against a BLAS and LAPACK library which does offer threading support. If you link with libraries which do not offer support for threading then you will observe a severe performance loss.

#### Builds for NVIDIA GPU support ####
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

do work, but with less optimal performance. Especially, setups which allow only for one row (or column) in the 2D MPI grid do result in less than optimal performance.
This is illustrated in the figure below where we show the run-time for the solution of a 20k matrix, with the number of MPI processes varying from 2 to 40. Please not that setups which enforce one process row (or process column), since the total number of MPI tasks is a prime number should always be avoided.

In case you do have the free choice of the number of MPI-tasks which you want to use, try to use a setup which can be split up in a "quadratic" way. If this is not possible, you might want to use less MPI tasks within ELPA than in your calling application and try the internal redistribution of ELPA to a new process grid.

#### Hybrid MPI-OpenMP runs ####
For the optimal performance of hybrid MPI-OpenMP runs with the  *ELPA*-library, it is mandatory that you do not overbook the node with a combination of MPI tasks and OpenMP threads. Also, disable "nested OpenMP" and ensure that your threaded BLAS and LAPACK libraries do use more than one thread. Last but not least, please check that on your system the appropriate pinning of the MPI tasks and the OpenMP threads per tasks is ensured. Thus please, keep an eye on
- the number of MPI tasks * OpenMP threads <= number of cores per node
- set the number of OpenMP threads by setting the OMP_NUM_THREADS variable
- set the number of threads in the BLAS and LAPACK library (for Intel's MKL set MKL_NUM_THREADS to a value larger 1)
- check the pinning of MPI tasks and OpenMP threads, but do not pin to hyperthreads


#### GPU runs ####
If you want to use the GPU version of ELPA, please ensure that the same number of MPI tasks is mapped to each GPU in the node. If this cannot be achieved, then do not fully occupy the node with all MPI tasks. For example on a hypothetical node with 34 cores and 3 GPUs, do use only 33 MPI tasks per node and map always 11 MPI tasks to each GPU. Furthermore, if you have (the very common situation) with more than 1 MPI task per GPU, the performance will be improved quite dramatically if you ensure that the NVIDIA MPS daemon is running on each node. Please make sure that only one MPS daemon is started per node. For more details please also have a look at [this](https://www.sciencedirect.com/science/article/abs/pii/S0010465520304021) publication.

