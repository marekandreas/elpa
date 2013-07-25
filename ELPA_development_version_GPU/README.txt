Experimental GPU Accelerated ELPA
----------------------------------

The directories contain an experimental GPU accelerated version
of ELPA for real-valued problems. 


1. Installation
---------------
The main makefile is test/Makefile, which will build both the test
applications and the library. The Makefile will require some configuration
for site specific locations of scalapack (and associated libraries), as well
as cublas. 

Selecting thex GPU or the CPU build of the library is controlled by
a preprocessor macro CPU_VERSION in the file src/elpa2.F90.


2. Running the example
----------------------
The test application to exercise the GPU branch is test/test_real2.f90. 
The application has been modified from the original version to accept
three command line arguments: 
- Matrix size (na)
- number of Eigenvectors (nev)
- Scalapack block size (nblk)

An example for invocation of this test under MPI is
mpirun -n 4 ./test_real2 10240 5120 128


3. Some notes on the implementation
-----------------------------------
In the current version, only the band reduction (bandred_real_gpu_v2.f90), 
and the tridiagonal to band backtransformation (ev_tridi_band_gpu_v3.f90)
have been GPU accelerated. In order to avoid a slowdown of the other parts 
of ELPA, it is recommended to run ELPA under proxy (ie have multiple
MPI ranks share the GPU).

Bandred_real mainly relies on CUBLAS for the GPU acceleration. CUDA Fortran
constructs are used to transfer the data between host and GPU. In order
to achieve overlap, these transfers are not using the CUDA Fortran array
asignments, but rather explicitly call cudaMemcpy(Async).

Most of the work in ev_tridi_band_gpu_v3 is focusing on the compute_hh_trafo
kernels. In addition to the CUDA Fortran version at the bottom of the file
ev_tridi_band_gpu_v3, there existis a C verion in ev_tridi_band_gpu_c.cu.
The reason for creating the C version was to take advantage of K20 features
like the __ldg and shfl instructions, which currently don't have any support
in CUDA Fortran. The different version of the kernel are selected in 
the compute_hh_trafo subroutine, around line 623 in ev_tridi_band_gpu_v3.



