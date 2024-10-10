# Installation guide for the *ELPA* library #

## 0. Preamble ##

This file provides documentation on how to build the *ELPA* library in **version ELPA-2024.05.001**.
With release of **version ELPA-2017.05.001** the build process has been significantly simplified,
which makes it easier to install the *ELPA* library.

As anounced, with the the release 2021.11.001 the **legacy interface has been removed**.

The release of ELPA 2024.05.001 does not change the API and ABI compared to the release 2022.11.001.

## 1. How to install *ELPA* ##

First of all, if you do not want to build *ELPA* yourself, and you run Linux,
it is worth having a look at the [*ELPA* webpage](http://elpa.mpcdf.mpg.de)
and/or the repositories of your Linux distribution: there exist
pre-build packages for a number of Linux distributions like Fedora,
Debian, and OpenSuse. More, will hopefully follow in the future.

If you want to build (or have to since no packages are available) *ELPA* yourself,
please note that *ELPA* is shipped with a typical `configure` and `make`
autotools procedure. This is the **only supported way** how to build and install *ELPA*.


If you obtained *ELPA* from the official git repository, you will not find
the needed configure script! You will have to create the configure script with autoconf. You can also run the `autogen.sh` script that does this step for you.


## 2. Installing *ELPA* from source ##

*ELPA* can be installed with the build steps
- `configure`
- `make`
- `make check`   | or `make check CHECK_LEVEL=extended`
- `make install`

Please look at `configure --help` for all available options.

An excerpt of the most important (*ELPA* specific) options reads as follows:

| configure option                     | description                                           |
|:------------------------------------ |:----------------------------------------------------- |
|  `--enable-optional-argument-in-C-API`         | treat error arguments in C-API as optional            |
|  `--enable-openmp`                             | use OpenMP threading, default no.                     |
|  `--enable-redirect`                           | for ELPA test programs, allow redirection of <br> stdout/stderr per MPI taks in a file <br> (useful for timing), default no. |
|  `--enable-single-precision`                   | build with single precision version                   |
|  `--disable-timings`                           | more detailed timing, default yes <br> **If disabled some features like autotune will <br> not work anymmore !** |
|  `--disable-band-to-full-blocking`             | build ELPA2 with blocking in band_to_full <br> (default:enabled) |
|  `--disable-mpi-module`                        | do not use the Fortran MPI module, <br> get interfaces by 'include "mpif.h') |
|  `--disable-generic`                           | do not build GENERIC kernels, default: enabled        |
|  `--enable-sparc64`                            | do not build SPARC64 kernels, default: disabled        |
|  `--disable-sse`                               | do not build SSE kernels, default: enabled            |
|  `--disable-sse-assembly`                      | do not build SSE_ASSEMBLY kernels, default: enabled   |
|  `--disable-avx`                               | do not build AVX kernels, default: enabled            |
|  `--disable-avx2`                              | do not build AVX2 kernels, default: enabled           |
|  `--enable-avx512`                             | build AVX512 kernels, default: disabled               |
|  `--enable-sve128`                             | Experimental feature build ARM SVE128 kernels, default: disabled               |
|  `--enable-sve256`                             | Experimental feature build ARM SVE256 kernels, default: disabled               |
|  `--enable-sve512`                             | Experimental feature build ARM SVE512 kernels, default: disabled               |
|  `--enable-nvidia-gpu`                         | build NVIDIA GPU kernels, default: disabled           |
|  `--enable-gpu`                                | same as --enable-nvidia-gpu                           |
|  `--enable-amd-gpu`                            | EXPERIMENTAL: build AMD GPU kernels, default: disabled           |
|  `--enable-intel-gpu`                          | VERY EXPERIMENTAL: build INTEL GPU kernels, default: disabled           |
|  `--enable-bgp`                                | build BGP kernels, default: disabled                  |
|  `--enable-bgq`                                | build BGQ kernels, default: disabled                  |
|  `--with-mpi=[yes|no]`                         | compile with MPI. Default: yes                        |
|  `--with-cuda-path=PATH`                       | prefix where CUDA is installed [default=auto]         |
|  `--with-cuda-sdk-path=PATH`                   | prefix where CUDA SDK is installed [default=auto]     |
|  `--with-NVIDIA-GPU-compute-capability=VALUE`  | use compute capability VALUE for GPU version, <br> default: "sm_35" |
|  `--with-fixed-real-kernel=KERNEL`             | compile with only a single specific real kernel.      |
|  `--with-fixed-complex-kernel=KERNEL`          | compile with only a single specific complex kernel.   |
|  `--with-nvidia-gpu-support-only`              | Compile and always use the NVIDIA GPU version         |
|  `--with-amd-gpu-support-only`                 | EXPERIMENTAL: Compile and always use the AMD GPU version         |
|  `--with-intel-gpu-support-only`               | EXPERIMENTAL: Compile and always use the INTEL GPU version         |
|  `--with-likwid=[yes|no|PATH]`                 | use the likwid tool to measure performance (has an performance impact!), default: no |
|  `--with-default-real-kernel=KERNEL`           | set the real kernel KERNEL as default                 |
|  `--with-default-complex-kernel=KERNEL`        | set the compplex kernel KERNEL as default             |
|  `--enable-scalapack-tests`                    | build SCALAPACK test cases for performance <br> omparison, needs MPI, default no. |
|  `--enable-autotune-redistribute-matrix`       | EXPERIMENTAL FEATURE; NOT FULLY SUPPORTED YET: Allows ELPA during autotuning to re-distribute the matrix to find the best (ELPA internal) block size for block-cyclic distribution (Needs Scalapack functionality |
|  `--enable-autotuning`                         | enables autotuning functionality, default yes         |
|  `--enable-c-tests`                            | enables the C tests for elpa, default yes             |
|  `--disable-assumed-size`                      | do NOT use assumed-size Fortran arrays. default use   |
|  `--enable-scalapack-tests`                    | build also ScalaPack tests for performance comparison; needs MPI |
|  `--disable-Fortran2008-features`              | disable Fortran 2008 if compiler does not support it  |
|  `--enable-pyhton`                             | build and install python wrapper, default no          |
|  `--enable-python-tests`                       | enable python tests, default no.                      |
|  `--enable-skew-symmetric-support`             | enable support for real valued skew-symmetric matrices |
|  `--enable-store-build-config`                 | stores the build config in the library object |
|  `--64bit-integer-math-support`                | assumes that BLAS/LAPACK/SCALAPACK use 64bit integers (experimentatl) |
|  `--64bit-integer-mpi-support`                 | assumes that MPI uses 64bit integers (experimental) |
|  `--heterogenous-cluster-support`              | allows ELPA to run on clusters of nodes with different Intel CPUs (experimental) |
|  `--enable-allow-thread-limiting`              | in case of MPI and OPENMP builds, ELPA is allowed to limit the number of OpenMP threads, if the threading support level is not sufficient (default: on) |
|  `--with-threading-support-check-during-build` | during configure run a test to check which level of threading your MPI library does support (default: on) |
|  `--disable-runtime-threading-support-checks`  | in case of MPI and OpenMP ELPA disable runtime checks of the threading level of the supported MPI-libray (default: on) |


We recommend that you do not build ELPA in its main directory but that you use it
in a sub-directory:

```
mkdir build
cd build

../configure [with all options needed for your system, see below]
```

In this way, you have a clean separation between original *ELPA* source files and the compiled
object files

Please note, that it is necessary to set the **compiler options** like optimisation flags etc.
for the Fortran and C part.
For example sth. like this is a usual way: `./configure FCFLAGS="-O2 -mavx" CFLAGS="-O2 -mavx"`
For details, please have a look at the documentation for the compilers of your choice.

**Note** that most kernels can only be build if the correct compiler flags for this kernel (e.g. AVX-512)
have been enabled.


### 2.1 Choice of building with or without MPI ###

It is possible to build the *ELPA* library with or without MPI support.

Normally *ELPA* is build with MPI, in order to speed-up calculations by using distributed
parallelisation over several nodes. This is, however, only reasonably if the programs
calling the *ELPA* library are already MPI parallelized, and *ELPA* can use the same
block-cyclic distribution of data as in the calling program. **This is the main use case *ELPA* is being developed for**.

Programs which do not support MPI parallelisation can still make use of the *ELPA* library if it
has also been build without MPI support. This might be suitable for development purposes, or if you have an application which
should benefit from the *ELPA* eigensolvers, even if your application does not support MPI. **In order to achieve a reasonable performance
*ELPA*, however, should be then either build with OpenMP threading support or be used (if GPUs are available) with GPU support.**

If you want to build *ELPA* with MPI support, please have a look at "2.1.1 Setting of MPI compiler and libraries".
For builds without MPI support, please have a look at "2.1.2 Building *ELPA* without MPI support".
**NOTE** that if *ELPA* is build without MPI support, it will be serial unless the OpenMP parallelization is
explicitely enabled or GPU support is enabled.

Please note, that it is absolutely supported that both versions of the *ELPA* library are build
and installed in the same directory.

#### 2.1.1 Setting of MPI compiler and libraries ####

In the standard case *ELPA* needs a MPI compiler and MPI libraries. The configure script
will try to set this by itself. If, however, on the build system the compiler wrapper
cannot automatically found, it is recommended to set it by hand with a variable, e.g.

```
configure FC=mpif90
```

In some cases, on your system different MPI libraries and compilers are installed. Then it might happen
that during the build step an error like "no module mpi" or "cannot open module mpi" is given.
You can disable that the  *ELPA* library uses a MPI modules (and instead uses MPI header files) by
adding

```
--disable-mpi-module
```

to the configure call.

Please continue reading at "2.2 Enabling GPU support"


#### 2.1.2 Building *ELPA* without MPI support ####

If you want to build *ELPA* without MPI support, add

```
--with-mpi=no
```

to your configure call.

You have to specify which compilers should be used with e.g.,

```
configure FC=gfortran --with-mpi=no
```

**DO NOT specify a MPI compiler here!**

Note, that the installed *ELPA* library files will be suffixed with
`_onenode`, in order to discriminate this build from possible ones with MPI.


Please continue reading at "2.2 Enabling GPU support"

### 2.2 Enabling GPU support ###

The *ELPA* library can be build with GPU support. If *ELPA* is build with GPU
support, users can choose at RUNTIME, whether to use the GPU version or not.

For GPU support, NVIDIA GPUs with compute capability >= 3.5 are needed.

GPU support is set with

```
--enable-nvidia-gpu
```

It might be necessary to also set the options (please see configure --help)

```
--with-cuda-path
--with-cuda-sdk-path
--with-GPU-compute-capability
```

Please note that with release 2021.11.001 also GPU support of AMD and Intel GPUS has been introduced.
However, this is still considered experimental. Especially the following features do not yet work, or have not
been tested.

AMD GPUs:
- multi-GPU runs on _one_ node have been tested (with MPI)
- multi-GPU runs on _mutliple_ nodes have not been tested

Intel GPUs:
```
--enable-intel-gpu=[openmp|sycl]
```
- ELPA 2-stage optimized Intel GPU kernels are still missing.


Please continue reading at "2.2.1 Enabling OpenMP support".


#### 2.2.1 Enabling OpenMP support ####

The *ELPA* library can be build with OpenMP support. This can be support of hybrid
MPI/OpenMP parallelization, since *ELPA* is build with MPI support (see A ) or only
shared-memory parallization, since *ELPA* is build without MPI support (see B).

To enable OpenMP support, add

--enable-openmp

as configure option.

In any case, whether you are building MPI+OpenMP or only OpenMP (without MPI) it is recommended (for performance reasons)
to use BLAS and LAPACK libraries which _also_ do have threading support. For example, you can link against with the Intel MKL
library in the flavor without threading or with threading. Please consult the documentation of your BLAS and LAPACK libraries.


If you want to build a hybrid version of *ELPA* with MPI and with OpenMP support, your
MPI library **should** provide a sufficient level of threading support (i.e. "MPI_THREAD_SERIALIZED" or
"MPI_THREAD_MULTIPLE"). On HPC systems this is almost always the case. In many MPI packages available with
Linux distributions, however, the threading support is quite often limited and **not** sufficient for *ELPA*.
Since release 2021.05.001 ELPA does check during the build time in the configure if the threading support of
your MPI library is sufficient. This option (--enable-threading-support-checks) is enabled by default. **DO NOT
SWITCH THIS OFF, UNLESS YOU KNOW WHAT YOU ARE DOING OR UNLESS CONFIGURE INSTRUCTS YOU TO DO SO.**

If this test passes without an abort of configure and *no* instructions how to cure a threading level support issue,
your hybrid MPI-OpenMP build will be fine and for performance reasons **runtime** checks for threading support will
be disabled.

In the case that configure aborts with these messages
```
configure: WARNING: Your MPI implementation does not provide a sufficient threading level for OpenMP
configure: WARNING: You do have several options:
configure: WARNING:  * disable OpenMP (--disable-openmp): this will ensure correct results, but maybe some performance drop
configure: WARNING:  * use an MPI-library with the required threading support level (see the INSTALL and USER_GUIDE): this will 
configure: WARNING:    ensure correct results and best performance
configure: WARNING:  * allow ELPA at runtime to change the number of threads to 1 by setting "--enable-runtime-threading-support-checks
configure: WARNING:     --enable-allow-thread-limiting --without-threading-support-check-during-build": this will ensure correct results, but 
configure: WARNING:     maybe not the best performance (depends on the threading of your blas/lapack libraries), see the USER_GUIDE
configure: WARNING:  * switch of the checking of threading support "--disable-runtime-threading-support-checks 
configure: WARNING:    --without-threading-support-check-during-build: DO THIS AT YOUR OWN RISK! This will be fast, but might
configure: WARNING:    (depending on your MPI library sometimes) lead to wrong results
configure: error: You do have to take an action of the choices above!
```

your MPI library does _not_ provide a sufficient level of threading support. 
You can continue with buildig *ELPA* in several ways:

- disable OpenMP (by setting --disable-openmp): this will ensure (if the build is successful) that the results of *ELPA* will be correct. However, this option might
  lead to a performance drop, if your application calling *ELPA* does use OpenMP threading, since in this situation you will have less MPI tasks than cores on your machine and *ELPA* will only use MPI and thus not utilize all cores.

- the best solution will be to use an MPI library which does offer the required level of threading support. However, this _might_ require some work on your side to build your own MPI library. In any way it does not harm to search the internet whether for your Linux distribution their exist already such MPI packages (quite often they do but they are not the default onces).

- if you do not want to disable OpenMP and you cannot provide a MPI library with a sufficient level of threading support, you can re-run configure with the options "--enable-allow-thread-limiting --without-threading-support-check-during-build". If these options are enabled, *ELPA* will skip the test during the configure step, but will always do a **runtime** check whether the MPI library does provide a sufficient level of threading support. If this is not the case, **internally** to *ELPA* (i.e. not affecting your application calling *ELPA*) only **1** OpenMP thread will be used. In case you do use a threaded implementation of BLAS and LAPACK (which performance wise you should always do when using an OpenMP build of *ELPA*), one can still use more than one thread within the BLAS and LAPACK library, **if** the number of threads in these libraries can be controlled with another mechanism then setting the **OMP_NUM_THREADS** environment variable. For example, in case of Intel's MKL library one can controll the number of threads with the MKL_NUM_THREADS environment variable.

- by switching of the threading level support checks **both at build and runtime**. This can be achieved by calling configure with --disable-runtime-threading-support-checks   --without-threading-support-check-during-build" In this case *ELPA* **assumes** that your MPI library does provide a sufficient level. **DO NOT USE THIS OPTION UNLESS YOU ARE SURE WHAT YOU ARE DOING !** This setting could cause different problems like crashes, sporadic wrong results and so forth since this will lead to undefined behaviour! The *ELPA* developers **will not accept bug reports if this option is used, unless you can document in a detailed way that you first know what you are doing and second can proof that this option did not create the bug you would like to report (see below)!** You might wonder why this option is then at all available. Simply, because some very experienced HPC-experts did ask for this option because of a the situation we will discuss now.

Last but not least we want to mention that prior to executing this check, configure will print this information:
```
configure: **************************************************************************************************************************
configure: * Please notice if the following step hangs or aborts abnormaly then you cannot run a short MPI-program during configure *
configure: * In this case please re-run configure with '--without-threading-support-check-during-build' _AND_ follow the hints in   *
configure: * the INSTALL and USER_GUIDE documents!                                                                                  *
configure: * In case you get some other warnings about threading support follow on of the steps detailed there                      *
configure: **************************************************************************************************************************
```
You do not have to care about this, unless configure hangs after printing this message, or configure aborts **without** printing the error messages 
```
configure: error: You do have to take an action of the choices above!
```
as discussed before.

This behaviour might occure, if:
- you (and also configure) does not have the rights to run an MPI program on the compilation machine. Sometimes HPC centers implement this, in order to ensure that login nodes are only used for compilation but not for compute.
- you do have to cross-compile (i.e. you build *ELPA* for a specific architecture on a different architecture)
- some other reason why an MPI program cannot run successfully 

If you encounter this situation you can switch of this check during configure by setting "--without-threading-support-check-during-build". However, of course *ELPA* cannot know then whether your MPI library does provide a sufficient level of threading support or not. Thus you will have to tell configure what to do by either
- also setting "--enable-allow-thread-limiting" (see above)
- or setting "--disable-runtime-threading-support-checks" (see above, **especially the warnings**)
We recommend the following procedure: 

- in a first step you set "--with-threading-support-check-during-build=no" and "--enable-allow-thread-limiting". 
- after the successful build you do run *ELPA* on the target machine with the environment variable "OMP_NUM_THREADS" set to 2. 
- Now, carefully inspect the output (stdout **and** stderr). If *ELPA* does not give a warning that it will limit the number of OpenMP threads to 1 due to an insufficent level of threading support in the MPI library, you can assume that your MPI library does provide a sufficient level. 
- Then **and only then** you can rebuild *ELPA* with the settings "--without-threading-support-check-during-build" and "--disable-runtime-threading-support-checks".


Note that as in case with/without MPI, you can also build and install versions of *ELPA*
with/without OpenMP support at the same time.

However, the GPU choice at runtime is not compatible with OpenMP support.

Please continue reading at "2.3 Standard libraries in default installation paths".


### 2.3 Standard libraries in default installation paths ###

In order to build the *ELPA* library, some (depending on the settings during the
configure step) libraries are needed.

Typically these are:
  - Basic Linear Algebra Subroutines (BLAS)                   (always needed)
  - Lapack routines                                           (always needed)
  - Basic Linear Algebra Communication Subroutines (BLACS)    (only needed if MPI support was set)
  - Scalapack routines                                        (only needed if MPI support was set)
  - a working MPI library                                     (only needed if MPI support was set)
  - a working OpenMP library                                  (only needed if OpenMP support was set)
  - a working CUDA/cublas library                             (only needed if GPU support was set)

If the needed library are installed on the build system in standard paths (e.g. /usr/lib64)
in the most cases the *ELPA* configure step will recognize the needed libraries
automatically. No setting of any library paths should be necessary.

If your configure steps finish succcessfully, please continue at "2.5 Choice of ELPA2 compute kernels".
If your configure step aborts, or you want to use libraries in non standard paths please continue at
"2.4 Non standard paths or non standard libraries".

### 2.4 Non standard paths or non standard libraries ###

If standard libraries are on the build system either installed in non standard paths, or
special non standard libraries (e.g. *Intel's MKL*) should be used, it might be necessary
to specify the appropriate link-line with the **SCALAPACK_LDFLAGS** and **SCALAPACK_FCFLAGS**
variables.

For example, due to performance reasons it might be benefical to use the *BLAS*, *BLACS*, *LAPACK*,
and *SCALAPACK* implementation from *Intel's MKL* library.

Together with the Intel Fortran Compiler the call to configure might then look like:

```
configure SCALAPACK_LDFLAGS="-L$MKL_HOME/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential \
                             -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -Wl,-rpath,$MKL_HOME/lib/intel64" \
	  SCALAPACK_FCFLAGS="-L$MKL_HOME/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential \
	                      -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -I$MKL_HOME/include/intel64/lp64"
```

and for *INTEL MKL* together with *GNU GFORTRAN* :

```
configure SCALAPACK_LDFLAGS="-L$MKL_HOME/lib/intel64 -lmkl_scalapack_lp64 -lmkl_gf_lp64 -lmkl_sequential \
                             -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -Wl,-rpath,$MKL_HOME/lib/intel64" \
	  SCALAPACK_FCFLAGS="-L$MKL_HOME/lib/intel64 -lmkl_scalapack_lp64 -lmkl_gf_lp64 -lmkl_sequential \
	                     -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -I$MKL_HOME/include/intel64/lp64"
```

Please, for the correct link-line refer to the documentation of the correspondig library. In case of *Intel's MKL* we
suggest the [Intel Math Kernel Library Link Line Advisor](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor).


### 2.5 Choice of ELPA2 compute kernels ###

ELPA 2stage can be used with different implementations of compute intensive kernels, which are architecture dependent.
Some kernels (all for x86_64 architectures) are enabled by default (and must be disabled if you do not want them),
others are disabled by default and must be enabled if they are wanted.

One can enable "kernel classes" by setting e.g.

```
--enable-avx2 
```

This will try to build all the AVX2 kernels. Please see configure --help for all options

With

```
--disable-avx2
```

one chan choose not to build the AVX2 kernels.


During the configure step all possible kernels will be printed, and whether they will be enabled or not.

It is possible to build *ELPA* with as many kernels as desired, the user can then choose at runtime which
kernels should be used.

It this is not desired, it is possible to build *ELPA* with only one (not necessary the same) kernel for the
real and complex valued case, respectively. This can be done with the `--with-fixed-real-kernel=NAME` or
`--with-fixed-complex-kernel=NAME` configure options. For details please do a "configure --help"

#### 2.5.1 Cross compilation ####

The ELPA library does _not_ supports cross-compilation by itself, i.e. compilation of the ELPA library on an architecture wich is not
identical than the architecture ELPA should be used on.

Whenever a cross-compilation situation might occur, great care has to be taken during the build process by the user.

At the moment we see two potential pitfalls:

1.) The "build architecure" is inferior to the "target" architecture (w.r.t. the instructions sets)

In this case, at the moment, the ELPA library can only be build with instructions sets supported on the build
system. All later instruction sets will _not_ be used in the compilation. This case might lead to less optimal
performance compared to the case that ELPA is build directly on the target system.

For example, if the "build architecture" consists of an HASWELL node (supporting up to Intel's AVX2 instruction set) and the 
"target architecture" is a Skylake node (supporting Intel's AVX-512 instruction set) than the AVX-512 kernels can not be build
This will lead to a performance degradation on the Skylake nodes, but is otherwise harmless (no chrashes).


2.) The "build architecure" is superior to the "target" architecture (w.r.t. the instructions sets)

This case is a critical one, since ELPA will by default build with instructions sets which are not supported on the target
system. This will lead to crashes, if during build the user does not take care to solve this issue.

For example, if the "build architecture" supports Intels' AVX-2 instruction set and the 
"target architecture" does only support Intel's AVX instruction set, then by default ELPA will be build with AVX-2 instruction set
and this will also be used at runtime (since it improves the performance). However, at the moment, since the target system does not support
AVX-2 instructions this will lead to a crash.

One can avoid this unfortunate situation by disabling instructions set which are _not_ supported on the target system.
In the case above, setting

```
--disable-avx2
```

during build, will remdy this problem.


### 2.6 Doxygen documentation ###
A doxygen documentation can be created with the `--enable-doxygen-doc` configure option

### 2.7 Some examples ###

#### 2.7.1 Intel cores supporting AVX2 (Haswell and newer) ####

It is possible to build ELPA with the Intel compiler (if available) for the Fortran part, but
with GNU compiler for the C part.

1. Building with Intel Fortran compiler and GNU C compiler:

Remarks:
  - you have to know the name of the Intel Fortran compiler wrapper
  - you do not have to specify a C compiler (with CC); GNU C compiler is recognized automatically
  - you should specify compiler flags for Intel Fortran compiler; in the example only `-O3 -xAVX2` is set
  - you should be careful with the CFLAGS, the example shows typical flags

```
FC=mpi_wrapper_for_intel_Fortran_compiler CC=mpi_wrapper_for_gnu_C_compiler ./configure FCFLAGS="-O3 -xAVX2" CFLAGS="-O3 -march=native -mavx2 -mfma -funsafe-loop-optimizations -funsafe-math-optimizations -ftree-vect-loop-version -ftree-vectorize" --enable-option-checking=fatal SCALAPACK_LDFLAGS="-L$MKLROOT/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread " SCALAPACK_FCFLAGS="-I$MKL_HOME/include/intel64/lp64"
```

2. Building with GNU Fortran compiler and GNU C compiler:

Remarks: 
  - you have to know the name of the GNU Fortran compiler wrapper
  - you DO have to specify a C compiler (with CC); GNU C compiler is recognized automatically
  - you should specify compiler flags for GNU Fortran compiler; in the example only `-O3 -march=native -mavx2 -mfma` is set
  - you should be careful with the CFLAGS, the example shows typical flags

```
FC=mpi_wrapper_for_gnu_Fortran_compiler CC=mpi_wrapper_for_gnu_C_compiler ./configure FCFLAGS="-O3 -march=native -mavx2 -mfma" CFLAGS="-O3 -march=native -mavx2 -mfma  -funsafe-loop-optimizations -funsafe-math-optimizations -ftree-vect-loop-version -ftree-vectorize" --enable-option-checking=fatal SCALAPACK_LDFLAGS="-L$MKLROOT/lib/intel64 -lmkl_scalapack_lp64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread " SCALAPACK_FCFLAGS="-I$MKL_HOME/include/intel64/lp64"
```

3. Building with Intel Fortran compiler and Intel C compiler:

Remarks:
  - you have to know the name of the Intel Fortran compiler wrapper
  - you have to specify the Intel C compiler
  - you should specify compiler flags for Intel Fortran compiler; in the example only "-O3 -xAVX2" is set
  - you should be careful with the CFLAGS, the example shows typical flags

```
FC=mpi_wrapper_for_intel_Fortran_compiler CC=mpi_wrapper_for_intel_C_compiler ./configure FCFLAGS="-O3 -xAVX2" CFLAGS="-O3 -xAVX2" --enable-option-checking=fatal SCALAPACK_LDFLAGS="-L$MKLROOT/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread " SCALAPACK_FCFLAGS="-I$MKL_HOME/include/intel64/lp64"
```

#### 2.7.2 Intel cores supporting AVX-512 (Skylake and newer) ####

It is possible to build ELPA with the Intel compiler (if available) for the Fortran part, but
with GNU compiler for the C part.

1. Building with Intel Fortran compiler and GNU C compiler:

Remarks:
  - you have to know the name of the Intel Fortran compiler wrapper
  - you do not have to specify a C compiler (with CC); GNU C compiler is recognized automatically
  - you should specify compiler flags for Intel Fortran compiler; in the example only `-O3 -xCORE-AVX512` is set
  - you should be careful with the CFLAGS, the example shows typical flags

```
FC=mpi_wrapper_for_intel_Fortran_compiler CC=mpi_wrapper_for_gnu_C_compiler ./configure FCFLAGS="-O3 -xCORE-AVX512" CFLAGS="-O3 -march=skylake-avx512 -mfma -funsafe-loop-optimizations -funsafe-math-optimizations -ftree-vect-loop-version -ftree-vectorize" --enable-option-checking=fatal SCALAPACK_LDFLAGS="-L$MKLROOT/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread " SCALAPACK_FCFLAGS="-I$MKL_HOME/include/intel64/lp64" --enable-avx2 --enable-avx512
```

2. Building with GNU Fortran compiler and GNU C compiler:

Remarks: 
  - you have to know the name of the GNU Fortran compiler wrapper
  - you DO have to specify a C compiler (with CC); GNU C compiler is recognized automatically
  - you should specify compiler flags for GNU Fortran compiler; in the example only `-O3 -march=skylake-avx512 -mfma` is set
  - you should be careful with the CFLAGS, the example shows typical flags

```
FC=mpi_wrapper_for_gnu_Fortran_compiler CC=mpi_wrapper_for_gnu_C_compiler ./configure FCFLAGS="-O3 -march=skylake-avx512 -mfma" CFLAGS="-O3 -march=skylake-avx512 -mfma  -funsafe-loop-optimizations -funsafe-math-optimizations -ftree-vect-loop-version -ftree-vectorize" --enable-option-checking=fatal SCALAPACK_LDFLAGS="-L$MKLROOT/lib/intel64 -lmkl_scalapack_lp64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread " SCALAPACK_FCFLAGS="-I$MKL_HOME/include/intel64/lp64" --enable-avx2 --enable-avx512
```

3. Building with Intel Fortran compiler and Intel C compiler:

Remarks:
  - you have to know the name of the Intel Fortran compiler wrapper
  - you have to specify the Intel C compiler
  - you should specify compiler flags for Intel Fortran compiler; in the example only "-O3 -xCORE-AVX512" is set
  - you should be careful with the CFLAGS, the example shows typical flags

```
FC=mpi_wrapper_for_intel_Fortran_compiler CC=mpi_wrapper_for_intel_C_compiler ./configure FCFLAGS="-O3 -xCORE-AVX512" CFLAGS="-O3 -xCORE-AVX512" --enable-option-checking=fatal SCALAPACK_LDFLAGS="-L$MKLROOT/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread " SCALAPACK_FCFLAGS="-I$MKL_HOME/include/intel64/lp64" --enable-avx2 --enable-avx512
```


#### 2.7.3 Building for NVIDIA A100 GPUS (and Intel Icelake CPUs) ####

For the GPU builds of ELPA it is mandatory that you choose a GNU compiler for the C part, the Fortran part can be compiled with any compiler, for example with the Intel Fortran compiler

1. Building with Intel Fortran compiler and GNU C compiler:

Remarks:
  - you have to know the name of the Intel Fortran compiler wrapper
  - you do not have to specify a C compiler (with CC); GNU C compiler is recognized automatically
  - you should specify compiler flags for Intel Fortran compiler; in the example only `-O3 -xCORE-AVX512` is set
  - you should be careful with the CFLAGS, the example shows typical flags

```
FC=mpi_wrapper_for_intel_Fortran_compiler CC=mpi_wrapper_for_gnu_C_compiler ./configure FCFLAGS="-O3 -xCORE-AVX512" CFLAGS="-O3 -march=skylake-avx512 -mfma -funsafe-loop-optimizations -funsafe-math-optimizations -ftree-vect-loop-version -ftree-vectorize" --enable-option-checking=fatal SCALAPACK_LDFLAGS="-L$MKLROOT/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread " SCALAPACK_FCFLAGS="-I$MKL_HOME/include/intel64/lp64" --enable-avx2 --enable-avx512 --enable-nvidia-gpu --with-cuda-path=PATH_TO_YOUR_CUDA_INSTALLATION --with-NVIDIA-GPU-compute-capability=sm_80
```

2. Building with GNU Fortran compiler and GNU C compiler:

Remarks: 
  - you have to know the name of the GNU Fortran compiler wrapper
  - you DO have to specify a C compiler (with CC); GNU C compiler is recognized automatically
  - you should specify compiler flags for GNU Fortran compiler; in the example only `-O3 -march=skylake-avx512 -mfma` is set
  - you should be careful with the CFLAGS, the example shows typical flags

```
FC=mpi_wrapper_for_gnu_Fortran_compiler CC=mpi_wrapper_for_gnu_C_compiler ./configure FCFLAGS="-O3 -march=skylake-avx512 -mfma" CFLAGS="-O3 -march=skylake-avx512 -mfma  -funsafe-loop-optimizations -funsafe-math-optimizations -ftree-vect-loop-version -ftree-vectorize" --enable-option-checking=fatal SCALAPACK_LDFLAGS="-L$MKLROOT/lib/intel64 -lmkl_scalapack_lp64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread " SCALAPACK_FCFLAGS="-I$MKL_HOME/include/intel64/lp64" --enable-avx2 --enable-avx512 --enable-nvidia-gpu --with-cuda-path=PATH_TO_YOUR_CUDA_INSTALLATION --with-NVIDIA-GPU-compute-capability=sm_80
```

#### 2.7.4 Building for IBM SUMMIT HPC system ####

For more information please have  a look at the [ELSI wiki](https://git.elsi-interchange.org/elsi-devel/elsi-interface/-/wikis/install-elpa).

1. Building with GNU Fortran compiler and GNU C compiler:

```
FC=mpif90 CC=mpicc ./configure --prefix=$(pwd) CFLAGS="-O2 -mcpu=power9" CFLAGS="-O2 -mcpu=power9" CPP="cpp -E" LDFLAGS="-L${OLCF_NETLIB_SCALAPACK_ROOT}/lib -lscalapack -L${OLCF_ESSL_ROOT}/lib64 -lessl -L${OLCF_NETLIB_LAPACK_ROOT}/lib64 -llapack" --enable-gpu --with-cuda-path=${OLCF_CUDA_ROOT} --with-GPU-compute-capability=sm_70 --disable-sse-assembly --disable-sse --disable-avx --disable-avx2 --disable-avx512 --enable-c-tests=no
```


2. Building with PGI Fortran compiler and PGI C compiler:

```
FC=mpif90 CC=mpicc ./configure --prefix=$(pwd) CFLAGS="-fast -tp=pwr9" CFLAGS="-fast -tp=pwr9" CPP="cpp -E" LDFLAGS="-L${OLCF_NETLIB_SCALAPACK_ROOT}/lib -lscalapack -L${OLCF_ESSL_ROOT}/lib64 -lessl -L${OLCF_NETLIB_LAPACK_ROOT}/lib64 -llapack" --enable-gpu --with-cuda-path=${OLCF_CUDA_ROOT} --with-GPU-compute-capability=sm_70 --disable-sse-assembly --disable-sse --disable-avx --disable-avx2 --disable-avx512 --enable-c-tests=no
```

3. Building with IBM Fortran compiler and IBM C compiler:

```
FC=mpixlf CC=mpixlc ../configure --prefix=$(pwd) FCFLAGS="-O2 -qarch=pwr9 -qstrict -WF,-qfpp=linecont" CFLAGS="-O2 -qarch=pwr9 -qstrict" CPP="cpp -E" LDFLAGS="-L${OLCF_NETLIB_SCALAPACK_ROOT}/lib -lscalapack -L${OLCF_ESSL_ROOT}/lib64 -lessl -L${OLCF_NETLIB_LAPACK_ROOT}/lib64 -llapack" --enable-gpu --with-cuda-path=${OLCF_CUDA_ROOT} --with-GPU-compute-capability=sm_70 --disable-sse-assembly --disable-sse --disable-avx --disable-avx2 --disable-avx512 --enable-c-tests=no
```


#### 2.7.5 EXPERIMENTAL: Building for AMD GPUs (currently tested only --with-mpi=0 ####


In order to build *ELPA* for AMD GPUs please ensure that you have a working installation of HIP, ROCm, BLAS, and LAPACK

```
./configure CXX=hipcc CXXFLAGS="-I/opt/rocm-4.0.0/hip/include/ -I/opt/rocm-4.0.0/rocblas/inlcude -g" CC=hipcc CFLAGS="-I/opt/rocm-4.0.0/hip/include/ -I/opt/rocm-4.0.0/rocblas/include -g" LIBS="-L/opt/rocm-4.0.0/rocblas/lib" --enable-option-checking=fatal --with-mpi=0 FC=gfortran FCFLAGS="-g -LPATH_TO_YOUR_LAPACK_INSTALLATION -lopenblas -llapack" --disable-sse --disable-sse-assembly --disable-avx --disable-avx2 --disable-avx512 --enable-AMD-gpu --enable-single-precision
```

#### 2.7.8 Problems of building with clang-12.0 ####
The libtool tool adds some flags to the compiler commands (to be used for linking by ld) which are not known
by the clang-12 compiler. One way to solve this issue is by calling directly after the configue step
```
sed -i 's/\\$wl-soname \\$wl\\$soname/-fuse-ld=ld -Wl,-soname,\\$soname/g' libtool
sed -i 's/\\$wl--whole-archive\\$convenience \\$wl--no-whole-archive//g' libtool
```
