# Installation guide #

## Preamle ##

This file provides documentation on how to build the *ELPA* library in **version ELPA-2016.05.003**.
Although most of the documentation is generic to any *ELPA* release, some configure options
described in this document might be specific to the above mentioned version of *ELPA*.

## How to install ELPA ##

First of all, if you do not want to build *ELPA* yourself, and you run Linux,
it is worth having a look at the [*ELPA* webpage*] (http://elpa.mpcdf.mpg.de)
and/or the repositories of your Linux distribution: there exist
pre-build packages for a number of Linux distributions like Fedora,
Debian, and OpenSuse. More, will hopefully follow in the future.

If you want to build (or have to since no packages are available) *ELPA* yourself,
please note that *ELPA* is shipped with a typical "configure" and "make"
autotools procedure. This is the **only supported way** how to build and install *ELPA*.

If you obtained *ELPA* from the official git repository, you will not find
the needed configure script! Please look at the "**INSTALL_FROM_GIT_VERSION**" file
for the documentation how to proceed.


## (A): Installing ELPA as library with configure ##

*ELPA* can be installed with the build steps
- configure
- make
- make check
- make install

Please look at configure --help for all available options.

Please note, that it is necessary to set the **compiler options** like optimisation flags etc.
for the Fortran and C part.
For example sth. like this is a usual way ./configure FCFLAGS="-O2 -mavx" CFLAGS="-O2 -mavx"
For details, please have a look at the documentation for the compilers of your choice.

### Setting of MPI compiler and libraries ###

In the standard case *ELPA* need a MPI compiler and MPI libraries. The configure script
will try to set this by itself. If, however, on the build system the compiler wrapper
cannot automatically found, it is recommended to set it by hand with a variable, e.g.

configure FC=mpif90

### Hybrid MPI/OpenMP library build ###
The *ELPA* library can be build to support hybrid MPI/OpenMP support. To do this the
"--enable-openmp" configure option should be said. If also a hybrid version of *ELPA*
is wanted, it is recommended to build to version of *ELPA*: one with pure MPI and
a hybrid version. They can be both installed in the same path, since the have different
so library names.

### Standard libraries in default installation paths###

In order to build the *ELPA* library, some (depending on the settings during the
configure step, see below) libraries are needed.

Typically these are:
  - Basic Linear Algebra Subroutines (BLAS)
  - Lapack routines
  - Basic Linear Algebra Communication Subroutines (BLACS)
  - Scalapack routines
  - a working MPI library

If the needed library are installed on the build system in standard paths (e.g. /usr/lib64)
the in most cases the *ELPA* configure step will recognize the needed libraries
automatically. No setting of any library paths should be necessary.

### Non standard paths or non standard libraries ###

If standard libraries are on the build system either installed in non standard paths, or
special non standard libraries (e.g. *Intel's MKL*) should be used, it might be necessary
to specify the appropriate link-line with the **SCALAPACK_LDFLAGS** and **SCALAPACK_FCFLAGS** 
variables.

For example, due to performance reasons it might be benefical to use the *BLAS*, *BLACS*, *LAPACK*, 
and *SCALAPACK* implementation from *Intel's MKL* library.

Togehter with the Intel Fortran Compiler the call to configure might then look like:

configure SCALAPACK_LDFLAGS="-L$MKL_HOME/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential \
                             -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -Wl,-rpath,$MKL_HOME/lib/intel64" \
	  SCALAPACK_FCFLAGS="-L$MKL_HOME/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential \
	                      -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -I$MKL_HOME/include/intel64/lp64"

and for *INTEL MKL* togehter with *GNU GFORTRAN* :

configure SCALAPACK_LDFLAGS="-L$MKL_HOME/lib/intel64 -lmkl_scalapack_lp64 -lmkl_gf_lp64 -lmkl_sequential \
                             -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -Wl,-rpath,$MKL_HOME/lib/intel64" \
	  SCALAPACK_FCFLAGS="-L$MKL_HOME/lib/intel64 -lmkl_scalapack_lp64 -lmkl_gf_lp64 -lmkl_sequential \
	                     -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -I$MKL_HOME/include/intel64/lp64"


Please, for the correct link-line refer to the documentation of the correspondig library. In case of *Intel's MKL* we
sugest the [Intel Math Kernel Library Link Line Advisor] (https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor).


### Choice of ELPA2 compute kernels ###

In the default the configure script tries to configure and build all ELPA2 compute kernels which are available for
the architecture. Then the specific kernel can be chosen at run-time via the api or an environment variable (see
the **USERS_GUIDE** for details).

It this is not desired, it is possible to build *ELPA* with only one (not necessary the same) kernel for the
real and complex valued case, respectively. This can be done with the "--with-real-..-kernel-only" and
"--with-complex-..-kernel-only" configure options. For details please do a "configure --help"

### No MPI, one node shared-memory version of ELPA ###

Since release 2016.05.001 it is possible to build *ELPA* without any MPI support. This version can be used
by applications, which do not have any MPI parallelisation. To set this version, use the
"--with-mpi=0" configure flag. It is strongly recommmended to also set the "--enable-openmp"
option, otherwise no parallelisation whatsoever will be present.

It is possible to install the different flavours of ELPA (with/without MPI, with/without OpenMP) in the same
directory, since the library is named differently for each build.

### Doxygen documentation ###
A doxygen documentation can be created with the "--enable-doxygen-doc" configure option








