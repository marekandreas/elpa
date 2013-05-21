#    This file is part of ELPA.
#
#    The ELPA library was originally created by the ELPA consortium, 
#    consisting of the following organizations:
#
#    - Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG), 
#    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
#      Informatik,
#    - Technische Universität München, Lehrstuhl für Informatik mit
#      Schwerpunkt Wissenschaftliches Rechnen , 
#    - Fritz-Haber-Institut, Berlin, Abt. Theorie, 
#    - Max-Plack-Institut für Mathematik in den Naturwissenschaftrn, 
#      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition, 
#      and  
#    - IBM Deutschland GmbH
#
#
#    More information can be found here:
#    http://elpa.rzg.mpg.de/
#
#    ELPA is free software: you can redistribute it and/or modify
#    it under the terms of the version 3 of the license of the 
#    GNU Lesser General Public License as published by the Free 
#    Software Foundation.
#
#    ELPA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
#
#    ELPA reflects a substantial effort on the part of the original
#    ELPA consortium, and we ask you to respect the spirit of the
#    license that we chose: i.e., please contribute any changes you
#    may have back to the original ELPA library distribution, and keep
#    any derivatives of ELPA under the same license that we chose for
#    the original distribution, the GNU Lesser General Public License.
#
#
# ------------------------------------------------------------------------------
# Please set the variables below according to your system!
# ------------------------------------------------------------------------------
# Settings for Intel Fortran (Linux), Intel Composer XE 2011 (ifort 12.1) with AVX for Sandy Bridge:
##
X86=1
F90=mpiifort -O3 -traceback -fpe0 -g -mavx
F90OPT=$(F90)
#CC=icc -O3
#CCOPT=$(CC) -mavx
CC=gcc -O3
CCOPT=$(CC) -mavx -funsafe-loop-optimizations -funsafe-math-optimizations -ftree-vect-loop-version -ftree-vectorize
LIBS = -mkl=sequential -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lstdc++
#
# ------------------------------------------------------------------------------
# Settings for Intel Fortran (Linux), Intel Composer XE 2011 (ifort 12.1) with SSE3:
#
#X86=1
#F90=mpiifort -O3 -traceback -fpe0 -g -msse3
#F90OPT=$(F90) -msse3
#CC=gcc -O3
#CCOPT=$(CC) -msse3 -funsafe-loop-optimizations -funsafe-math-optimizations -ftree-vect-loop-version -ftree-vectorize
##CC=icc -O3
##CCOPT=$(CC) -msse3
#LIBS = -mkl=sequential -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
#
# ------------------------------------------------------------------------------
# Settings for Intel Fortran (Linux), Intel Composer XE 2011 (ifort 12.1) and GCC 4.6 with FMA4 for AMD Bulldozer:
#
#X86=1
#F90=mpiifort -O3 -traceback -fpe0 -g -msse3
#CC=gcc -O3
#F90OPT=$(F90) -msse3
#CCOPT=$(CC) -funsafe-loop-optimizations -funsafe-math-optimizations -ftree-vect-loop-version -ftree-vectorize -mfma4 -mxop -march=bdver1 -D__USE_AVX128__
#CCOPT=$(CC) -funsafe-loop-optimizations -funsafe-math-optimizations -ftree-vect-loop-version -ftree-vectorize -mfma4 -mxop -march=bdver1
#LIBS = -L/opt/acml5.0.0/gfortran64_fma4/lib/ -lacml -lgfortran libscalapack.a
#LIBS = -L/lrz/sys/libraries/acml/5.2.0/ifort64_fma4_mp/lib -lacml_mp -lgfortran libscalapack.a
#
# ------------------------------------------------------------------------------
# Settings for Intel Fortran (Linux) old 11.x Toolchain, !!!!!!! do not use !!!!!!:
#
#X86=0
#F90=mpif90 -O3 -traceback -g -fpe0
#F90OPT=$(F90) -xSSE4.2
#LIBS = -L/opt/intel/Compiler/11.0/069/mkl/lib/em64t -lmkl_lapack -lmkl -lguide -lpthread \
#-lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
#
# ------------------------------------------------------------------------------
# Settings for Intel Fortran on MacOSX (home-built BLACS and scalapack):
#
#X86=0
#F90=mpif90 -O3 -traceback -g -fpe0
#F90OPT=$(F90) # -xSSE4.2 ### on Mac OSX, the -xSSE4.2 option is possibly buggy in ifort!
#LIBS = -L/opt/intel/mkl/lib -I/opt/intel/mkl/include -lmkl_intel_lp64 -lmkl_sequential -lmkl_core \
#   /usr/local/BLACS/LIB/blacs_MPI-OSX-0.a /usr/local/BLACS/LIB/blacsF77init_MPI-OSX-0.a \
#   /usr/local/SCALAPACK-1.8.0/libscalapack.a 
#
# ------------------------------------------------------------------------------
# Settings for IBM AIX Power6
#
#X86=0
#F90 = mpxlf95_r -q64 -O2 -g -qarch=auto -qtune=auto
#F90OPT = mpxlf95_r -q64 -O4 -g -qarch=auto -qtune=auto
#LIBS = -L/usr/local/lib -lscalapack -llapack-essl -lessl -lblacsF77init -lblacs -lblacsF77init -lblacs -lc
#
# ------------------------------------------------------------------------------
# Settings for IBM AIX BlueGene
#
#X86=0
#F90 = mpixlf95_r -O3 -g -qarch=auto -qtune=auto
#F90OPT = mpixlf95_r -O4 -g -qarch=auto -qtune=auto
#LIBS = -L/usr/local/lib -lscalapack -llapack -lblacsF77init -lblacs -lblacsF77init -lblacs \
-L/opt/ibmmath/essl/4.4/lib -lesslbg -lc
#
# ------------------------------------------------------------------------------

all: test_real read_real test_complex test_real_gen read_real_gen test_complex_gen test_real2 test_complex2

test_real: test_real.o elpa1.o
	$(F90) -o $@ test_real.o elpa1.o $(LIBS)

read_real: read_real.o elpa1.o
	$(F90) -o $@ read_real.o elpa1.o $(LIBS)

test_complex: test_complex.o elpa1.o
	$(F90) -o $@ test_complex.o elpa1.o $(LIBS)

test_real_gen: test_real_gen.o elpa1.o
	$(F90) -o $@ test_real_gen.o elpa1.o $(LIBS)

read_real_gen: read_real_gen.o elpa1.o
	$(F90) -o $@ read_real_gen.o elpa1.o $(LIBS)

test_complex_gen: test_complex_gen.o elpa1.o
	$(F90) -o $@ test_complex_gen.o elpa1.o $(LIBS)

ifeq ($(X86),1)
test_real2: test_real2.o elpa1.o elpa2.o elpa2_tum_kernels_complex_sse-avx_1hv.o elpa2_tum_kernels_complex_sse-avx_2hv.o elpa2_tum_kernels_real_sse-avx_2hv.o elpa2_tum_kernels_real_sse-avx_4hv.o elpa2_tum_kernels_real_sse-avx_6hv.o
	$(F90) -o $@ test_real2.o elpa1.o elpa2.o elpa2_tum_kernels_complex_sse-avx_1hv.o elpa2_tum_kernels_complex_sse-avx_2hv.o elpa2_tum_kernels_real_sse-avx_2hv.o elpa2_tum_kernels_real_sse-avx_4hv.o elpa2_tum_kernels_real_sse-avx_6hv.o $(LIBS)

test_complex2: test_complex2.o elpa1.o elpa2.o elpa2_tum_kernels_complex_sse-avx_1hv.o elpa2_tum_kernels_complex_sse-avx_2hv.o elpa2_tum_kernels_real_sse-avx_2hv.o elpa2_tum_kernels_real_sse-avx_4hv.o elpa2_tum_kernels_real_sse-avx_6hv.o
	$(F90) -o $@ test_complex2.o elpa1.o elpa2.o elpa2_tum_kernels_complex_sse-avx_1hv.o elpa2_tum_kernels_complex_sse-avx_2hv.o elpa2_tum_kernels_real_sse-avx_2hv.o elpa2_tum_kernels_real_sse-avx_4hv.o elpa2_tum_kernels_real_sse-avx_6hv.o $(LIBS)

#test_real2: test_real2.o elpa1.o elpa2.o elpa2_kernels_complex.o elpa2_tum_kernels_real_sse-avx_2hv.o elpa2_tum_kernels_real_sse-avx_4hv.o elpa2_tum_kernels_real_sse-avx_6hv.o
#	$(F90) -o $@ test_real2.o elpa1.o elpa2.o elpa2_kernels_complex.o elpa2_tum_kernels_real_sse-avx_2hv.o elpa2_tum_kernels_real_sse-avx_4hv.o elpa2_tum_kernels_real_sse-avx_6hv.o $(LIBS)
#
#test_complex2: test_complex2.o elpa1.o elpa2.o elpa2_kernels_complex.o elpa2_tum_kernels_real_sse-avx_2hv.o elpa2_tum_kernels_real_sse-avx_4hv.o elpa2_tum_kernels_real_sse-avx_6hv.o
#	$(F90) -o $@ test_complex2.o elpa1.o elpa2.o elpa2_kernels_complex.o elpa2_tum_kernels_real_sse-avx_2hv.o elpa2_tum_kernels_real_sse-avx_4hv.o elpa2_tum_kernels_real_sse-avx_6hv.o $(LIBS)
else
test_real2: test_real2.o elpa1.o elpa2.o elpa2_kernels_real.o elpa2_kernels_complex.o
	$(F90) -o $@ test_real2.o elpa1.o elpa2.o elpa2_kernels_real.o elpa2_kernels_complex.o $(LIBS)

test_complex2: test_complex2.o elpa1.o elpa2.o elpa2_kernels_real.o elpa2_kernels_complex.o
	$(F90) -o $@ test_complex2.o elpa1.o elpa2.o elpa2_kernels_real.o elpa2_kernels_complex.o $(LIBS)
endif

test_real.o: test_real.f90 elpa1.o
	$(F90) -c $<

read_real.o: read_real.f90 elpa1.o
	$(F90) -c $<

test_complex.o: test_complex.f90 elpa1.o
	$(F90) -c $<

test_real_gen.o: test_real_gen.f90 elpa1.o
	$(F90) -c $<

read_real_gen.o: read_real_gen.f90 elpa1.o
	$(F90) -c $<

test_complex_gen.o: test_complex_gen.f90 elpa1.o
	$(F90) -c $<

test_real2.o: test_real2.f90 elpa1.o elpa2.o
	$(F90) -c $<

test_complex2.o: test_complex2.f90 elpa1.o elpa2.o
	$(F90) -c $<

elpa1.o: ../src/elpa1.f90
	$(F90) -c $<

elpa2.o: ../src/elpa2.f90 elpa1.o
	$(F90) -c ../src/elpa2.f90

ifeq ($(X86),1)	
elpa2_tum_kernels_real_sse-avx_2hv.o: ../src/elpa2_kernels/elpa2_tum_kernels_real_sse-avx_2hv.c
	$(CCOPT) -c ../src/elpa2_kernels/elpa2_tum_kernels_real_sse-avx_2hv.c
	
elpa2_tum_kernels_real_sse-avx_4hv.o: ../src/elpa2_kernels/elpa2_tum_kernels_real_sse-avx_4hv.c
	$(CCOPT) -c ../src/elpa2_kernels/elpa2_tum_kernels_real_sse-avx_4hv.c
	
elpa2_tum_kernels_real_sse-avx_6hv.o: ../src/elpa2_kernels/elpa2_tum_kernels_real_sse-avx_6hv.c
	$(CCOPT) -c ../src/elpa2_kernels/elpa2_tum_kernels_real_sse-avx_6hv.c
	
elpa2_tum_kernels_complex_sse-avx_1hv.o: ../src/elpa2_kernels/elpa2_tum_kernels_complex_sse-avx_1hv.cpp
	$(CCOPT) -c ../src/elpa2_kernels/elpa2_tum_kernels_complex_sse-avx_1hv.cpp
	
elpa2_tum_kernels_complex_sse-avx_2hv.o: ../src/elpa2_kernels/elpa2_tum_kernels_complex_sse-avx_2hv.cpp
	$(CCOPT) -c ../src/elpa2_kernels/elpa2_tum_kernels_complex_sse-avx_2hv.cpp
	
elpa2_kernels_complex.o: ../src/elpa2_kernels/elpa2_kernels_complex.f90
	$(F90OPT) -c ../src/elpa2_kernels/elpa2_kernels_complex.f90
else
elpa2_kernels_real.o: ../src/elpa2_kernels/elpa2_kernels_real.f90
	$(F90OPT) -c ../src/elpa2_kernels/elpa2_kernels_real.f90
	
elpa2_kernels_complex.o: ../src/elpa2_kernels/elpa2_kernels_complex.f90
	$(F90OPT) -c ../src/elpa2_kernels/elpa2_kernels_complex.f90
endif

clean:
	rm -f *.o *.mod test_real test_complex test_real_gen test_complex_gen test_real2 test_complex2 read_real read_real_gen 
