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

# ------------------------------------------------------------------------------
# Please set the variables below according to your system!
# ------------------------------------------------------------------------------
# Settings for Intel Fortran (Linux):
#
F90=mpif90 -O3 -traceback -g -fpe0
F90OPT=$(F90) -xSSE4.2
LIBS = -L/opt/intel/Compiler/11.0/069/mkl/lib/em64t -lmkl_lapack -lmkl -lguide -lpthread \
-lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
#
# ------------------------------------------------------------------------------
# Settings for Intel Fortran on MacOSX (home-built BLACS and scalapack):
#
#F90=mpif90 -O3 -traceback -g -fpe0
#F90OPT=$(F90) # -xSSE4.2 ### on Mac OSX, the -xSSE4.2 option is possibly buggy in ifort!
#LIBS = -L/opt/intel/mkl/lib -I/opt/intel/mkl/include -lmkl_intel_lp64 -lmkl_sequential -lmkl_core \
#   /usr/local/BLACS/LIB/blacs_MPI-OSX-0.a /usr/local/BLACS/LIB/blacsF77init_MPI-OSX-0.a \
#   /usr/local/SCALAPACK-1.8.0/libscalapack.a 
#
# ------------------------------------------------------------------------------
# Settings for IBM AIX Power6
#
#F90 = mpxlf95_r -q64 -O2 -g -qarch=auto -qtune=auto
#F90OPT = mpxlf95_r -q64 -O4 -g -qarch=auto -qtune=auto
#LIBS = -L/usr/local/lib -lscalapack -llapack-essl -lessl -lblacsF77init -lblacs -lblacsF77init -lblacs -lc
#
# ------------------------------------------------------------------------------
# Settings for IBM BlueGene/P
#
#F90 = mpixlf95_r -O3 -g -qarch=auto -qtune=auto
#F90OPT = mpixlf95_r -O4 -g -qarch=auto -qtune=auto
#LIBS = -L/usr/local/lib -lscalapack -llapack -lblacsF77init -lblacs -lblacsF77init -lblacs \
#-L/opt/ibmmath/essl/4.4/lib -lesslbg -lc
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

test_real2: test_real2.o elpa1.o elpa2.o elpa2_kernels.o
	$(F90) -o $@ test_real2.o elpa1.o elpa2.o elpa2_kernels.o $(LIBS)

test_complex2: test_complex2.o elpa1.o elpa2.o elpa2_kernels.o
	$(F90) -o $@ test_complex2.o elpa1.o elpa2.o elpa2_kernels.o $(LIBS)

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

elpa2_kernels.o: ../src/elpa2_kernels.f90
	$(F90OPT) -c ../src/elpa2_kernels.f90

clean:
	rm -f *.o *.mod test_real test_complex test_real_gen test_complex_gen test_real2 test_complex2 read_real read_real_gen
