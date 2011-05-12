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
# Settings for IBM AIX BlueGene
#
#F90 = mpixlf95_r -O3 -g -qarch=auto -qtune=auto
#F90OPT = mpixlf95_r -O4 -g -qarch=auto -qtune=auto
#LIBS = -L/usr/local/lib -lscalapack -llapack -lblacsF77init -lblacs -lblacsF77init -lblacs \
#-L/opt/ibmmath/essl/4.4/lib -lesslbg -lc
#
# ------------------------------------------------------------------------------

all: test_real test_complex test_real_gen test_complex_gen test_real2 test_complex2

test_real: test_real.o elpa1.o
	$(F90) -o $@ test_real.o elpa1.o $(LIBS)

test_complex: test_complex.o elpa1.o
	$(F90) -o $@ test_complex.o elpa1.o $(LIBS)

test_real_gen: test_real_gen.o elpa1.o
	$(F90) -o $@ test_real_gen.o elpa1.o $(LIBS)

test_complex_gen: test_complex_gen.o elpa1.o
	$(F90) -o $@ test_complex_gen.o elpa1.o $(LIBS)

test_real2: test_real2.o elpa1.o elpa2.o elpa2_kernels.o
	$(F90) -o $@ test_real2.o elpa1.o elpa2.o elpa2_kernels.o $(LIBS)

test_complex2: test_complex2.o elpa1.o elpa2.o elpa2_kernels.o
	$(F90) -o $@ test_complex2.o elpa1.o elpa2.o elpa2_kernels.o $(LIBS)

test_real.o: test_real.f90 elpa1.o
	$(F90) -c $<

test_complex.o: test_complex.f90 elpa1.o
	$(F90) -c $<

test_real_gen.o: test_real_gen.f90 elpa1.o
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
	rm -f *.o *.mod test_real test_complex test_real_gen test_complex_gen test_real2 test_complex2
