# Modules needed at RZG:
# module load impi/4.0.0 mkl/10.3 cuda/4.0 pgi/12.5
# ------------------------------------------------------------------------------
# Please set the variables below according to your system!
# ------------------------------------------------------------------------------
# Settings for Intel Fortran (Linux):
#
#F90=mpif90 -O3 -fast -ta=nvidia -Minfo=accel   -I/opt/intel/Compiler/11.1/069/mkl/include  -Mcuda -g
#F90=mpif90 -O3 -fast -Minfo=accel   -I/opt/intel/Compiler/11.1/069/mkl/include -Mcuda -ta=nvidia -g
#F90=mpif90 -O3 -fast -g -Minfo=accel   -I/afs/@cell/common/soft/intel/Compiler/11.1/c/mkl/include  -ta=nvidia,cuda4.0,cc20  -Mcuda 
#F90=ftn -O3 -fast -I$(I_MPI_ROOT)/intel64/include  -L$(I_MPI_ROOT)/intel64/lib  -ta=nvidia,cuda4.2,cc20 -Mcuda -Minfo=accel

PREF=

F90=$(PREF) mpif90 -g -O3 -fast -Minline -I/afs/@cell/common/soft/intel/ics/2011.0.013/12.1/mkl/include/intel64/lp64 -I//u/pmessmer/software/mpich/include  -ta=nvidia,cuda5.0,cc35,keepgpu,keepptx  -Mcuda=ptxinfo -Minfo=accel

FFLAGS = -O3 -fast
F90FLAGS = $(FFLAGS)
F90OPT = $(F90)
ARCHITECTURE = PGI
#LIBS =ev_tridi_band_gpu_c.o  -Mcuda -L/afs/@cell/common/soft/intel/ics/2011.0.013/12.1/mkl/lib/intel64/ \
  -I/afs/@cell/common/soft/intel/ics/2011.0.013/12.1/mkl/include/intel64/lp64 -lmkl_intel_lp64 -lmkl_intel_lp64 \
   -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm  -lcublas

#LIBS =ev_tridi_band_gpu_c.o  -Mcuda -L/afs/@cell/common/soft/intel/ics/2011.0.013/12.1/mkl/lib/intel64/ \
  -I/afs/@cell/common/soft/intel/ics/2011.0.013/12.1/mkl/include/intel64/lp64 -lmkl_intel_lp64 \
   -lmkl_sequential -lmkl_core -lpthread -L/afs/rzg/home/a/amarek/scalapack/scalapack-2.0.2 -lscalapack  -lcublas

LIBS =ev_tridi_band_gpu_c.o  -Mcuda -L/afs/@cell/common/soft/intel/ics/2011.0.013/12.1/mkl/lib/intel64/ \
  -I/afs/@cell/common/soft/intel/ics/2011.0.013/12.1/mkl/include/intel64/lp64  -lmkl_intel_lp64 -lmkl_intel_lp64 \
   -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64  -lpthread   -lcublas -L/u/pmessmer/software/mpich/lib





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

#all: test_real read_real test_complex test_real_gen read_real_gen test_complex_gen test_real2 test_complex2
all: test_real2

test_real: test_real.o elpa1.o 
	$(F90) -o $@ test_real.o elpa1.o $(LIBS)

read_real: read_real.o elpa1.o 
	$(F90) -o $@ read_real.o elpa1.o  $(LIBS)

test_complex: test_complex.o elpa1.o 
	$(F90) -o $@ test_complex.o elpa1.o  $(LIBS)

test_real_gen: test_real_gen.o elpa1.o 
	$(F90) -o $@ test_real_gen.o elpa1.o $(LIBS)

read_real_gen: read_real_gen.o elpa1.o 
	$(F90) -o $@ read_real_gen.o elpa1.o  $(LIBS)

test_complex_gen: test_complex_gen.o elpa1.o 
	$(F90) -o $@ test_complex_gen.o elpa1.o  $(LIBS)

test_real2: test_real2.o elpa1.o elpa2.o elpa2_kernels.o ev_tridi_band_gpu_c.o
	$(F90) -o $@ test_real2.o elpa1.o elpa2.o elpa_pdgeqrf.o elpa_pdlarfb.o elpa_qrkernels.o tum_utils.o elpa2_kernels.o $(LIBS)

test_complex2: test_complex2.o elpa1.o elpa2.o elpa2_kernels.o 
	$(F90) -o $@ test_complex2.o elpa1.o elpa2.o elpa_pdgeqrf.o elpa_pdlarfb.o elpa_qrkernels.o tum_utils.o elpa2_kernels.o  $(LIBS)

test_real.o: test_real.f90 elpa1.o 
	$(F90) -c $<

read_real.o: read_real.f90 elpa1.o
	$(F90) -c $<

tum_utils.o: ../src/elpa_qr/tum_utils.f90
	$(F90) -c $<

elpa_qrkernels.o: ../src/elpa_qr/elpa_qrkernels.f90
	$(F90) -c $<

elpa_pdlarfb.o: ../src/elpa_qr/elpa_pdlarfb.f90 tum_utils.o elpa_qrkernels.o
	$(F90) -c $<

elpa_pdgeqrf.o: ../src/elpa_qr/elpa_pdgeqrf.f90 elpa1.o tum_utils.o elpa_pdlarfb.o elpa_qrkernels.o
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

elpa2.o:	elpa1.o\
		elpa_pdgeqrf.o
	$(F90) -c ../src/elpa2.F90

elpa2_kernels.o: ../src/elpa2_kernels.f90
	$(F90OPT) -c ../src/elpa2_kernels.f90

ev_tridi_band_gpu_c.o: ../src/ev_tridi_band_gpu_c.cu
	nvcc -g -G -arch sm_35 -O -c --ptxas-options=-v ../src/ev_tridi_band_gpu_c.cu

clean:
	rm -f *.o *.mod test_real test_complex test_real_gen test_complex_gen test_real2 test_complex2 read_real read_real_gen


