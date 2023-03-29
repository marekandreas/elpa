#! /bin/bash


# modules for Intel

if [[ $1 == "pvc" ]]
then
  module load autoconf/2.71 intel-comp-rt/ci-neo-master/025928 intel-nightly/20230310 intel/mkl-nda/nightly-cev-20230314 intel/mpi/2021.8.0
fi

mkdir -p build_$1
cd build
../autogen.sh

if [[ $1 == "pvc" ]]
then
  ../configure CC="mpiicc -cc=icx" CXX="mpiicpc -cxx=icpx" FC="mpiifort -fc=ifx" \
    CXXFLAGS="-g -O3-march=skylake-avx512 -I$(dirname $(dirname $(which icpx)))/linux/include/sycl -fsycl-targets=spir64 -fsycl -qopenmp" \
    CFLAGS="-g -g -O3-march=skylake-avx512 -qopenmp" \
    FCFLAGS="-g -O3-march=skylake-avx512-fsycl -qopenmp-nostandard-realloc-lhs -align array64byte-I$(dirname $(dirname $(which icpx)))/linux/include/sycl" \
    LDFLAGS="-lsycl -lOpenCL -lpthread -lstdc++" \
    SCALAPACK_FCFLAGS="-I$MKLROOT/lib/intel64/lp64 -fsycl -rdynamic" \
    SCALAPACK_LDFLAGS="-fsycl -L$MKLROOT/lib/intel64 -Wc,-fsycl -lmkl_sycl -lmkl_intel_ilp64 -lmkl_scalapack_ilp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_ilp64 -lsycl -lOpenCL -lpthread -lm -ldl -lirng -lstdc++" \
    --disable-static --enable-sse --enable-sse-assembly --enable-avx --enable-avx2 --enable-avx512 --enable-single-precision --enable-ifx-compiler \
    --disable-c-tests --without-threading-support-check-during-build --enable-intel-gpu-backend=sycl --enable-64bit-integer-math-support --enable-intel-gpu-sycl

fi

make -j32
