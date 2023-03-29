#! /bin/bash

if [[ $1 == "pvc" ]]
then
  module load autoconf/2.71 intel-comp-rt/ci-neo-master/025928 intel-nightly/20230310 intel/mkl-nda/nightly-cev-20230314 intel/mpi/2021.8.0
elif [[ $1 == "a100" ]]
then
  module load autoconf/2.71 nvidia/cuda-12.0 intel/oneapi/2023.0.0 gnu/10.3.0
elif [[ $1 == "h100" ]]
then
  module load autoconf/2.71 nvidia/cuda-12.0 intel/oneapi/2023.0.0 gnu/10.3.0
elif [[ $1 == "icx" ]]
then
  module load autoconf/2.71 intel-nightly/20230310 intel/mkl-nda/nightly-cev-20230314 intel/mpi/2021.8.0
else
  echo "Unknown Architecture: $1"
  exit 1
fi

mkdir -p build_$1
cd build_$1
../autogen.sh

if [[ $1 == "pvc" ]]
then
  ../configure CC="mpiicc -cc=icx" CXX="mpiicpc -cxx=icpx" FC="mpiifort -fc=ifx" \
    CXXFLAGS="-g -O3 -march=skylake-avx512 -I$(dirname $(dirname $(which icpx)))/linux/include/sycl -fsycl-targets=spir64 -fsycl -qopenmp" \
    CFLAGS="-g -g -O3 -march=skylake-avx512 -qopenmp" \
    FCFLAGS="-g -O3 -march=skylake-avx512 -fsycl -qopenmp -nostandard-realloc-lhs -align array64byte -I$(dirname $(dirname $(which icpx)))/linux/include/sycl" \
    LDFLAGS="-lsycl -lOpenCL -lpthread -lstdc++" \
    SCALAPACK_FCFLAGS="-I$MKLROOT/lib/intel64/lp64 -fsycl -rdynamic" \
    SCALAPACK_LDFLAGS="-fsycl -L$MKLROOT/lib/intel64 -Wc,-fsycl -lmkl_sycl -lmkl_intel_ilp64 -lmkl_scalapack_ilp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_ilp64 -lsycl -lOpenCL -lpthread -lm -ldl -lirng -lstdc++" \
    --disable-static --enable-sse --enable-sse-assembly --enable-avx --enable-avx2 --enable-avx512 --enable-single-precision --enable-ifx-compiler \
    --disable-c-tests --without-threading-support-check-during-build --enable-intel-gpu-backend=sycl --enable-64bit-integer-math-support --enable-intel-gpu-sycl
elif [[ $1 == "h100" ]] || [[ $1 == "a100" ]]
then
  if [[$1 == "h100"]]
  then
    compute_capability="sm_90"
  else
    compute_capability="sm_80"
  fi
  echo "building with compute capability $compute_capability"
  ../configure CC="mpiicc -cc=icx" FC=mpiifort CPP="gcc -E" \
CFLAGS="-O3 -march=skylake-avx512" FCFLAGS="-O3 -xCORE-AVX512" \
SCALAPACK_FCFLAGS="-I$MKLROOT/intel64/lp64" \
SCALAPACK_LDFLAGS="-L$MKLROOT/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -Wl,-rpath,$MKLROOT/lib/intel64 -lstdc++" \
--disable-openmp --disable-64bit-integer-math-support --disable-64bit-integer-mpi-support --enable-mpi-module --enable-detect-mpi-launcher --enable-generic --disable-sparc64 --disable-neon-arch64 --disable-vsx --disable-sse --disable-sse-assembly --disable-avx --disable-avx2 --disable-avx512 --disable-sve128 --disable-sve256 --disable-sve512 --disable-bgp --disable-bgp --enable-assumed-size --disable-ifx-compiler --enable-Fortran2008-features --enable-option-checking=fatal --disable-heterogenous-cluster-support --enable-timings --enable-band-to-full-blocking --without-threading-support-check-during-build --disable-runtime-threading-support-checks --disable-allow-thread-limiting --disable-gpu --enable-nvidia-gpu --disable-amd-gpu --disable-intel-gpu-sycl --disable-nvidia-sm80-gpu --disable-NVIDIA-gpu-memory-debug --disable-cuda-aware-mpi --disable-gpu-streams --disable-nvtx --disable-c-tests --enable-skew-symmetric-support --with-mpi=yes --disable-redirect --enable-single-precision --disable-autotuning --disable-scalapack-tests --disable-autotune-redistribute-matrix --with-papi=no --with-likwid=no --disable-store-build-config --disable-python --disable-python-tests --with-cuda-path="/opt/hpc_software/compilers/nvidia/cuda-12.0/" --with-NVIDIA-GPU-compute-capability=$compute_capability --with-cusolver
else
  echo "not implemented yet!"
  exit 1
fi

make -j32
