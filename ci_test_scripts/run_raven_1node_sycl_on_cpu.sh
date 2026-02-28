#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./ELPA_CI_2gpu.out.%j
#SBATCH -e ./ELPA_CI_2gpu.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J ELPA_CI
#
#SBATCH --ntasks-per-node=72  # Launch 72 tasks per node
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=02:00:00

set -eo pipefail

module purge
module load git autoconf automake libtool
module load intel/2025.2 mkl/2025.2 impi/2021.16 gcc/14


runner_path=$(pwd)

#export I_MPI_DEBUG=5

export LD_LIBRARY_PATH=$(pwd)/.libs:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH="${INTEL_HOME}/compiler/latest/linux/compiler/lib/intel64_lin"


#export LD_LIBRARY_PATH="${INTEL_HOME}/tbb/2021.7.1/env/../lib/intel64/gcc4.8:${INTEL_HOME}/compiler/2022.2.1/linux/lib:${INTEL_HOME}/compiler/2022.2.1/linux/lib/x64:${INTEL_HOME}/compiler/2022.2.1/linux/lib/oclfpga/host/linux64/lib:${INTEL_HOME}/compiler/2022.2.1/linux/compiler/lib/intel64_lin:${INTEL_HOME}/compiler/latest/linux/lib/x64/:${INTEL_HOME}/compiler/latest/linux/compiler/lib/intel64_lin:$LD_LIBRARY_PATH"

#export OCL_ICD_FILENAMES="${INTEL_HOME}/compiler/latest/linux/lib/x64/libintelocl.so"
#pwd

source ${INTEL_HOME}/setvars.sh

export LD_LIBRARY_PATH="${INTEL_HOME}/compiler/latest/lib":$LD_LIBRARY_PATH
export OCL_ICD_FILENAMES="${INTEL_HOME}/compiler/latest/lib/libintelocl.so"

export ELPA_EXTRA_gpu_sycl_backend=2 # SYCL Backend to use: 0 = Level Zero, 1 = OpenCL, 2 = all. ALL => Set ONEAPI_DEVICE_SELECTOR!",

export ONEAPI_DEVICE_SELECTOR="*:cpu"
sycl-ls

../configure --enable-option-checking=fatal FC=mpiifx CC=mpiicx CXX=mpiicpx CFLAGS="-O3 -g -xCORE-AVX512" CXXFLAGS="-O3 -g -xCORE-AVX512 -fsycl -I${INTEL_HOME}/compiler/latest/include -I${MKL_HOME}/include" FCFLAGS="-O3 -g -xCORE-AVX512" LIBS="-L${INTEL_HOME}/compiler/latest/lib -lsycl -Wl,-rpath,${INTEL_HOME}/compiler/latest/lib" SCALAPACK_FCFLAGS="-I${MKL_HOME}/include/intel64/lp64 -fsycl" SCALAPACK_LDFLAGS="-fsycl -Wc,-fsycl -L${MKL_HOME}/lib/intel64 -lmkl_sycl -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lsycl -lOpenCL -lpthread -lm -ldl -lirng -lstdc++ -Wl,-rpath,${MKL_HOME}/lib/intel64" --enable-ifx-compiler --enable-single-precision=no --with-mpi=yes --enable-avx512 --enable-intel-gpu-sycl-kernels --enable-intel-gpu-backend=sycl --disable-cpp-tests

make -j 72

#srun ./validate_real_double_eigenvectors_1stage_gpu_random 2000 2000 64
#srun ./validate_complex_double_eigenvectors_1stage_gpu_random 2000 2000 64
#srun ./validate_real_double_eigenvectors_2stage_default_kernel_gpu_random 2000 2000 64
#srun ./validate_complex_double_eigenvectors_2stage_default_kernel_gpu_random 2000 2000 64

make -j 36 check TEST_FLAGS="150 50 16"
