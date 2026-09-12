#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./ELPA_CI_gpu_aware_mpi.out.%j
#SBATCH -e ./ELPA_CI_gpu_aware_mpi.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J ELPA_CI
#
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --nvmps
#SBATCH --gres=gpu:a100:4
#SBATCH --mail-type=none
#SBATCH --time=02:00:00

set -eo pipefail

module purge
module load autoconf/2.71
module load gcc/13 cuda/12.6 mkl/2025.2
module load openmpi_gpu/5.0 openmpi_gpu_UCC_UCX/.5.0

export LD_LIBRARY_PATH="$PWD/.libs:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^/u/pekarp/local/lib$' | paste -sd: -)

export OMPI_MCA_coll_ucc_enable=1
export OMPI_MCA_coll_ucc_priority=100

../configure --enable-option-checking=fatal CC=mpicc FC=mpif90 CXX=mpicxx \
CFLAGS="-O3 -g -march=skylake-avx512 -I$MKL_HOME/include/intel64/lp64 -I$CUDA_HOME/include" \
CXXFLAGS="-std=c++17 -O3 -march=skylake-avx512 -I$MKL_HOME/include/intel64/lp64 -I$CUDA_HOME/include" \
FCFLAGS="-O3 -g -march=skylake-avx512 -I$MKL_HOME/include/intel64/lp64 -I$CUDA_HOME/include" \
LDFLAGS="-L$MKL_HOME/lib/intel64 -lmkl_scalapack_lp64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_openmpi_lp64 -lpthread -Wl,-rpath,$MKL_HOME/lib/intel64" \
 --with-mpi=yes --enable-band-to-full-blocking --enable-nvidia-gpu-kernels \
 --with-NVIDIA-GPU-compute-capability=sm_80 --with-cuda-path=$CUDA_HOME --enable-avx512-kernels --enable-nvtx \
 --enable-c-tests=no --enable-cpp-tests=no --enable-single-precision=no --disable-assumed-size \
 --enable-gpu-aware-mpi NVCCFLAGS="-O3" --with-test-programs=gpu

make -j 4
TASKS=4 make check TEST_FLAGS="100 50 16"
