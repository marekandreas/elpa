#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./ELPA_CI_gpu.out.%j
#SBATCH -e ./ELPA_CI_gpu.err.%j
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
module load git autoconf automake libtool
module load gcc/14 openmpi/5.0 mkl/2025.3 cuda/12.8 nccl/2.26.2

runner_path=$(pwd)

export LD_LIBRARY_PATH=$(pwd)/.libs:$LD_LIBRARY_PATH

../configure --enable-option-checking=fatal CC=mpicc FC=mpif90 CXX=mpicxx \
CFLAGS="-O3 -g -march=skylake-avx512 -I$MKL_HOME/include/intel64/lp64 -I$CUDA_HOME/include" \
CXXFLAGS="-std=c++17 -O3 -march=skylake-avx512 -I$MKL_HOME/include/intel64/lp64 -I$CUDA_HOME/include" \
FCFLAGS="-O3 -g -march=skylake-avx512 -I$MKL_HOME/include/intel64/lp64 -I$CUDA_HOME/include" \
LDFLAGS="-L$MKL_HOME/lib/intel64 -lmkl_scalapack_lp64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_openmpi_lp64 -lpthread -Wl,-rpath,$MKL_HOME/lib/intel64" \
 --with-mpi=yes --enable-band-to-full-blocking --disable-assumed-size --enable-avx512-kernels \
 --enable-nvidia-gpu-kernels --with-NVIDIA-GPU-compute-capability=sm_80 --with-cuda-path=$CUDA_HOME --enable-gpu-ccl=nccl --with-nccl-path=$NCCL_HOME NVCCFLAGS="-lineinfo" \
 --enable-c-tests=no --enable-cpp-tests=no --enable-single-precision=no --with-test-programs=all


make -j 4 validate_real_double_eigenvectors_2stage_default_kernel_gpu_random

# Test for small nev in ELPA2-GPU. Minimum 4 MPI processes needed here
srun -n 4 ./validate_real_double_eigenvectors_2stage_default_kernel_gpu_random 150 1 16
