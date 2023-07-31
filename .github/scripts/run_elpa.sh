#! /bin/bash

arch=$1
mat_size=$2
num_evs=$2
numbers=$3
num_ranks=$4

build_folder=build_${numbers}_${arch}_${num_ranks}_${num_evs}

if [[ $arch == "pvc" ]] || [[ $arch == "pvc-1100" ]]
then
  module load autoconf/2.71 intel-comp-rt/ci-neo-master intel-nightly intel/mkl-nda intel/mpi
  export LIBOMPTARGET_LEVEL_ZERO_USE_IMMEDIATE_COMMAND_LIST=all
  export LIBOMPTARGET_LEVEL_ZERO_INTEROP_USE_IMMEDIATE_COMMAND_LIST=1
  export LIBOMPTARGET_LEVEL0_USE_COPY_ENGINE=main
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=0
  export SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0:0
  export I_MPI_OFFLOAD=1
  export I_MPI_OFFLOAD_TOPOLIB=level_zero
elif [[ $arch == "a100" ]] || [[ $arch == "h100" ]]
then
  # Only run on a single GPU
  export CUDA_VISIBLE_DEVICES=0
  module load autoconf/2.71 intel/oneapi/2023.2.0 gnu/10.3.0
  module load nvidia/cuda-12.0
elif [[ $arch == "spr" ]]
then
  module load autoconf/2.71 intel-nightly intel/mkl-nda intel/mpi
else
  echo "Unknown Architecture: $1"
  exit 1
fi

cd $build_folder

echo $CUDA_HOME
echo $LD_LIBRARY_PATH

mkdir -p ../cpa_results
mpirun -n $num_ranks ./validate_${numbers}_double_eigenvectors_2stage_default_kernel_gpu_random $mat_size $num_evs 64 | tee ../cpa_results/elpa_${numbers}_${arch}_${num_ranks}_${mat_size}.txt
