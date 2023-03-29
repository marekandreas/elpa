#! /bin/bash

arch=$1
mat_size=$2
num_evs=$2
numbers=$3
num_ranks=$4

build_folder="build_$arch"

if [[ $arch == "pvc" ]]
then
  module load autoconf/2.71 intel-comp-rt/ci-neo-master/025928 intel-nightly/20230310 intel/mkl-nda/nightly-cev-20230314 intel/mpi/2021.8.0
  export LIBOMPTARGET_LEVEL_ZERO_USE_IMMEDIATE_COMMAND_LIST=all
  export LIBOMPTARGET_LEVEL_ZERO_INTEROP_USE_IMMEDIATE_COMMAND_LIST=1
  export LIBOMPTARGET_LEVEL0_USE_COPY_ENGINE=main
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=0
  export SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0:0
  export I_MPI_OFFLOAD=1
  export I_MPI_OFFLOAD_TOPOLIB=level_zero
elif [[ $arch == "a100" ]]
then
  module load autoconf/2.71 intel/oneapi/2023.0.0 gnu/10.3.0
  module load nvidia/cuda-12.0
elif [[ $arch == "h100" ]]
then
  echo "Loading modules for H100"
  module load autoconf/2.71 intel/oneapi/2023.0.0 gnu/10.3.0
  module load nvidia/cuda-12.0
elif [[ $arch == "icx" ]]
then
  module load autoconf/2.71 intel-nightly/20230310 intel/mkl-nda/nightly-cev-20230314 intel/mpi/2021.8.0
else
  echo "Unknown Architecture: $1"
  exit 1
fi

cd $build_folder

echo $CUDA_HOME
echo $LD_LIBRARY_PATH

mpirun -n $num_ranks ./validate_${numbers}_double_eigenvectors_2stage_default_kernel_gpu_random $mat_size $num_evs 64 | tee elpa_${numbers}_${arch}_${num_ranks}_${mat_size}.txt
