#!/bin/bash
#source /etc/profile.d/modules.sh
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh ; else source /etc/profile.d/mpcdf_modules.sh; fi
set -ex

source ./ci_test_scripts/.ci-env-vars
module list
ulimit -s unlimited
ulimit -v unlimited
export CHECK_LEVEL=extended

if [ "$(hostname)" != "miy01" -a "$(hostname)" != "miy02" -a "$(hostname)" != "miy03" ] ; then export I_MPI_STATS=10; fi
if [ "$(hostname)" != "miy01" -a "$(hostname)" != "miy02" -a "$(hostname)" != "miy03" ] ; then unset SLURM_MPI_TYPE I_MPI_SLURM_EXT I_MPI_PMI_LIBRARY I_MPI_PMI2 I_MPI_HYDRA_BOOTSTRAP; fi

export OMP_NUM_THREADS=$2

eval make check TASKS=$1 ${3} || { cat test-suite.log; exit 1; }

