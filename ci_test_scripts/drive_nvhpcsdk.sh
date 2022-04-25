#!/bin/bash
set -e
set -x

#some defaults
mpiTasks=$1
cp /u/elpa/runners/job_script_templates/run_raven_1node_4GPU_nvhpcsdk.sh .

perl -i -p -e "s/TASKS/$mpiTasks/g" ./run_raven_1node_4GPU_nvhpcsdk.sh

sbatch -W run_raven_1node_4GPU_nvhpcsdk.sh
exitCode=$?
if [ $exitCode -ne 0 ]; then exit 1; fi
exit 0;

