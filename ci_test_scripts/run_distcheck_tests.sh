#!/bin/bash
set -e
set -x

#some defaults
mpiTasks=2
matrixSize=150
nrEV=$matrixSize
blockSize=16
ompThreads=1
configueArg=""
batchCommand=""
slurmBatch="no"
gpuJob="no"

function usage() {
	cat >&2 <<-EOF

		Call all the necessary steps to perform an ELPA CI test

		Usage:
		  run_distcheck_tests [-c configure arguments] [-d distcheck configure arguments] [-h] [-t MPI Tasks] [-m matrix size] [-n number of eigenvectors] [-b block size] [-o OpenMP threads] [-q submit command] [-S submit to Slurm] [-g GPU job]"

		Options:
		 -c configure arguments
		    Line of arguments passed to configure call

                 -d configure arguments to distcheck
		    Line of arguments passed to distcheck configure call

		 -t MPI Tasks
		    Number of MPI processes used during test runs of ELPA tests

		 -m Matrix size
		    Size of the mxm matrix used during runs of ELPA tests

		 -n Number of eigenvectors
		    Number of eigenvectors to be computed during runs of ELPA tests

		 -b block size
		    Block size of block-cyclic distribution during runs of ELPA tests

		 -o OpenMP threads
		    Number of OpenMP threads used during runs of ELPA tests

		 -q submit command
		    Job steps will be submitted via command to a batch system (default no submission)

		 -S submit to slurm
		    if "yes" a SLURM batch job will be submitted

		 -g gpu job
		    if "yes" a gpu job is assumed

		 -h
		    Print this help text
	EOF
}


while getopts "c:d:t:j:m:n:b:o:s:q:i:S:g:h" opt; do
	case $opt in
		t)
			mpiTasks=$OPTARG;;
		m)
			matrixSize=$OPTARG;;
		n)
			nrEV=$OPTARG;;
		b)
			blockSize=$OPTARG;;
		o)
			ompThreads=$OPTARG;;
		c)
			configureArgs=$OPTARG;;
		d)
			distcheckConfigureArgs=$OPTARG;;

		q)
			batchCommand=$OPTARG;;
		S)
			slurmBatch=$OPTARG;;
		g)
			gpuJob=$OPTARG;;
		:)
			echo "Option -$OPTARG requires an argument" >&2;;
		h)
			usage
			exit 1;;
		*)
			exit 1;;
	esac
done


if [ "$slurmBatch" == "yes" ]
then

  # default exit code
  exitCode=1
  CLUSTER=""
  if [[ "$HOST" =~ "raven" ]]
  then
    CLUSTER="raven"
  fi
  if [[ "$HOST" =~ "cobra" ]]
  then
    CLUSTER="cobra"
  fi
  if [[ "$HOST" =~ "talos" ]]
  then
    CLUSTER="talos"
  fi
  if [[ "$HOST" =~ "ada" ]]
  then
    CLUSTER="ada"
  fi
  if [[ "$HOST" =~ "draco" ]]
  then
    CLUSTER="draco"
  fi

  echo "Running on $CLUSTER with runner $CI_RUNNER_DESCRIPTION with tag $CI_RUNNER_TAGS on $mpiTasks tasks"

  #distcheck
  if [[ "$CI_RUNNER_TAGS" =~ "distcheck" ]]
  then
    
    # GPU runners
    if [ "$gpuJob" == "yes" ]
    then
      echo "GPU run"
      cp $HOME/runners/job_script_templates/run_${CLUSTER}_1node_2GPU.sh .
      echo "if \[ \$SLURM_PROCID -eq 0 \]" >> ./run_${CLUSTER}_1node_2GPU.sh
      echo "then" >> ./run_${CLUSTER}_1node_2GPU.sh
      echo "echo \"process \$SLURM_PROCID running configure\"" >> ./run_${CLUSTER}_1node_2GPU.sh
      echo "#decouple from SLURM (maybe this could be removed)" >> ./run_${CLUSTER}_1node_2GPU.sh
      echo "export _save_SLURM_MPI_TYPE=\$SLURM_MPI_TYPE" >> ./run_${CLUSTER}_1node_2GPU.sh
      echo "export _save_I_MPI_SLURM_EXT=\$I_MPI_SLURM_EXT"  >> ./run_${CLUSTER}_1node_2GPU.sh
      echo "export _save_I_MPI_PMI_LIBRARY=\$I_MPI_PMI_LIBRARY" >> ./run_${CLUSTER}_1node_2GPU.sh
      echo "export _save_I_MPI_PMI2=\$I_MPI_PMI2" >> ./run_${CLUSTER}_1node_2GPU.sh
      echo "export _save_I_MPI_HYDRA_BOOTSTRAP=\$I_MPI_HYDRA_BOOTSTRAP" >> ./run_${CLUSTER}_1node_2GPU.sh
      echo "unset SLURM_MPI_TYPE I_MPI_SLURM_EXT I_MPI_PMI_LIBRARY I_MPI_PMI2 I_MPI_HYDRA_BOOTSTRAP" >> ./run_${CLUSTER}_1node_2GPU.sh
      echo " " >> ./run_${CLUSTER}_1node_2GPU.sh
      echo "./configure " "$configureArgs" >> ./run_${CLUSTER}_1node_2GPU.sh
      echo " " >> ./run_${CLUSTER}_1node_2GPU.sh
      echo "fi" >> ./run_${CLUSTER}_1node_2GPU.sh
      echo " " >> ./run_${CLUSTER}_1node_2GPU.sh
      echo "export OMP_NUM_THREADS=$ompThreads" >> ./run_${CLUSTER}_1node_2GPU.sh
      echo "export TASKS=$mpiTasks" >> ./run_${CLUSTER}_1node_2GPU.sh
      echo "export DISTCHECK_CONFIGURE_FLAGS=\" $distcheckConfigureArgs \" "  >> ./run_${CLUSTER}_1node_2GPU.sh	
      echo "make distcheck TEST_FLAGS=\" $matrixSize $nrEV $blockSize \" || { chmod u+rwX -R . ; exit 1 ; } " >> ./run_${CLUSTER}_1node_2GPU.sh
      
      echo " " >> ./run_${CLUSTER}_1node_2GPU.sh	
      echo "exitCode=\$?" >> ./run_${CLUSTER}_1node_2GPU.sh	
      echo " " >> ./run_${CLUSTER}_1node_2GPU.sh	
      echo "#copy everything back from /tmp/elpa to runner directory" >> ./run_${CLUSTER}_1node_2GPU.sh	
      echo "#cp -r * \$runner_path" >> ./run_${CLUSTER}_1node_2GPU.sh	
      echo "#cd .. && rm -rf /tmp/elpa_\$SLURM_JOB_ID" >> ./run_${CLUSTER}_1node_2GPU.sh	
      echo "exit \$exitCode" >> ./run_${CLUSTER}_1node_2GPU.sh	
      echo " "	
      echo "Job script for the run"	
      cat ./run_${CLUSTER}_1node_2GPU.sh	
      echo " "	
      echo "Submitting to SLURM"	
      if sbatch -W ./run_${CLUSTER}_1node_2GPU.sh; then
        exitCode=$?
      else
        exitCode=$?
        echo "Submission exited with exitCode $exitCode"
      fi
      cat ./ELPA_CI_2gpu.out.*
      #if [ $exitCode -ne 0 ]	
      #then	
      cat ./ELPA_CI_2gpu.err.*	
      #fi
    
    # CPU runners    
    else
      echo "CPU run"  
      cp $HOME/runners/job_script_templates/run_${CLUSTER}_1node.sh .
      echo " " >> ./run_${CLUSTER}_1node.sh
      echo "if [ \$SLURM_PROCID -eq 0 ]" >> ./run_${CLUSTER}_1node.sh
      echo "then" >> ./run_${CLUSTER}_1node.sh
      echo "echo \"process \$SLURM_PROCID running configure\"" >> ./run_${CLUSTER}_1node.sh
      echo "#decouple from SLURM (maybe this could be removed)" >> ./run_${CLUSTER}_1node.sh
      echo "export _save_SLURM_MPI_TYPE=\$SLURM_MPI_TYPE" >> ./run_${CLUSTER}_1node.sh
      echo "export _save_I_MPI_SLURM_EXT=\$I_MPI_SLURM_EXT"  >> ./run_${CLUSTER}_1node.sh
      echo "export _save_I_MPI_PMI_LIBRARY=\$I_MPI_PMI_LIBRARY" >> ./run_${CLUSTER}_1node.sh
      echo "export _save_I_MPI_PMI2=\$I_MPI_PMI2" >> ./run_${CLUSTER}_1node.sh
      echo "export _save_I_MPI_HYDRA_BOOTSTRAP=\$I_MPI_HYDRA_BOOTSTRAP" >> ./run_${CLUSTER}_1node.sh
      echo "unset SLURM_MPI_TYPE I_MPI_SLURM_EXT I_MPI_PMI_LIBRARY I_MPI_PMI2 I_MPI_HYDRA_BOOTSTRAP" >> ./run_${CLUSTER}_1node.sh
      echo " " >> ./run_${CLUSTER}_1node.sh
      echo "./configure " "$configureArgs" " || { cat config.log; exit 1; }"  >> ./run_${CLUSTER}_1node.sh
      echo "fi" >> ./run_${CLUSTER}_1node.sh
      echo " " >> ./run_${CLUSTER}_1node.sh
      echo "export TASKS=$mpiTasks" >> ./run_${CLUSTER}_1node.sh
      echo "export DISTCHECK_CONFIGURE_FLAGS=\" $distcheckConfigureArgs \" "  >> ./run_${CLUSTER}_1node.sh
      echo "make distcheck TEST_FLAGS=\" $matrixSize $nrEV $blockSize \" || { chmod u+rwX -R . ; exit 1 ; } " >> ./run_${CLUSTER}_1node.sh
      
      echo " " >> ./run_${CLUSTER}_1node.sh
      echo "exitCode=\$?" >> ./run_${CLUSTER}_1node.sh
      echo " " >> ./run_${CLUSTER}_1node.sh
      echo "#copy everything back from /tmp/elpa to runner directory" >> ./run_${CLUSTER}_1node.sh
      echo "#cp -r * \$runner_path" >> ./run_${CLUSTER}_1node.sh
      echo "#cd .. && rm -rf /tmp/elpa_\$SLURM_JOB_ID" >> ./run_${CLUSTER}_1node.sh
      echo "exit \$exitCode" >> ./run_${CLUSTER}_1node.sh
      echo " "
      echo "Job script for the run"
      cat ./run_${CLUSTER}_1node.sh
      echo " "
      echo "Submitting to SLURM"
      if sbatch -W ./run_${CLUSTER}_1node.sh; then
        exitCode=$?
      else
        exitCode=$?
        echo "Submission exited with exitCode $exitCode"
      fi

      cat ./ELPA_CI.out.*
      #if [ $exitCode -ne 0 ]
      #then
      cat ./ELPA_CI.err.*
      #fi
    
    fi
  
  fi

  if [ -f ./test-suite.log ]
  then
    cat ./test-suite.log
  fi
  exit $exitCode

fi
