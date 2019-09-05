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
		  run_distcheck_tests [-c configure arguments] [-h] [-t MPI Tasks] [-m matrix size] [-n number of eigenvectors] [-b block size] [-o OpenMP threads] [-q submit command] [-S submit to Slurm] [-g GPU job]"

		Options:
		 -c configure arguments
		    Line of arguments passed to configure call
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


while getopts "c:t:j:m:n:b:o:s:q:i:S:g:h" opt; do
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
  if [[ "$HOST" =~ "cobra" ]]
  then
    CLUSTER="cobra"
  fi
  if [[ "$HOST" =~ "talos" ]]
  then
    CLUSTER="talos"
  fi
  if [[ "$HOST" =~ "freya" ]]
  then
    CLUSTER="freya"
  fi
  if [[ "$HOST" =~ "draco" ]]
  then
    CLUSTER="draco"
  fi

  echo "Running on $CLUSTER with runner $CI_RUNNER_DESCRIPTION with tag $CI_RUNNER_TAGS on $mpiTasks tasks"

  #distcheck
  if [[ "$CI_RUNNER_TAGS" =~ "distcheck" ]]
  then
    cp $HOME/runners/job_script_templates/run_${CLUSTER}_1node.sh .
    echo "./configure " "$configureArgs" >> ./run_${CLUSTER}_1node.sh
    echo " " >> ./run_${CLUSTER}_1node.sh
    echo "export TASKS=$mpiTasks" >> ./run_${CLUSTER}_1node.sh
    echo "export DISTCHECK_CONFIGURE_FLAGS=\" $configureArgs \" "  >> ./run_${CLUSTER}_1node.sh
    echo "make distcheck TEST_FLAGS=\" $matrixSize $nrEV $blockSize \" || { chmod u+rwX -R . ; exit 1 ; } " >> ./run_${CLUSTER}_1node.sh
    echo " " >> ./run_${CLUSTER}_1node.sh
    echo "#copy everything back from /tmp/elpa to runner directory" >> ./run_${CLUSTER}_1node.sh
    echo "cp -r * \$runner_path" >> ./run_${CLUSTER}_1node.sh

    echo " "
    echo "Job script for the run"
    cat ./run_${CLUSTER}_1node.sh
    echo " "
    echo "Submitting to SLURM"
    sbatch -W ./run_${CLUSTER}_1node.sh
    exitCode=$?

    echo " "
    echo "Exit Code of sbatch: $exitCode"
    echo " "
    cat ./ELPA_CI.out.*
    #if [ $exitCode -ne 0 ]
    #then
      cat ./ELPA_CI.err.*
    #fi
    if [ -f ./test-suite.log ]
    then
      cat ./test-suite.log
    fi

  fi

  #if [ $exitCode -ne 0 ]
  #then
    cat ./test-suite.log
  #fi

  exit $exitCode

fi
