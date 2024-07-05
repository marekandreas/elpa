#!/bin/bash
set -e
set -x

#some defaults
makeTasks=1
mpiTasks=2
matrixSize=150
nrEV=$matrixSize
blockSize=16
ompThreads=1
configueArg=""
skipStep=0
batchCommand=""
interactiveRun="yes"
slurmBatch="no"
gpuJob="no"

function usage() {
	cat >&2 <<-EOF

		Call all the necessary steps to perform an ELPA CI test

		Usage:
		  run_ci_tests [-c configure arguments] [-j makeTasks] [-h] [-t MPI Tasks] [-m matrix size] [-n number of eigenvectors] [-b block size] [-o OpenMP threads] [-s skipStep] [-q submit command] [-i interactive run] [-S submit to Slurm] [-g GPU job]"

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

		 -j makeTasks
		    Number of processes make should use during build (default 1)

		 -s skipStep
		    Skip the test run if 1 (default 0)

		 -q submit command
		    Job steps will be submitted via command to a batch system (default no submission)

		 -i interactive_run
		    if "yes" NO no batch command will be triggered

		 -S submit to slurm
		    if "yes" a SLURM batch job will be submitted

		 -g gpu job
		    if "yes" a GPU job is assumed
		 -h
		    Print this help text
	EOF
}


while getopts "c:t:j:m:n:b:o:s:q:i:S:g:h" opt; do
	case $opt in
		j)
			makeTasks=$OPTARG;;
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
		s)
			skipStep=$OPTARG;;
		q)
			batchCommand=$OPTARG;;
		i)
			interactiveRun=$OPTARG;;
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


if [ $skipStep -eq 1 ]
then
  echo "Skipping the test since option -s has been specified"
  exit 0
fi
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

  echo "Running on $CLUSTER with runner $CI_RUNNER_DESCRIPTION with tag $CI_RUNNER_TAGS on $mpiTasks tasks"

  # GPU runners
  if [ "$gpuJob" == "yes" ]
  then
    cp $HOME/runners/job_script_templates/run_${CLUSTER}_1node_2GPU.sh .
    echo "if \[ \$SLURM_PROCID -eq 0 \]" >> ./run_${CLUSTER}_1node_2GPU.sh
    echo "then" >> ./run_${CLUSTER}_1node_2GPU.sh
    echo "echo SLURM_NODELIST: \$SLURM_NODELIST" >> ./run_${CLUSTER}_1node_2GPU.sh
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
    echo "make -j 16" >> ./run_${CLUSTER}_1node_2GPU.sh
    echo "touch build_done" >> ./run_${CLUSTER}_1node_2GPU.sh
    echo "fi" >> ./run_${CLUSTER}_1node_2GPU.sh
    echo " " >> ./run_${CLUSTER}_1node_2GPU.sh
    echo "export OMP_NUM_THREADS=$ompThreads" >> ./run_${CLUSTER}_1node_2GPU.sh
    echo "export TASKS=$mpiTasks" >> ./run_${CLUSTER}_1node_2GPU.sh
    #echo "while ! \[ -f ./build_done \];" >> ./run_${CLUSTER}_1node_2GPU.sh
    #echo "do" >> ./run_${CLUSTER}_1node_2GPU.sh
    #echo "echo \""\ > /dev/null" >> ./run_${CLUSTER}_1node_2GPU.sh
    #echo "done" >> ./run_${CLUSTER}_1node_2GPU.sh
    echo "CHECK_LEVEL=extended make check TEST_FLAGS=\" $matrixSize $nrEV $blockSize \" " >> ./run_${CLUSTER}_1node_2GPU.sh
    echo " " >> ./run_${CLUSTER}_1node_2GPU.sh
    echo "exitCode=\$?" >> ./run_${CLUSTER}_1node_2GPU.sh
    echo " " >> ./run_${CLUSTER}_1node_2GPU.sh
    echo "##copy everything back from /tmp/elpa to runner directory"  >> ./run_${CLUSTER}_1node_2GPU.sh
    echo "#cp -r * \$runner_path"  >> ./run_${CLUSTER}_1node_2GPU.sh
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

    #if (( $exitCode > 0 ))
    #then
      cat ./ELPA_CI_2gpu.err.*
    #fi
    
  else

    #SSE, AVX, AVX2, and AVX-512 runners
    if [[ "$CI_RUNNER_TAGS" =~ "sse" ]] || [[ "$CI_RUNNER_TAGS" =~ "avx" ]] || [[ "$CI_RUNNER_TAGS" =~ "avx2" ]]  || [ ["$CI_RUNNER_TAGS" =~ "avx512" ]]
    then
      cp $HOME/runners/job_script_templates/run_${CLUSTER}_1node.sh .
      echo " " >> ./run_${CLUSTER}_1node.sh
      echo "if \[ \$SLURM_PROCID -eq 0 \]" >> ./run_${CLUSTER}_1node.sh
      echo "then" >> ./run_${CLUSTER}_1node.sh
      echo "echo SLURM_NODELIST: \$SLURM_NODELIST" >>  ./run_${CLUSTER}_1node.sh
      echo "echo \"process \$SLURM_PROCID running configure\"" >> ./run_${CLUSTER}_1node.sh
      echo "\#decouple from SLURM \(maybe this could be removed\)" >> ./run_${CLUSTER}_1node.sh
      echo "export _save_SLURM_MPI_TYPE=\$SLURM_MPI_TYPE" >> ./run_${CLUSTER}_1node.sh
      echo "export _save_I_MPI_SLURM_EXT=\$I_MPI_SLURM_EXT"  >> ./run_${CLUSTER}_1node.sh
      echo "export _save_I_MPI_PMI_LIBRARY=\$I_MPI_PMI_LIBRARY" >> ./run_${CLUSTER}_1node.sh
      echo "export _save_I_MPI_PMI2=\$I_MPI_PMI2" >> ./run_${CLUSTER}_1node.sh
      echo "export _save_I_MPI_HYDRA_BOOTSTRAP=\$I_MPI_HYDRA_BOOTSTRAP" >> ./run_${CLUSTER}_1node.sh
      echo "unset SLURM_MPI_TYPE I_MPI_SLURM_EXT I_MPI_PMI_LIBRARY I_MPI_PMI2 I_MPI_HYDRA_BOOTSTRAP" >> ./run_${CLUSTER}_1node.sh
      echo " " >> ./run_${CLUSTER}_1node.sh
      echo "./configure " "$configureArgs"  >> ./run_${CLUSTER}_1node.sh
      echo " " >> ./run_${CLUSTER}_1node.sh
      echo "make -j 16 " >> ./run_${CLUSTER}_1node.sh
      echo "touch build_done" >> ./run_${CLUSTER}_1node.sh
      echo "fi" >> ./run_${CLUSTER}_1node.sh
      echo " " >> ./run_${CLUSTER}_1node.sh
      echo "export OMP_NUM_THREADS=$ompThreads" >> ./run_${CLUSTER}_1node.sh
      echo "export TASKS=$mpiTasks" >> ./run_${CLUSTER}_1node.sh
      echo "while ! [ -f ./build_done ];" >> ./run_${CLUSTER}_1node.sh
      echo "do" >> ./run_${CLUSTER}_1node.sh
      echo "echo \" \" > /dev/null" >> ./run_${CLUSTER}_1node.sh
      echo "done" >> ./run_${CLUSTER}_1node.sh
      echo "CHECK_LEVEL=extended make -j 10 check TEST_FLAGS=\" $matrixSize $nrEV $blockSize \"  " >> ./run_${CLUSTER}_1node.sh
      echo " " >> ./run_${CLUSTER}_1node.sh
      echo "exitCode=\$?" >> ./run_${CLUSTER}_1node.sh
      echo " " >> ./run_${CLUSTER}_1node.sh
      echo "#copy everything back from /tmp/elpa to runner directory"  >> ./run_${CLUSTER}_1node.sh
      echo "#cp -r * \$runner_path"  >> ./run_${CLUSTER}_1node.sh
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
        echo "Submission excited with exitCode $exitCode"
      fi

      echo " "
      echo "Exit Code of sbatch: $exitCode"
      echo " "
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


# not skipped then proceed
if [ "$slurmBatch" == "no" ]
then
  # this is the old, messy implementation for
  # - appdev
  # - freya-interactive
  # - draco
  # - buildtest
  # - virtual machine runners
  # hopefully this can be removed soon
  echo "Using old CI logic for appdev"
  if [ "$batchCommand" == "srun" ]
  then
    # use srun to start mpi jobs
    if [ "$interactiveRun" == "no" ]
    then
      # interactive runs are not possible
      echo "Running with $batchCommand with $SRUN_COMMANDLINE_CONFIGURE"
      # $batchCommand --ntasks-per-core=1 --ntasks=1 --cpus-per-task=1 $SRUN_COMMANDLINE_CONFIGURE bash -c ' {source /etc/profile.d/modules.sh && source ./ci_test_scripts/ci-env-vars && eval  ./configure $configureArgs; }'
      $batchCommand --ntasks-per-core=1 --ntasks=1 --cpus-per-task=1 $SRUN_COMMANDLINE_CONFIGURE ./ci_test_scripts/configure_step.sh "$configureArgs"

      if [ $? -ne 0 ]; then cat config.log && exit 1; fi
      $batchCommand --ntasks-per-core=1 --ntasks=1 --cpus-per-task=8 $SRUN_COMMANDLINE_BUILD ./ci_test_scripts/build_step.sh $makeTasks
      if [ $? -ne 0 ]; then exit 1; fi
      $batchCommand --ntasks-per-core=1 --ntasks=1 --cpus-per-task=2 $SRUN_COMMANDLINE_RUN ./ci_test_scripts/test_step.sh $mpiTasks $ompThreads "TEST_FLAGS=\" $matrixSize $nrEV $blockSize \" "
      if [ $? -ne 0 ]; then exit 1; fi

      grep -i "Expected %stop" test-suite.log && exit 1 || true ;
      if [ $? -ne 0 ]; then exit 1; fi
    else
      # interactive runs are possible
      #eval ./configure $configureArgs
      ./ci_test_scripts/configure_step.sh "$configureArgs"

      if [ $? -ne 0 ]; then cat config.log && exit 1; fi
    
      make -j $makeTasks
      if [ $? -ne 0 ]; then exit 1; fi
    
      OMP_NUM_THREADS=$ompThreads CHECK_LEVEL=extended make check TASKS=$mpiTasks TEST_FLAGS="$matrixSize $nrEV $blockSize" || { cat test-suite.log; exit 1; }
      if [ $? -ne 0 ]; then exit 1; fi
     
      grep -i "Expected %stop" test-suite.log && exit 1 || true ;
      if [ $? -ne 0 ]; then exit 1; fi

    fi

  else
    # do not use srun to start mpi applications
 
    #eval ./configure $configureArgs
    ./ci_test_scripts/configure_step.sh "$configureArgs"

    if [ $? -ne 0 ]; then cat config.log && exit 1; fi
    
    make -j $makeTasks
    if [ $? -ne 0 ]; then exit 1; fi
    
    OMP_NUM_THREADS=$ompThreads CHECK_LEVEL=extended make check TASKS=$mpiTasks TEST_FLAGS="$matrixSize $nrEV $blockSize" || { cat test-suite.log; exit 1; }
    if [ $? -ne 0 ]; then exit 1; fi
     
    grep -i "Expected %stop" test-suite.log && exit 1 || true ;
    if [ $? -ne 0 ]; then exit 1; fi
  fi
else
  # a submission to SLURM via a batch script will be done
  echo "Submitting to a SLURM batch system"
fi

