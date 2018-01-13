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

function usage() {
	cat >&2 <<-EOF

		Call all the necessary steps to perform an ELPA CI test

		Usage:
		  run_ci_tests [-c configure arguments] [-j makeTasks] [-h] [-t MPI Tasks] [-m matrix size] [-n number of eigenvectors] [-b block size] [-o OpenMP threads] [-s skipStep]

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

		 -j makeTaks
		    Number of processes make should use during build (default 1)

		 -s skipStep
		    Skip the test run if 1 (default 0)

		 -h
		    Print this help text
	EOF
}


while getopts "c:t:j:m:n:b:o:s:h" opt; do
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
		:)
			echo "Option -$OPTARG requires an argument" >&2;;
		h)
			usage
			exit 1;;
		*)
			exit 1;;
	esac
done

if [ $skipStep -eq 0 ]
then
  echo "Skipping the test since option -s has been specified"
  exit 0
else
  eval ./configure $configureArgs
   if [ $? -ne 0 ]; then cat config.log && exit 1; fi
  
  make -j $makeTasks
  if [ $? -ne 0 ]; then exit 1; fi
  
  OMP_NUM_THREADS=$ompThreads make check TASKS=$mpiTasks TEST_FLAGS="$matrixSize $nrEV $blockSize" || { cat test-suite-log; exit 1; }
  if [ $? -ne 0 ]; then exit 1; fi
   
  grep -i "Expected %stop" test-suite.log && exit 1 || true ;
  if [ $? -ne 0 ]; then exit 1; fi
fi

