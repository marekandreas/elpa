#!/bin/bash

# Arg 1 - number of MPI processes to start
# Arg 2 - matrix size
# Arg 3 - Number of EVs
# Arg 4 - SCALAPACK block size

function runSingleTest() {
	fileName="results/t5-cpu-bandred-final/test-$1-$2-$3-$4.txt"

	echo "Ranks: $1, matrix size: $2, EVs: $3, block size: $4, file: \"$fileName\""

	cat /dev/null > tmp.txt

	cmd="mpiexec -np $1 ./test_real2 $2 $3 $4 &> $fileName"
	echo $cmd
	eval $cmd	
}

# Arg 1 - number of MPI processes to start

function runTestGroup() {
    # Test suites
    runSingleTest $1 5120 512 64
    runSingleTest $1 5120 2560 64
    runSingleTest $1 5120 5120 64

    runSingleTest $1 5120 512 128
    runSingleTest $1 5120 2560 128
    runSingleTest $1 5120 5120 128
}

# Actual tests
runTestGroup 1
runTestGroup 2
runTestGroup 4
runTestGroup 8



