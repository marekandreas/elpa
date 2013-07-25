#mpirun -np 6 -host=sb010 ./test_real2 20480 64 128
#mpirun -np 4 -host=sb010 ./test_real2 20480 64 128
#mpirun -np 2 -host=sb010 ./test_real2 20480 64 128
#mpirun -np 1 -host=sb010 ./test_real2 20480 64 128

mpiexec -np $1 ./test_real2 7000 70 64

