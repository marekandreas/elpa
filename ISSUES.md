## A list of known (and hopefully soon solved) issues of *ELPA* ##

For more details and recent updates please visit the online [issue system](https://gitlab.mpcdf.mpg.de/elpa/elpa/issues)
Issues which are not mentioned in a newer release are (considered as) solved.

### ELPA 2024.05.001 release ###
- internal matrix redististribution does not (yet) work for ELPA 2stage


### ELPA 2024.03.001 release ###
Fixed compilation for skew-symmetric case
Corrected hermitian_multiply step

### ELPA 2023.11.001 release ###
Fixed compilation for skew-symmetric case
Erroneous hermitian_multiply step

### ELPA 2023.05.001 release ###
Currently no known issues

### ELPA 2022.05.001 release ###
Currently no known issues

### ELPA 2021.11.002 release ###
Fix the error when choosing the Nvidia GPU kernels

### ELPA 2021.11.001 release ###
Error with the activation of the Nvidia GPU kernel, _if_ the standard and the new SM80_kernel have been configured

### ELPA 2021.05.002 release ###
Correct the SO version which wrongly in
release 2021.05.001 went backwards 

### ELPA 2021.05.001 release ###
Currently no issues known

### ELPA 2020.11.001 release ###
- fixes a problem with GPU kernels
- fixes a problem with VSX kernels
- fixes a problem with NEON kernels
- do not use MPI_COMM_WORLD in check_gpu function
- add missing test_scalapack_template.F90 to EXTRA_DIST list

### ELPA 2020.05.001 release ###
- fixes a problem with configuring the GPU kernels

### ELPA 2019.11.001 release ###
- memory leak in in GPU version has been fixed
- no other issues currently known

### ELPA 2019.05.002 release ###
- memory leak in GPU version

### ELPA 2019.05.001 release ###
- memory leak in GPU version

### ELPA 2018.11.001 release ###
- on (officially not supported) 32bit systems hangs in MPI can occur

### ELPA 2018.05.001 release ###
- on (officially not supported) 32bit systems hangs in MPI can occur
- printing of autotune has been implemented

### ELPA 2017.11.001 release ###
- the elpa autotune print functions cannot print at the moment
- on (officially not supported) 32bit systems hangs in MPI can occur

### ELPA 2017.05.003 release ###
- at the moment no issues are known

### ELPA 2017.05.002 release ###
- at the moment no issues are known

### ELPA 2017.05.001 release ###
- accidentaly a memory leak has been introduced

### ELPA 2017.05.001.rc2 release ###
- compilation with Intel Compiler 2018 beta does not work
  (see https://gitlab.mpcdf.mpg.de/elpa/elpa/issues/54)

### ELPA 2017.05.001.rc1 release ###
- compilation with Intel Compiler 2018 beta does not work
  (see https://gitlab.mpcdf.mpg.de/elpa/elpa/issues/54)

### ELPA 2016.11.001 release ###
- at the moment no issues are known

### ELPA 2016.05.003 release  ###

- the Fortran module file "precision" was installed as private. Lead to linking problems

### ELPA 2016.05.002 release  ###

- QR decomposition fails for certain combinations of matrix sizes, number of eigenvalues to compute and block size

### ELPA 2016.05.001 release  ###

- QR decomposition fails for certain combinations of matrix sizes, number of eigenvalues to compute and block size
- The generated check-scripts (in the step "make check") do not call the binary with `mpiexec` when *ELPA* is build with
  MPI

