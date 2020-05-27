## A list of known (and hopefully soon solved) issues of *ELPA* ##

For more details and recent updates please visit the online [issue system](https://gitlab.mpcdf.mpg.de/elpa/elpa/issues)
Issues which are not mentioned in a newer release are (considered as) solved.

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

