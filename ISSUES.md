## A list of known (and hopefully soon solved) issues of *ELPA* ##

For more details and recent updates please visit the online [issue system] (https://gitlab.mpcdf.mpg.de/elpa/elpa/issues)
Issues which are not mentioned in a newer release are (considered as) solved

### ELPA 2016.11.001 release ###
- at the moment no issues are known

### ELPA 2016.05.003 release  ###

- the Fortran module file "precision" was installed as private. Lead to linking problems

### ELPA 2016.05.002 release  ###

- QR decomposition fails for certain combinations of matrix sizes, number of eigenvalues to compute and block size

### ELPA 2016.05.001 release  ###

- QR decomposition fails for certain combinations of matrix sizes, number of eigenvalues to compute and block size
- The generated check-scripts (in the step "make check") do not call the binary with "mpiexec" when *ELPA* is build with
  MPI

