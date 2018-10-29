#!/usr/bin/env python
import numpy as np
from pyelpa import ProcessorLayout, DistributedMatrix, Elpa
from mpi4py import MPI
import sys

# set some parameters for matrix layout
na = 1000
nev = 200
nblk = 16

# initialize processor layout, needed for calling ELPA
comm = MPI.COMM_WORLD
layout_p = ProcessorLayout(comm)

# create arrays
a = DistributedMatrix(layout_p, na, nev, nblk)
eigenvectors = DistributedMatrix(layout_p, na, nev, nblk)
eigenvalues = np.zeros(na, dtype=np.float64)

# initialize elpa
e = Elpa.from_distributed_matrix(a)

# set input matrix (a.data) on this core (a is stored in a block-cyclic
# distributed layout; local size: a.na_rows x a.na_cols)
a.data[:, :] = np.random.rand(a.na_rows, a.na_cols).astype(np.float64)

print("Call ELPA eigenvectors")
sys.stdout.flush()
# now compute nev of na eigenvectors and eigenvalues
e.eigenvectors(a.data, eigenvalues, eigenvectors.data)
print("Done")

# now eigenvectors.data contains the local part of the eigenvector matrix
# which is stored in a block-cyclic distributed layout

# now eigenvalues contains all computed eigenvalues on all cores

print("Call ELPA eigenvalues")
sys.stdout.flush()
# now compute nev of na eigenvalues
e.eigenvalues(a.data, eigenvalues)
print("Done")

# now eigenvalues contains all computed eigenvalues on all cores
