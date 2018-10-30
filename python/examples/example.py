#!/usr/bin/env python
import numpy as np
from pyelpa import DistributedMatrix
import sys

# set some parameters for matrix layout
na = 1000
nev = 200
nblk = 16

# create distributed matrix
a = DistributedMatrix.from_comm_world(na, nev, nblk)

# function for setting the matrix
# this is the easiest but also slowest way
def set_matrix(a):
    for global_row, global_col in a.global_indices():
        a.set_data_for_global_index(global_row, global_col,
                                    global_row*global_col)

# set a
set_matrix(a)

print("Call ELPA eigenvectors")
sys.stdout.flush()

# now compute nev of na eigenvectors and eigenvalues
data = a.compute_eigenvectors()
eigenvalues = data['eigenvalues']
eigenvectors = data['eigenvectors']

print("Done")

# now eigenvectors.data contains the local part of the eigenvector matrix
# which is stored in a block-cyclic distributed layout and eigenvalues contains
# all computed eigenvalues on all cores

# set a again because it has changed after calling elpa
set_matrix(a)

print("Call ELPA eigenvalues")
sys.stdout.flush()

# now compute nev of na eigenvalues
eigenvalues = a.compute_eigenvalues()

print("Done")

# now eigenvalues contains all computed eigenvalues on all cores
