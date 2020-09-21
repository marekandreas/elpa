#!/usr/bin/env python3
import numpy as np
from pyelpa import DistributedMatrix
import sys

# set some parameters for matrix layout
na = 1000
nev = 200
nblk = 16

# create distributed matrix
a = DistributedMatrix.from_comm_world(na, nev, nblk)

# set matrix a by looping over indices
# this is the easiest but also slowest way
for global_row, global_col in a.global_indices():
    a.set_data_for_global_index(global_row, global_col,
                                global_row*global_col)

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
# this time set it by looping over blocks, this is more efficient
for global_row, global_col, row_block_size, col_block_size in \
        a.global_block_indices():
    # set block with product of indices
    x = np.arange(global_row, global_row + row_block_size)[:, None] * \
        np.arange(global_col, global_col + col_block_size)[None, :]
    a.set_block_for_global_index(global_row, global_col,
                                 row_block_size, col_block_size, x)

print("Call ELPA eigenvalues")
sys.stdout.flush()

# now compute nev of na eigenvalues
eigenvalues = a.compute_eigenvalues()

print("Done")

# now eigenvalues contains all computed eigenvalues on all cores
