"""pyelpa -- python wrapper for ELPA

This wrapper uses cython to wrap the C API of ELPA (Eigenvalue SoLvers for
Petaflop-Applications) so that it can be called from python.

Examples:

1. Use the Elpa object to access the eigenvectors/eigenvalues wrapper:

>>> import numpy as np
... from pyelpa import ProcessorLayout, DistributedMatrix, Elpa
... from mpi4py import MPI
... import sys
... 
... # set some parameters for matrix layout
... na = 1000
... nev = 200
... nblk = 16
... 
... # initialize processor layout, needed for calling ELPA
... comm = MPI.COMM_WORLD
... layout_p = ProcessorLayout(comm)
... 
... # create arrays
... a = DistributedMatrix(layout_p, na, nev, nblk)
... eigenvectors = DistributedMatrix(layout_p, na, nev, nblk)
... eigenvalues = np.zeros(na, dtype=np.float64)
... 
... # initialize elpa
... e = Elpa.from_distributed_matrix(a)
... 
... # set input matrix (a.data) on this core (a is stored in a block-cyclic
... # distributed layout; local size: a.na_rows x a.na_cols)
... # Caution: using this, the global matrix will not be symmetric; this is just
... # and example to show how to access the data
... a.data[:, :] = np.random.rand(a.na_rows, a.na_cols).astype(np.float64)
... 
... # now compute nev of na eigenvectors and eigenvalues
... e.eigenvectors(a.data, eigenvalues, eigenvectors.data)
... 
... # now eigenvectors.data contains the local part of the eigenvector matrix
... # which is stored in a block-cyclic distributed layout
... 
... # now eigenvalues contains all computed eigenvalues on all cores
... 
... # now compute nev of na eigenvalues
... e.eigenvalues(a.data, eigenvalues)
... 
... # now eigenvalues contains all computed eigenvalues on all cores


2. Use the functions provided by the DistributedMatrix object:

>>> import numpy as np
... from pyelpa import DistributedMatrix
... 
... # set some parameters for matrix layout
... na = 1000
... nev = 200
... nblk = 16
... 
... a = DistributedMatrix.from_comm_world(na, nev, nblk)
... # use a diagonal matrix as input
... matrix = np.diagflat(np.arange(na)**2)
... # set from global matrix
... a.set_data_from_global_matrix(matrix)
... 
... data = a.compute_eigenvectors()
... eigenvalues = data['eigenvalues']
... eigenvectors = data['eigenvectors']
... # now eigenvectors.data contains the local part of the eigenvector matrix
... # which is stored in a block-cyclic distributed layout
... 
... # now eigenvalues contains all computed eigenvalues on all cores
"""
from .wrapper import Elpa
from .distributedmatrix import ProcessorLayout, DistributedMatrix

__all__ = ['ProcessorLayout', 'DistributedMatrix', 'Elpa']
