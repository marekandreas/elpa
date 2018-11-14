"""distributedmatrix.py -- classes for distributed matrices

This file contains the python classes to use with the wrapper.
"""
import numpy as np
from functools import wraps
from .wrapper import Elpa

class ProcessorLayout:
    """Create rectangular processor layout for use with distributed matrices"""
    def __init__(self, comm):
        """Initialize processor layout.

        Args:
            comm: MPI communicator from mpi4py
        """
        nprocs = comm.Get_size()
        rank = comm.Get_rank()
        for np_cols in range(int(np.sqrt(nprocs)), 0, -1):
            if nprocs % np_cols == 0:
              break
        #if nprocs == 1:
        #    np_cols = 1
        np_rows = nprocs//np_cols
        # column major distribution of processors
        my_pcol = rank // np_rows
        my_prow = rank % np_rows
        self.np_cols, self.np_rows = np_cols, np_rows
        self.my_pcol, self.my_prow = my_pcol, my_prow
        self.comm = comm
        self.comm_f = comm.py2f()


class DistributedMatrix:
    """Class for generating a distributed block-cyclic matrix

    The data attribute contains the array in the correct size for the local
    processor.
    """
    def __init__(self, processor_layout, na, nev, nblk, dtype=np.float64):
        """Initialize distributed matrix for a given processor layout.

        Args:
            processor_layout (ProcessorLayout): has to be created from MPI
                communicator
            na (int): dimension of matrix
            nev (int): number of eigenvectors/eigenvalues to be computed
            nblk (int): block size of distributed matrix
            dtype: data type of matrix
        """
        self.na = na
        self.nev = nev
        self.nblk = nblk
        self.processor_layout = processor_layout

        # get local size
        self.na_rows = self.numroc(na, nblk, processor_layout.my_prow, 0,
                                   processor_layout.np_rows)
        self.na_cols = self.numroc(na, nblk, processor_layout.my_pcol, 0,
                                   processor_layout.np_cols)
        # create array
        self.data = np.empty((self.na_rows, self.na_cols),
                             dtype=dtype, order='F')

        self.elpa = None

    @classmethod
    def from_communicator(cls, comm, na, nev, nblk, dtype=np.float64):
        """Initialize distributed matrix from a MPI communicator.

        Args:
            comm: MPI communicator from mpi4py
            na (int): dimension of matrix
            nev (int): number of eigenvectors/eigenvalues to be computed
            nblk (int): block size of distributed matrix
            dtype: data type of matrix
        """
        processor_layout = ProcessorLayout(comm)
        return cls(processor_layout, na, nev, nblk, dtype)

    @classmethod
    def from_comm_world(cls, na, nev, nblk, dtype=np.float64):
        """Initialize distributed matrix from the MPI_COMM_WORLD communicator.

        Args:
            na (int): dimension of matrix
            nev (int): number of eigenvectors/eigenvalues to be computed
            nblk (int): block size of distributed matrix
            dtype: data type of matrix
        """
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        processor_layout = ProcessorLayout(comm)
        return cls(processor_layout, na, nev, nblk, dtype)

    @classmethod
    def like(cls, matrix):
        """Get a DistributedMatrix with the same parameters as matrix"""
        return cls(matrix.processor_layout, matrix.na, matrix.nev, matrix.nblk,
                   matrix.data.dtype)

    def get_local_index(self, global_row, global_col):
        """compute local row and column indices from global ones

        Returns a tuple of the local row and column indices
        """
        local_row = self.indxg2l(global_row, self.nblk,
                                 self.processor_layout.my_prow, 0,
                                 self.processor_layout.np_rows)
        local_col = self.indxg2l(global_col, self.nblk,
                                 self.processor_layout.my_pcol, 0,
                                 self.processor_layout.np_cols)
        return local_row, local_col

    def get_global_index(self, local_row, local_col):
        """compute global row and column indices from local ones

        Returns a tuple of the global row and column indices
        """
        global_row = self.indxl2g(local_row, self.nblk,
                                  self.processor_layout.my_prow, 0,
                                  self.processor_layout.np_rows)
        global_col = self.indxl2g(local_col, self.nblk,
                                  self.processor_layout.my_pcol, 0,
                                  self.processor_layout.np_cols)
        return global_row, global_col

    def is_local_index(self, global_row, global_col):
        """check if global index is stored on current processor"""
        return self.is_local_row(global_row) and self.is_local_col(global_col)

    def is_local_row(self, global_row):
        """check if global row is stored on this processor"""
        process_row = self.indxg2p(global_row, self.nblk,
                                   self.processor_layout.my_prow, 0,
                                   self.processor_layout.np_rows)
        return process_row == self.processor_layout.my_prow

    def is_local_col(self, global_col):
        process_col = self.indxg2p(global_col, self.nblk,
                                   self.processor_layout.my_pcol, 0,
                                   self.processor_layout.np_cols)
        return process_col == self.processor_layout.my_pcol

    @staticmethod
    def indxg2l(indxglob, nb, iproc, isrcproc, nprocs):
        """compute local index from global index indxglob

        original netlib scalapack source:

        .. code-block:: fortran

            INDXG2L = NB*((INDXGLOB-1)/(NB*NPROCS))+MOD(INDXGLOB-1,NB)+1
        """
        # adapt to python 0-based indexing
        return nb*(indxglob//(nb*nprocs)) + indxglob%nb

    @staticmethod
    def indxl2g(indxloc, nb, iproc, isrcproc, nprocs):
        """compute global index from local index indxloc

        original netlib scalapack source:

        .. code-block:: fortran

            INDXL2G = NPROCS*NB*((INDXLOC-1)/NB) + MOD(INDXLOC-1,NB) +
                      MOD(NPROCS+IPROC-ISRCPROC, NPROCS)*NB + 1
        """
        # adapt to python 0-based indexing
        return nprocs*nb*(indxloc//nb) + indxloc%nb + \
            ((nprocs+iproc-isrcproc)%nprocs)*nb

    @staticmethod
    def indxg2p(indxglob, nb, iproc, isrcproc, nprocs):
        """compute process coordinate for global index

        original netlib scalapack source:

        .. code-block:: fortran

            INDXG2P = MOD( ISRCPROC + (INDXGLOB - 1) / NB, NPROCS )
        """
        # adapt to python 0-based indexing
        return (isrcproc + indxglob // nb) % nprocs

    @staticmethod
    def numroc(n, nb, iproc, isrcproc, nprocs):
        """Get local dimensions of distributed block-cyclic matrix.

        Programmed after scalapack source (tools/numroc.f on netlib).
        """
        mydist = (nprocs + iproc - isrcproc) % nprocs
        nblocks = n // nb
        result = (nblocks // nprocs) * nb
        extrablks = nblocks % nprocs
        if mydist < extrablks:
            result += nb
        elif mydist == extrablks:
            result += n % nb
        return int(result)

    def _initialized_elpa(function):
        # wrapper to ensure one-time initialization of Elpa object
        @wraps(function)
        def wrapped_function(self):
            if self.elpa is None:
                self.elpa = Elpa.from_distributed_matrix(self)
            return function(self)
        return wrapped_function

    @_initialized_elpa
    def compute_eigenvectors(self):
        """Compute eigenvalues and eigenvectors

        The eigenvectors are stored in columns.
        This function returns a dictionary with entries 'eigenvalues' and
        'eigenvectors'.

        After computing the eigenvectors, the original content of the matrix is
        lost.
        """
        eigenvectors = DistributedMatrix.like(self)
        eigenvalues = np.zeros(self.na, dtype=np.float64)
        # call ELPA
        self.elpa.eigenvectors(self.data, eigenvalues, eigenvectors.data)
        return {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors}

    @_initialized_elpa
    def compute_eigenvalues(self):
        """Compute only the eigenvalues.

        This function returns the eigenvalues as an array.

        After computing the eigenvalues, the original content of the matrix is
        lost.
        """
        eigenvalues = np.zeros(self.na, dtype=np.float64)
        # call ELPA
        self.elpa.eigenvalues(self.data, eigenvalues)
        return eigenvalues

    def set_data_from_global_matrix(self, matrix):
        """Set local part of the global matrix"""
        for local_row in range(self.na_rows):
            for local_col in range(self.na_cols):
                global_row, global_col = self.get_global_index(local_row,
                                                               local_col)
                self.data[local_row, local_col] = matrix[global_row,
                                                         global_col]

    def dot(self, vector):
        """Compute dot product of matrix with vector.

        This blocked implementation is much faster than the naive
        implementation.
        """
        if len(vector.shape) > 1 or vector.shape[0] != self.na:
            raise ValueError("Error: shape of vector {} incompatible to "
                             "matrix of size {:d}x{:d}.".format(
                                 vector.shape, self.na, self.na))
        from mpi4py import MPI
        summation = np.zeros_like(vector)
        # loop only over blocks here
        for local_row in range(0, self.na_rows, self.nblk):
            for local_col in range(0, self.na_cols, self.nblk):
                # do not go beyond the end of the matrix
                row_block_size = min(local_row + self.nblk,
                                     self.na_rows) - local_row
                col_block_size = min(local_col + self.nblk,
                                     self.na_cols) - local_col
                global_row, global_col = self.get_global_index(local_row,
                                                               local_col)
                # use numpy for faster dot product of local block
                summation[global_row:global_row+row_block_size] += \
                    np.dot(self.data[local_row:local_row + row_block_size,
                                     local_col:local_col + col_block_size],
                           vector[global_col:global_col+col_block_size])
        result = np.zeros_like(vector)
        self.processor_layout.comm.Allreduce(summation, result, op=MPI.SUM)
        return result

    def _dot_naive(self, vector):
        """Compute naive dot product of matrix with vector.

        Still in here as an example and for testing purposes.
        """
        from mpi4py import MPI
        summation = np.zeros_like(vector)
        for local_row in range(self.na_rows):
            for local_col in range(self.na_cols):
                global_row, global_col = self.get_global_index(local_row,
                                                               local_col)
                summation[global_row] += self.data[local_row, local_col] *\
                    vector[global_col]
        result = np.zeros_like(vector)
        self.processor_layout.comm.Allreduce(summation, result, op=MPI.SUM)
        return result

    def get_column(self, global_col):
        """Return global column"""
        from mpi4py import MPI
        column = np.zeros(self.na, dtype=self.data.dtype)
        temporary = np.zeros_like(column)
        if self.is_local_col(global_col):
            for global_row in range(self.na):
                if not self.is_local_row(global_row):
                    continue
                local_row, local_col = self.get_local_index(global_row,
                                                            global_col)
                temporary[global_row] = self.data[local_row, local_col]
        # this could be done more efficiently with a gather
        self.processor_layout.comm.Allreduce(temporary, column, op=MPI.SUM)
        return column

    def get_row(self, global_row):
        """Return global row"""
        from mpi4py import MPI
        row = np.zeros(self.na, dtype=self.data.dtype)
        temporary = np.zeros_like(row)
        if self.is_local_row(global_row):
            for global_col in range(self.na):
                if not self.is_local_col(global_col):
                    continue
                local_row, local_col = self.get_local_index(global_row,
                                                            global_col)
                temporary[global_col] = self.data[local_row, local_col]
        # this could be done more efficiently with a gather
        self.processor_layout.comm.Allreduce(temporary, row, op=MPI.SUM)
        return row

    def global_indices(self):
        """Return iterator over global indices of matrix.

        Use together with set_data_global_index and get_data_global_index.
        """
        for local_row in range(self.na_rows):
            for local_col in range(self.na_cols):
                yield self.get_global_index(local_row, local_col)

    def set_data_for_global_index(self, global_row, global_col, value):
        """Set value of matrix at global coordinates"""
        if self.is_local_index(global_row, global_col):
            local_row, local_col = self.get_local_index(global_row, global_col)
            self.data[local_row, local_col] = value

    def get_data_for_global_index(self, global_row, global_col):
        """Get value of matrix at global coordinates"""
        if self.is_local_index(global_row, global_col):
            local_row, local_col = self.get_local_index(global_row, global_col)
            return self.data[local_row, local_col]
        else:
            raise ValueError('Index out of bounds: global row {:d}, '
                             'global col {:d}'.format(global_row, global_col))

    def global_block_indices(self):
        """Return iterator over global indices of matrix blocks.

        Use together with set_block_global_index and get_block_global_index
        for more efficient loops.
        """
        for local_row in range(0, self.na_rows, self.nblk):
            for local_col in range(0, self.na_cols, self.nblk):
                # do not go beyond the end of the matrix
                row_block_size = min(local_row + self.nblk,
                                     self.na_rows) - local_row
                col_block_size = min(local_col + self.nblk,
                                     self.na_cols) - local_col
                global_row, global_col = self.get_global_index(local_row,
                                                               local_col)
                yield global_row, global_col, row_block_size, col_block_size

    def set_block_for_global_index(self, global_row, global_col,
                                   row_block_size, col_block_size, value):
        """Set value of block of matrix at global coordinates"""
        if self.is_local_index(global_row, global_col):
            local_row, local_col = self.get_local_index(global_row, global_col)
            if value.shape != (row_block_size, col_block_size):
                raise ValueError("value has the wrong shape. "
                                 "Expected: {}, found: {}."
                                 .format((row_block_size, col_block_size),
                                         value.shape)
                                 )
            self.data[local_row:local_row+row_block_size,
                      local_col:local_col+col_block_size] = value

    def get_block_for_global_index(self, global_row, global_col,
                                   row_block_size, col_block_size):
        """Get value of block of matrix at global coordinates"""
        if self.is_local_index(global_row, global_col):
            local_row, local_col = self.get_local_index(global_row, global_col)
            if local_row+row_block_size > self.na_rows or \
                    local_col+col_block_size > self.na_cols:
                raise ValueError("Block size wrong: exceeds dimensions of"
                                 " matrix.")
            return self.data[local_row:local_row+row_block_size,
                             local_col:local_col+col_block_size]
        else:
            raise ValueError('Index out of bounds: global row {:d}, '
                             'global col {:d}'.format(global_row, global_col))
