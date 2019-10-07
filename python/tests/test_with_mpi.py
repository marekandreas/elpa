import pytest

# combinations of na, nev, nblk to run all the tests with
parameter_list = [
    (200, 20, 16),
    (200, 200, 16),
    (200, 20, 64),
    (200, 200, 64),
    (200, 20, 4),
    (200, 200, 4),
    (50, 20, 16),
    (100, 20, 16),
]

def get_random_vector(size):
    """generate random vector with given size that is equal on all cores"""
    import numpy as np
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    am_i_root = comm.Get_rank() == 0
    vector = np.empty(size)
    if am_i_root:
        vector[:] = np.random.rand(size)
    comm.Bcast(vector)
    return vector


def test_processor_layout():
    from pyelpa import ProcessorLayout
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    layout_p = ProcessorLayout(comm)
    assert(comm.Get_size() == layout_p.np_cols*layout_p.np_rows)
    assert(layout_p.my_prow >= 0)
    assert(layout_p.my_pcol >= 0)
    assert(layout_p.my_prow <= comm.Get_size())
    assert(layout_p.my_pcol <= comm.Get_size())


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_distributed_matrix_from_processor_layout(na, nev, nblk):
    import numpy as np
    from pyelpa import ProcessorLayout, DistributedMatrix
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    layout_p = ProcessorLayout(comm)

    for dtype in [np.float64, np.float32, np.complex64, np.complex128]:
        a = DistributedMatrix(layout_p, na, nev, nblk, dtype=dtype)
        assert(a.data.dtype == dtype)
        assert(a.data.shape == (a.na_rows, a.na_cols))


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_distributed_matrix_from_communicator(na, nev, nblk):
    import numpy as np
    from pyelpa import DistributedMatrix
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    for dtype in [np.float64, np.float32, np.complex64, np.complex128]:
        a = DistributedMatrix.from_communicator(comm, na, nev, nblk,
                                                dtype=dtype)
        assert(a.data.dtype == dtype)
        assert(a.data.shape == (a.na_rows, a.na_cols))


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_distributed_matrix_from_world(na, nev, nblk):
    import numpy as np
    from pyelpa import DistributedMatrix

    for dtype in [np.float64, np.float32, np.complex64, np.complex128]:
        a = DistributedMatrix.from_comm_world(na, nev, nblk, dtype=dtype)
        assert(a.data.dtype == dtype)
        assert(a.data.shape == (a.na_rows, a.na_cols))


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_distributed_matrix_like_other_matrix(na, nev, nblk):
    import numpy as np
    from pyelpa import ProcessorLayout, DistributedMatrix
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    layout_p = ProcessorLayout(comm)

    for dtype in [np.float64, np.float32, np.complex64, np.complex128]:
        a = DistributedMatrix(layout_p, na, nev, nblk, dtype=dtype)
        b = DistributedMatrix.like(a)
        assert(a.na == b.na)
        assert(a.nev == b.nev)
        assert(a.nblk == b.nblk)
        assert(a.data.dtype == b.data.dtype)
        assert(a.data.shape == b.data.shape)


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_call_eigenvectors(na, nev, nblk):
    import numpy as np
    from pyelpa import ProcessorLayout, DistributedMatrix, Elpa
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    layout_p = ProcessorLayout(comm)

    for dtype in [np.float64, np.complex128]:
        # create arrays
        a = DistributedMatrix(layout_p, na, nev, nblk, dtype=dtype)
        a.data[:, :] = np.random.rand(a.na_rows, a.na_cols).astype(dtype)
        q = DistributedMatrix(layout_p, na, nev, nblk, dtype=dtype)
        ev = np.zeros(na, dtype=np.float64)

        e = Elpa.from_distributed_matrix(a)
        e.eigenvectors(a.data, ev, q.data)


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_call_eigenvalues(na, nev, nblk):
    import numpy as np
    from pyelpa import ProcessorLayout, DistributedMatrix, Elpa
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    layout_p = ProcessorLayout(comm)

    for dtype in [np.float64, np.complex128]:
        # create arrays
        a = DistributedMatrix(layout_p, na, nev, nblk, dtype=dtype)
        a.data[:, :] = np.random.rand(a.na_rows, a.na_cols).astype(dtype)
        ev = np.zeros(na, dtype=np.float64)

        e = Elpa.from_distributed_matrix(a)
        e.eigenvalues(a.data, ev)


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_compare_eigenvalues_to_those_from_eigenvectors(na, nev, nblk):
    import numpy as np
    from pyelpa import ProcessorLayout, DistributedMatrix, Elpa
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    layout_p = ProcessorLayout(comm)

    for dtype in [np.float64, np.complex128]:
        # create arrays
        a = DistributedMatrix(layout_p, na, nev, nblk, dtype=dtype)
        random_matrix = np.random.rand(a.na_rows, a.na_cols).astype(dtype)
        a.data[:, :] = random_matrix
        q = DistributedMatrix(layout_p, na, nev, nblk, dtype=dtype)
        ev = np.zeros(na, dtype=np.float64)
        ev2 = np.zeros(na, dtype=np.float64)

        e = Elpa.from_distributed_matrix(a)
        e.eigenvectors(a.data, ev, q.data)

        a.data[:, :] = random_matrix
        e.eigenvalues(a.data, ev2)

        assert(np.allclose(ev, ev2))


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_compare_eigenvalues_to_those_from_eigenvectors_self_functions(
        na, nev, nblk):
    import numpy as np
    from pyelpa import DistributedMatrix

    for dtype in [np.float64, np.complex128]:
        # create arrays
        a = DistributedMatrix.from_comm_world(na, nev, nblk, dtype=dtype)
        random_matrix = np.random.rand(a.na_rows, a.na_cols).astype(dtype)
        a.data[:, :] = random_matrix
        data = a.compute_eigenvectors()

        a.data[:, :] = random_matrix
        eigenvalues = a.compute_eigenvalues()

        assert(np.allclose(data['eigenvalues'], eigenvalues))


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_distributed_matrix_global_index(na, nev, nblk):
    import numpy as np
    from pyelpa import ProcessorLayout, DistributedMatrix
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    layout_p = ProcessorLayout(comm)

    for dtype in [np.float64, np.complex128]:
        a = DistributedMatrix(layout_p, na, nev, nblk, dtype=dtype)
        for local_row in range(a.na_rows):
            for local_col in range(a.na_cols):
                global_row, global_col = a.get_global_index(local_row,
                                                            local_col)
                l_row, l_col = a.get_local_index(global_row, global_col)
                assert(global_row >= 0 and global_row < a.na)
                assert(global_col >= 0 and global_col < a.na)
                assert(local_row == l_row and local_col == l_col)


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_distributed_matrix_local_index(na, nev, nblk):
    import numpy as np
    from pyelpa import ProcessorLayout, DistributedMatrix
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    layout_p = ProcessorLayout(comm)

    for dtype in [np.float64, np.complex128]:
        a = DistributedMatrix(layout_p, na, nev, nblk, dtype=dtype)
        for global_row in range(a.na):
            for global_col in range(a.na):
                if not a.is_local_index(global_row, global_col):
                    continue
                local_row, local_col = a.get_local_index(global_row,
                                                         global_col)
                g_row, g_col = a.get_global_index(local_row, local_col)
                assert(local_row >= 0 and local_row < a.na_rows)
                assert(local_col >= 0 and local_col < a.na_cols)
                assert(global_row == g_row and global_col == g_col)


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_distributed_matrix_indexing_loop(na, nev, nblk):
    import numpy as np
    from pyelpa import ProcessorLayout, DistributedMatrix
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    layout_p = ProcessorLayout(comm)

    for dtype in [np.float64, np.complex128]:
        a = DistributedMatrix(layout_p, na, nev, nblk, dtype=dtype)
        for local_row in range(a.na_rows):
            for local_col in range(a.na_cols):
                global_row, global_col = a.get_global_index(local_row,
                                                            local_col)
                a.data[local_row, local_col] = global_row*10 + global_col

        for global_row in range(a.na):
            for global_col in range(a.na):
                if not a.is_local_index(global_row, global_col):
                    continue
                local_row, local_col = a.get_local_index(global_row,
                                                         global_col)
                assert(a.data[local_row, local_col] ==
                       global_row*10 + global_col)


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_setting_global_matrix(na, nev, nblk):
    import numpy as np
    from pyelpa import ProcessorLayout, DistributedMatrix
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    layout_p = ProcessorLayout(comm)

    for dtype in [np.float64, np.complex128]:
        a = DistributedMatrix(layout_p, na, nev, nblk, dtype=dtype)
        # get global matrix that is equal on all cores
        matrix = get_random_vector(na*na).reshape(na, na).astype(dtype)
        a.set_data_from_global_matrix(matrix)

        # check data
        for global_row in range(a.na):
            for global_col in range(a.na):
                if not a.is_local_index(global_row, global_col):
                    continue
                local_row, local_col = a.get_local_index(global_row,
                                                         global_col)
                assert(a.data[local_row, local_col] ==
                       matrix[global_row, global_col])


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_dot_product(na, nev, nblk):
    import numpy as np
    from pyelpa import ProcessorLayout, DistributedMatrix

    for dtype in [np.float64, np.complex128]:
        a = DistributedMatrix.from_comm_world(na, nev, nblk, dtype=dtype)
        # get global matrix and vector that is equal on all cores
        matrix = get_random_vector(na*na).reshape(na, na).astype(dtype)
        vector = get_random_vector(na).astype(dtype)

        a.set_data_from_global_matrix(matrix)

        product_distributed = a.dot(vector)
        product_naive = a._dot_naive(vector)
        product_serial = np.dot(matrix, vector)

        assert(np.allclose(product_distributed, product_serial))
        assert(np.allclose(product_distributed, product_naive))

@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_dot_product_incompatible_size(na, nev, nblk):
    import numpy as np
    from pyelpa import DistributedMatrix

    for dtype in [np.float64, np.complex128]:
        a = DistributedMatrix.from_comm_world(na, nev, nblk, dtype=dtype)
        # get global matrix and vector that is equal on all cores
        matrix = get_random_vector(na*na).reshape(na, na).astype(dtype)
        vector = get_random_vector(na*2).astype(dtype)

        a.set_data_from_global_matrix(matrix)

        with pytest.raises(ValueError):
            product_distributed = a.dot(vector)


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_validate_eigenvectors(na, nev, nblk):
    import numpy as np
    from pyelpa import DistributedMatrix

    for dtype in [np.float64, np.complex128]:
        a = DistributedMatrix.from_comm_world(na, nev, nblk, dtype=dtype)
        # get a symmetric/hermitian matrix
        matrix = get_random_vector(na*na).reshape(na, na).astype(dtype)
        matrix = 0.5*(matrix + np.conj(matrix.T))
        a.set_data_from_global_matrix(matrix)

        data = a.compute_eigenvectors()
        eigenvalues = data['eigenvalues']
        eigenvectors = data['eigenvectors']
        # reset data of a
        a.set_data_from_global_matrix(matrix)
        for index in range(a.nev):
            eigenvector = eigenvectors.get_column(index)
            scaled_eigenvector = eigenvalues[index]*eigenvector
            # test solution
            assert(np.allclose(a.dot(eigenvector),
                               scaled_eigenvector))


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_validate_eigenvectors_to_numpy(na, nev, nblk):
    import numpy as np
    from numpy import linalg
    from pyelpa import DistributedMatrix

    for dtype in [np.float64, np.complex128]:
        a = DistributedMatrix.from_comm_world(na, nev, nblk, dtype=dtype)
        # get a symmetric/hermitian matrix
        matrix = get_random_vector(na*na).reshape(na, na).astype(dtype)
        matrix = 0.5*(matrix + np.conj(matrix.T))
        a.set_data_from_global_matrix(matrix)

        data = a.compute_eigenvectors()
        eigenvalues = data['eigenvalues']
        eigenvectors = data['eigenvectors']

        # get numpy solution
        eigenvalues_np, eigenvectors_np = linalg.eigh(matrix)

        assert(np.allclose(eigenvalues, eigenvalues_np))
        for index in range(a.nev):
            eigenvector = eigenvectors.get_column(index)
            assert(np.allclose(eigenvector, eigenvectors_np[:, index]) or
                   np.allclose(eigenvector, -eigenvectors_np[:, index]))


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_accessing_matrix(na, nev, nblk):
    import numpy as np
    from pyelpa import DistributedMatrix

    for dtype in [np.float64, np.complex128]:
        a = DistributedMatrix.from_comm_world(na, nev, nblk, dtype=dtype)
        matrix = get_random_vector(na*na).reshape(na, na).astype(dtype)
        a.set_data_from_global_matrix(matrix)

        for index in range(a.na):
            column = a.get_column(index)
            assert(np.allclose(column, matrix[:, index]))
            row = a.get_row(index)
            assert(np.allclose(row, matrix[index, :]))


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_global_index_iterator(na, nev, nblk):
    import numpy as np
    from pyelpa import DistributedMatrix

    for dtype in [np.float64, np.complex128]:
        a = DistributedMatrix.from_comm_world(na, nev, nblk, dtype=dtype)
        for i, j in a.global_indices():
            assert(a.is_local_index(i, j))


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_global_index_access(na, nev, nblk):
    import numpy as np
    from pyelpa import DistributedMatrix

    for dtype in [np.float64, np.complex128]:
        a = DistributedMatrix.from_comm_world(na, nev, nblk, dtype=dtype)
        for i, j in a.global_indices():
            x = dtype(i*j)
            a.set_data_for_global_index(i, j, x)
        for i, j in a.global_indices():
            x = a.get_data_for_global_index(i, j)
            assert(np.isclose(x, i*j))


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_global_block_iterator(na, nev, nblk):
    import numpy as np
    from pyelpa import DistributedMatrix

    for dtype in [np.float64, np.complex128]:
        a = DistributedMatrix.from_comm_world(na, nev, nblk, dtype=dtype)
        for i, j, blk_i, blk_j in a.global_block_indices():
            assert(a.is_local_index(i, j))
            assert(blk_i <= nblk)
            assert(blk_j <= nblk)
            assert(i+blk_i <= na)
            assert(j+blk_j <= na)


@pytest.mark.parametrize("na,nev,nblk", parameter_list)
def test_global_block_access(na, nev, nblk):
    import numpy as np
    from pyelpa import DistributedMatrix

    for dtype in [np.float64, np.complex128]:
        a = DistributedMatrix.from_comm_world(na, nev, nblk, dtype=dtype)
        for i, j, blk_i, blk_j in a.global_block_indices():
            x = np.arange(i, i+blk_i)[:, None] * np.arange(j, j+blk_j)[None, :]
            a.set_block_for_global_index(i, j, blk_i, blk_j, x)
        for i, j, blk_i, blk_j in a.global_block_indices():
            original = np.arange(i, i+blk_i)[:, None] * np.arange(j, j+blk_j)[None, :]
            x = a.get_block_for_global_index(i, j, blk_i, blk_j)
            assert(np.allclose(x, original))
        for i, j in a.global_indices():
            x = a.get_data_for_global_index(i, j)
            assert(np.isclose(x, i*j))
