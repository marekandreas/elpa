def test_import():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD


def test_barrier():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm.Barrier()
