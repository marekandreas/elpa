def test_numroc():
    from pyelpa import DistributedMatrix
    n = 100
    nb = 16
    assert(DistributedMatrix.numroc(n, nb, 0, 0, 3) == 36)
    assert(DistributedMatrix.numroc(n, nb, 1, 0, 3) == 32)
    assert(DistributedMatrix.numroc(n, nb, 1, 1, 3) == 36)
