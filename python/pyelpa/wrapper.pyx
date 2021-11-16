"""wrapper.pyx -- python wrapper for ELPA

This file contains the cython part of the wrapper.
"""
cimport numpy as np
import numpy as np
import sys

if 'mpi4py.MPI' in sys.modules.keys():
    raise NotImplementedError('Please load the pyelpa module before mpi4py, '
                              'otherwise there will be MPI problems.')

# import the function definitions from the ELPA header
cdef import from "<elpa/elpa.h>":
    cdef struct elpa_struct:
        pass
    ctypedef elpa_struct *elpa_t
    int elpa_init(int api_version)
    void elpa_uninit(int *error)
    elpa_t elpa_allocate(int *error)
    void elpa_deallocate(elpa_t handle, int *error)
    int elpa_setup(elpa_t handle)
    void elpa_set_integer(elpa_t handle, const char *name, int value, int *error)
    void elpa_get_integer(elpa_t handle, const char *name, int *value, int *error)
    void elpa_set_double(elpa_t handle, const char *name, double value, int *error)
    void elpa_get_double(elpa_t handle, const char *name, double *value, int *error)
    void elpa_eigenvectors_a_h_a_d(elpa_t handle, double *a, double *ev, double *q, int *error)
    void elpa_eigenvectors_a_h_a_f(elpa_t handle, float *a, float *ev, float *q, int *error)
    void elpa_eigenvectors_a_h_a_dc(elpa_t handle, double complex *a, double *ev, double complex *q, int *error)
    void elpa_eigenvectors_a_h_a_fc(elpa_t handle, float complex *a, float *ev, float complex *q, int *error)
    void elpa_eigenvalues_a_h_a_d(elpa_t handle, double *a, double *ev, int *error)
    void elpa_eigenvalues_a_h_a_f(elpa_t handle, float *a, float *ev, int *error)
    void elpa_eigenvalues_a_h_a_dc(elpa_t handle, double complex *a, double *ev, int *error)
    void elpa_eigenvalues_a_h_a_fc(elpa_t handle, float complex *a, float *ev, int *error)
    int ELPA_OK
    int ELPA_SOLVER_2STAGE


cdef class Elpa:
    """Wrapper for ELPA C interface.

    Provides routines for initialization, deinitialization, setting and getting
    properties and for calling the eigenvectors and eigenvalues routines.
    The routines eigenvectors and eigenvalues select the right ELPA routine to
    call depending on the argument type.
    """
    cdef elpa_t handle

    def __init__(self):
        """Run initialization and allocation of handle"""
        if elpa_init(20171201) != ELPA_OK:
            raise RuntimeError("ELPA API version not supported")
        cdef int error
        handle = elpa_allocate(&error)
        self.handle = handle

    def set_integer(self, description, int value):
        """Wraps elpa_set_integer"""
        cdef int error
        if isinstance(description, unicode):
            # encode to ascii for passing to C
            description = (<unicode>description).encode('ascii')
        cdef const char* c_string = description
        elpa_set_integer(<elpa_t>self.handle, description, value, &error)

    def get_integer(self, description):
        """Wraps elpa_get_integer"""
        cdef int error
        if isinstance(description, unicode):
            # encode to ascii for passing to C
            description = (<unicode>description).encode('ascii')
        cdef const char* c_string = description
        cdef int tmp
        elpa_get_integer(<elpa_t>self.handle, c_string, &tmp, &error)
        return tmp

    def set_double(self, description, double value):
        """Wraps elpa_set_double"""
        cdef int error
        if isinstance(description, unicode):
            # encode to ascii for passing to C
            description = (<unicode>description).encode('ascii')
        cdef const char* c_string = description
        elpa_set_double(<elpa_t>self.handle, description, value, &error)

    def get_double(self, description):
        """Wraps elpa_get_double"""
        cdef int error
        if isinstance(description, unicode):
            # encode to ascii for passing to C
            description = (<unicode>description).encode('ascii')
        cdef const char* c_string = description
        cdef double tmp
        elpa_get_double(<elpa_t>self.handle, c_string, &tmp, &error)
        return tmp

    def setup(self):
        """call setup function"""
        elpa_setup(<elpa_t>self.handle)

    def __del__(self):
        """Deallocation of handle and deinitialization"""
        cdef int error
        elpa_deallocate(<elpa_t>self.handle, &error)
        elpa_uninit(&error)

    def eigenvectors_d(self,
                       np.ndarray[np.float64_t, ndim=2] a,
                       np.ndarray[np.float64_t, ndim=1] ev,
                       np.ndarray[np.float64_t, ndim=2] q):
        cdef int error
        elpa_eigenvectors_a_h_a_d(<elpa_t>self.handle, <np.float64_t *>a.data,
                            <np.float64_t *>ev.data, <np.float64_t *>q.data,
                            <int*>&error)
        if error != ELPA_OK:
            raise RuntimeError("ELPA returned error value {:d}.".format(error))

    def eigenvectors_f(self,
                       np.ndarray[np.float32_t, ndim=2] a,
                       np.ndarray[np.float32_t, ndim=1] ev,
                       np.ndarray[np.float32_t, ndim=2] q):
        cdef int error
        elpa_eigenvectors_a_h_a_f(<elpa_t>self.handle, <np.float32_t *>a.data,
                            <np.float32_t *>ev.data, <np.float32_t *>q.data,
                            <int*>&error)
        if error != ELPA_OK:
            raise RuntimeError("ELPA returned error value {:d}.".format(error))

    def eigenvectors_dc(self,
                        np.ndarray[np.complex128_t, ndim=2] a,
                        np.ndarray[np.float64_t, ndim=1] ev,
                        np.ndarray[np.complex128_t, ndim=2] q):
        cdef int error
        elpa_eigenvectors_a_h_a_dc(<elpa_t>self.handle, <np.complex128_t *>a.data,
                             <np.float64_t *>ev.data, <np.complex128_t *>q.data,
                             <int*>&error)
        if error != ELPA_OK:
            raise RuntimeError("ELPA returned error value {:d}.".format(error))

    def eigenvectors_fc(self,
                        np.ndarray[np.complex64_t, ndim=2] a,
                        np.ndarray[np.float32_t, ndim=1] ev,
                        np.ndarray[np.complex64_t, ndim=2] q):
        cdef int error
        elpa_eigenvectors_a_h_a_fc(<elpa_t>self.handle, <np.complex64_t *>a.data,
                             <np.float32_t *>ev.data, <np.complex64_t *>q.data,
                             <int*>&error)
        if error != ELPA_OK:
            raise RuntimeError("ELPA returned error value {:d}.".format(error))

    def eigenvectors(self, a, ev, q):
        """Compute eigenvalues and eigenvectors.

        The data type of a is tested and the corresponding ELPA routine called

        Args:
            a (DistributedMatrix): problem matrix
            ev (numpy.ndarray): array of size a.na to store eigenvalues
            q (DistributedMatrix): store eigenvectors
        """
        if a.dtype == np.float64:
            self.eigenvectors_d(a, ev, q)
        elif a.dtype == np.float32:
            self.eigenvectors_f(a, ev, q)
        elif a.dtype == np.complex128:
            self.eigenvectors_dc(a, ev, q)
        elif a.dtype == np.complex64:
            self.eigenvectors_fc(a, ev, q)
        else:
            raise TypeError("Type not known.")

    def eigenvalues_d(self,
                       np.ndarray[np.float64_t, ndim=2] a,
                       np.ndarray[np.float64_t, ndim=1] ev):
        cdef int error
        elpa_eigenvalues_a_h_a_d(<elpa_t>self.handle, <np.float64_t *>a.data,
                           <np.float64_t *>ev.data, <int*>&error)
        if error != ELPA_OK:
            raise RuntimeError("ELPA returned error value {:d}.".format(error))

    def eigenvalues_f(self,
                       np.ndarray[np.float32_t, ndim=2] a,
                       np.ndarray[np.float32_t, ndim=1] ev):
        cdef int error
        elpa_eigenvalues_a_h_a_f(<elpa_t>self.handle, <np.float32_t *>a.data,
                           <np.float32_t *>ev.data, <int*>&error)
        if error != ELPA_OK:
            raise RuntimeError("ELPA returned error value {:d}.".format(error))

    def eigenvalues_dc(self,
                        np.ndarray[np.complex128_t, ndim=2] a,
                        np.ndarray[np.float64_t, ndim=1] ev):
        cdef int error
        elpa_eigenvalues_a_h_a_dc(<elpa_t>self.handle, <np.complex128_t *>a.data,
                            <np.float64_t *>ev.data, <int*>&error)
        if error != ELPA_OK:
            raise RuntimeError("ELPA returned error value {:d}.".format(error))

    def eigenvalues_fc(self,
                        np.ndarray[np.complex64_t, ndim=2] a,
                        np.ndarray[np.float32_t, ndim=1] ev):
        cdef int error
        elpa_eigenvalues_a_h_a_fc(<elpa_t>self.handle, <np.complex64_t *>a.data,
                            <np.float32_t *>ev.data, <int*>&error)
        if error != ELPA_OK:
            raise RuntimeError("ELPA returned error value {:d}.".format(error))

    def eigenvalues(self, a, ev):
        """Compute eigenvalues.

        The data type of a is tested and the corresponding ELPA routine called

        Args:
            a (DistributedMatrix): problem matrix
            ev (numpy.ndarray): array of size a.na to store eigenvalues
        """
        if a.dtype == np.float64:
            self.eigenvalues_d(a, ev)
        elif a.dtype == np.float32:
            self.eigenvalues_f(a, ev)
        elif a.dtype == np.complex128:
            self.eigenvalues_dc(a, ev)
        elif a.dtype == np.complex64:
            self.eigenvalues_fc(a, ev)
        else:
            raise TypeError("Type not known.")

    @classmethod
    def from_distributed_matrix(cls, a):
        """Initialize ELPA with values from a distributed matrix

        Args:
            a (DistributedMatrix): matrix to get values from
        """
        self = cls()
        # Set parameters the matrix and it's MPI distribution
        self.set_integer("mpi_comm_parent", <int>a.processor_layout.comm_f)
        self.set_integer("na", <int>a.na)
        self.set_integer("nev", <int>a.nev)
        self.set_integer("local_nrows", <int>a.na_rows)
        self.set_integer("local_ncols", <int>a.na_cols)
        self.set_integer("nblk", <int>a.nblk)
        self.set_integer("process_row", <int>a.processor_layout.my_prow)
        self.set_integer("process_col", <int>a.processor_layout.my_pcol)
        # Setup
        self.setup()
        # if desired, set tunable run-time options
        self.set_integer("solver", ELPA_SOLVER_2STAGE)
        return self
