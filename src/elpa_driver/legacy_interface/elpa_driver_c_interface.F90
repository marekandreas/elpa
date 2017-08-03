!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), formerly known as
!      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
!    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!      Informatik,
!    - Technische Universität München, Lehrstuhl für Informatik mit
!      Schwerpunkt Wissenschaftliches Rechnen ,
!    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!      and
!    - IBM Deutschland GmbH
!
!
!    More information can be found here:
!    http://elpa.mpcdf.mpg.de/
!
!    ELPA is free software: you can redistribute it and/or modify
!    it under the terms of the version 3 of the license of the
!    GNU Lesser General Public License as published by the Free
!    Software Foundation.
!
!    ELPA is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public License
!    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
!
!    ELPA reflects a substantial effort on the part of the original
!    ELPA consortium, and we ask you to respect the spirit of the
!    license that we chose: i.e., please contribute any changes you
!    may have back to the original ELPA library distribution, and keep
!    any derivatives of ELPA under the same license that we chose for
!    the original distribution, the GNU Lesser General Public License.
!
! Author: Andreas Marek, MCPDF
#include "config-f90.h"

  !lc> /*! \brief C interface to driver function "elpa_solve_evp_real_double"
  !lc> *
  !lc> *  \param  na                        Order of matrix a
  !lc> *  \param  nev                       Number of eigenvalues needed.
  !lc> *                                    The smallest nev eigenvalues/eigenvectors are calculated.
  !lc> *  \param  a                         Distributed matrix for which eigenvalues are to be computed.
  !lc> *                                    Distribution is like in Scalapack.
  !lc> *                                    The full matrix must be set (not only one half like in scalapack).
  !lc> *  \param lda                        Leading dimension of a
  !lc> *  \param ev(na)                     On output: eigenvalues of a, every processor gets the complete set
  !lc> *  \param q                          On output: Eigenvectors of a
  !lc> *                                    Distribution is like in Scalapack.
  !lc> *                                    Must be always dimensioned to the full size (corresponding to (na,na))
  !lc> *                                    even if only a part of the eigenvalues is needed.
  !lc> *  \param ldq                        Leading dimension of q
  !lc> *  \param nblk                       blocksize of cyclic distribution, must be the same in both directions!
  !lc> *  \param matrixCols                 distributed number of matrix columns
  !lc> *  \param mpi_comm_rows              MPI-Communicator for rows
  !lc> *  \param mpi_comm_cols              MPI-Communicator for columns
  !lc> *  \param mpi_coll_all               MPI communicator for the total processor set
  !lc> *  \param THIS_REAL_ELPA_KERNEL_API  specify used ELPA2 kernel via API
  !lc> *  \param useQR                      use QR decomposition 1 = yes, 0 = no
  !lc> *  \param useGPU                     use GPU (1=yes, 0=No)
  !lc> *  \param method                     choose whether to use ELPA 1stage or 2stage solver
  !lc> *                                    possible values: "1stage" => use ELPA 1stage solver
  !lc> *                                                      "2stage" => use ELPA 2stage solver
  !lc> *                                                       "auto"   => (at the moment) use ELPA 2stage solver
  !lc> *
  !lc> *  \result                     int: 1 if error occured, otherwise 0
  !lc> */
#define REALCASE 1
#define DOUBLE_PRECISION 1
  !lc> int elpa_solve_evp_real_double(int na, int nev, double *a, int lda, double *ev, double *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_REAL_ELPA_KERNEL_API, int useQR, int useGPU, char *method);
#include "../../general/precision_macros.h"
#include "./elpa_driver_c_interface_template.F90"
#undef DOUBLE_PRECISION
#undef REALCASE

#ifdef WANT_SINGLE_PRECISION_REAL
  !lc> /*! \brief C interface to driver function "elpa_solve_evp_real_single"
  !lc> *
  !lc> *  \param  na                        Order of matrix a
  !lc> *  \param  nev                       Number of eigenvalues needed.
  !lc> *                                    The smallest nev eigenvalues/eigenvectors are calculated.
  !lc> *  \param  a                         Distributed matrix for which eigenvalues are to be computed.
  !lc> *                                    Distribution is like in Scalapack.
  !lc> *                                    The full matrix must be set (not only one half like in scalapack).
  !lc> *  \param lda                        Leading dimension of a
  !lc> *  \param ev(na)                     On output: eigenvalues of a, every processor gets the complete set
  !lc> *  \param q                          On output: Eigenvectors of a
  !lc> *                                    Distribution is like in Scalapack.
  !lc> *                                    Must be always dimensioned to the full size (corresponding to (na,na))
  !lc> *                                    even if only a part of the eigenvalues is needed.
  !lc> *  \param ldq                        Leading dimension of q
  !lc> *  \param nblk                       blocksize of cyclic distribution, must be the same in both directions!
  !lc> *  \param matrixCols                 distributed number of matrix columns
  !lc> *  \param mpi_comm_rows              MPI-Communicator for rows
  !lc> *  \param mpi_comm_cols              MPI-Communicator for columns
  !lc> *  \param mpi_coll_all               MPI communicator for the total processor set
  !lc> *  \param THIS_REAL_ELPA_KERNEL_API  specify used ELPA2 kernel via API
  !lc> *  \param useQR                      use QR decomposition 1 = yes, 0 = no
  !lc> *  \param useGPU                     use GPU (1=yes, 0=No)
  !lc> *  \param method                     choose whether to use ELPA 1stage or 2stage solver
  !lc> *                                    possible values: "1stage" => use ELPA 1stage solver
  !lc> *                                                      "2stage" => use ELPA 2stage solver
  !lc> *                                                       "auto"   => (at the moment) use ELPA 2stage solver
  !lc> *
  !lc> *  \result                     int: 1 if error occured, otherwise 0
  !lc> */
#define REALCASE 1
#define SINGLE_PRECISION 1
#undef DOUBLE_PRECISION
  !lc> int elpa_solve_evp_real_single(int na, int nev, float *a, int lda, float *ev, float *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_REAL_ELPA_KERNEL_API, int useQR, int useGPU, char *method);
#include "../../general/precision_macros.h"
#include "elpa_driver_c_interface_template.F90"
#undef SINGLE_PRECISION
#undef DOUBLE_PRECISION
#undef REALCASE
#endif /* WANT_SINGLE_PRECISION_REAL */

  !lc> #include <complex.h>
  !lc> /*! \brief C interface to driver function "elpa_solve_evp_complex_double"
  !lc> *
  !lc> *  \param  na                           Order of matrix a
  !lc> *  \param  nev                          Number of eigenvalues needed.
  !lc> *                                       The smallest nev eigenvalues/eigenvectors are calculated.
  !lc> *  \param  a                            Distributed matrix for which eigenvalues are to be computed.
  !lc> *                                       Distribution is like in Scalapack.
  !lc> *                                       The full matrix must be set (not only one half like in scalapack).
  !lc> *  \param lda                           Leading dimension of a
  !lc> *  \param ev(na)                        On output: eigenvalues of a, every processor gets the complete set
  !lc> *  \param q                             On output: Eigenvectors of a
  !lc> *                                       Distribution is like in Scalapack.
  !lc> *                                       Must be always dimensioned to the full size (corresponding to (na,na))
  !lc> *                                       even if only a part of the eigenvalues is needed.
  !lc> *  \param ldq                           Leading dimension of q
  !lc> *  \param nblk                          blocksize of cyclic distribution, must be the same in both directions!
  !lc> *  \param matrixCols                    distributed number of matrix columns
  !lc> *  \param mpi_comm_rows                 MPI-Communicator for rows
  !lc> *  \param mpi_comm_cols                 MPI-Communicator for columns
  !lc> *  \param mpi_coll_all                  MPI communicator for the total processor set
  !lc> *  \param THIS_COMPLEX_ELPA_KERNEL_API  specify used ELPA2 kernel via API
  !lc> *  \param useGPU                        use GPU (1=yes, 0=No)
  !lc> *  \param method                        choose whether to use ELPA 1stage or 2stage solver
  !lc> *                                       possible values: "1stage" => use ELPA 1stage solver
  !lc> *                                                        "2stage" => use ELPA 2stage solver
  !lc> *                                                         "auto"   => (at the moment) use ELPA 2stage solver
  !lc> *
  !lc> *  \result                     int: 1 if error occured, otherwise 0
  !lc> */
#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
  !lc> int elpa_solve_evp_complex_double(int na, int nev, double complex *a, int lda, double *ev, double complex *q, int ldq, int nblk, int matrixCols,
  !lc>                                   int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_COMPLEX_ELPA_KERNEL_API, int useGPU, char *method);
#include "../../general/precision_macros.h"
#include "./elpa_driver_c_interface_template.F90"
#undef DOUBLE_PRECISION
#undef COMPLEXCASE


#ifdef WANT_SINGLE_PRECISION_COMPLEX
  !lc> #include <complex.h>
  !lc> /*! \brief C interface to driver function "elpa_solve_evp_complex_single"
  !lc> *
  !lc> *  \param  na                           Order of matrix a
  !lc> *  \param  nev                          Number of eigenvalues needed.
  !lc> *                                       The smallest nev eigenvalues/eigenvectors are calculated.
  !lc> *  \param  a                            Distributed matrix for which eigenvalues are to be computed.
  !lc> *                                       Distribution is like in Scalapack.
  !lc> *                                       The full matrix must be set (not only one half like in scalapack).
  !lc> *  \param lda                           Leading dimension of a
  !lc> *  \param ev(na)                        On output: eigenvalues of a, every processor gets the complete set
  !lc> *  \param q                             On output: Eigenvectors of a
  !lc> *                                       Distribution is like in Scalapack.
  !lc> *                                       Must be always dimensioned to the full size (corresponding to (na,na))
  !lc> *                                       even if only a part of the eigenvalues is needed.
  !lc> *  \param ldq                           Leading dimension of q
  !lc> *  \param nblk                          blocksize of cyclic distribution, must be the same in both directions!
  !lc> *  \param matrixCols                    distributed number of matrix columns
  !lc> *  \param mpi_comm_rows                 MPI-Communicator for rows
  !lc> *  \param mpi_comm_cols                 MPI-Communicator for columns
  !lc> *  \param mpi_coll_all                  MPI communicator for the total processor set
  !lc> *  \param THIS_COMPLEX_ELPA_KERNEL_API  specify used ELPA2 kernel via API
  !lc> *  \param useGPU                        use GPU (1=yes, 0=No)
  !lc> *  \param method                        choose whether to use ELPA 1stage or 2stage solver
  !lc> *                                       possible values: "1stage" => use ELPA 1stage solver
  !lc> *                                                        "2stage" => use ELPA 2stage solver
  !lc> *                                                         "auto"   => (at the moment) use ELPA 2stage solver
  !lc> *
  !lc> *  \result                     int: 1 if error occured, otherwise 0
  !lc> */
#define COMPLEXCASE 1
#define SINGLE_PRECISION 1
#undef DOUBLE_PRECISION
  !lc> int elpa_solve_evp_complex_single(int na, int nev, complex float *a, int lda, float *ev, complex float *q, int ldq, int nblk, int matrixCols,
  !lc>                                   int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_COMPLEX_ELPA_KERNEL_API, int useGPU, char *method);
#include "../../general/precision_macros.h"
#include "./elpa_driver_c_interface_template.F90"
#undef SINGLE_PRECISION
#undef DOUBLE_PRECISION
#undef COMPLEXCASE

#endif /* WANT_SINGLE_PRECISION_COMPLEX */
