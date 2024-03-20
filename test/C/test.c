/*   This file is part of ELPA.

     The ELPA library was originally created by the ELPA consortium,
     consisting of the following organizations:

     - Max Planck Computing and Data Facility (MPCDF), formerly known as
       Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
     - Bergische Universität Wuppertal, Lehrstuhl für angewandte
       Informatik,
     - Technische Universität München, Lehrstuhl für Informatik mit
       Schwerpunkt Wissenschaftliches Rechnen ,
     - Fritz-Haber-Institut, Berlin, Abt. Theorie,
     - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
       Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
       and
     - IBM Deutschland GmbH


     More information can be found here:
     http://elpa.mpcdf.mpg.de/

     ELPA is free software: you can redistribute it and/or modify
     it under the terms of the version 3 of the license of the
     GNU Lesser General Public License as published by the Free
     Software Foundation.

     ELPA is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU Lesser General Public License for more details.

     You should have received a copy of the GNU Lesser General Public License
     along with ELPA.  If not, see <http://www.gnu.org/licenses/>

     ELPA reflects a substantial effort on the part of the original
     ELPA consortium, and we ask you to respect the spirit of the
     license that we chose: i.e., please contribute any changes you
     may have back to the original ELPA library distribution, and keep
     any derivatives of ELPA under the same license that we chose for
     the original distribution, the GNU Lesser General Public License.
*/

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef WITH_MPI
#include <mpi.h>
#endif
#include <math.h>

#include <elpa/elpa.h>
#include <assert.h>


#if !(defined(TEST_REAL) ^ defined(TEST_COMPLEX))
#error "define exactly one of TEST_REAL or TEST_COMPLEX"
#endif

#if !(defined(TEST_SINGLE) ^ defined(TEST_DOUBLE))
#error "define exactly one of TEST_SINGLE or TEST_DOUBLE"
#endif

#if !(defined(TEST_SOLVER_1STAGE) ^ defined(TEST_SOLVER_2STAGE))
#error "define exactly one of TEST_SOLVER_1STAGE or TEST_SOLVER_2STAGE"
#endif

#ifdef TEST_GENERALIZED_DECOMP_EIGENPROBLEM
#define TEST_GENERALIZED_EIGENPROBLEM
#endif

#ifdef __cplusplus
#define double_complex std::complex<double>
#define float_complex std::complex<float>
#else
#define double_complex double complex
#define float_complex float complex
#endif

#ifdef TEST_SINGLE
#  define EV_TYPE float
#  ifdef TEST_REAL
#    define MATRIX_TYPE float
#    define PREPARE_MATRIX_RANDOM prepare_matrix_random_real_single_f
#    define PREPARE_MATRIX_RANDOM_SPD prepare_matrix_random_spd_real_single_f
#    define PREPARE_MATRIX_ANALYTIC prepare_matrix_analytic_real_single_f
#    define PREPARE_MATRIX_TOEPLITZ prepare_matrix_toeplitz_real_single_f
#    define CHECK_CORRECTNESS_EVP_NUMERIC_RESIDUALS check_correctness_evp_numeric_residuals_real_single_f
#    define CHECK_CORRECTNESS_EVP_GEN_NUMERIC_RESIDUALS check_correctness_evp_gen_numeric_residuals_real_single_f
#    define CHECK_CORRECTNESS_CHOLESKY check_correctness_cholesky_real_single_f
#    define CHECK_CORRECTNESS_HERMITIAN_MULTIPLY check_correctness_hermitian_multiply_real_single_f
#    define CHECK_CORRECTNESS_ANALYTIC check_correctness_analytic_real_single_f
#    define CHECK_CORRECTNESS_EIGENVALUES_TOEPLITZ check_correctness_eigenvalues_toeplitz_real_single_f
#  else
#    define MATRIX_TYPE float_complex
#    define PREPARE_MATRIX_RANDOM prepare_matrix_random_complex_single_f
#    define PREPARE_MATRIX_RANDOM_SPD prepare_matrix_random_spd_complex_single_f
#    define PREPARE_MATRIX_ANALYTIC prepare_matrix_analytic_complex_single_f
#    define PREPARE_MATRIX_TOEPLITZ prepare_matrix_toeplitz_complex_single_f
#    define CHECK_CORRECTNESS_EVP_NUMERIC_RESIDUALS check_correctness_evp_numeric_residuals_complex_single_f
#    define CHECK_CORRECTNESS_EVP_GEN_NUMERIC_RESIDUALS check_correctness_evp_gen_numeric_residuals_complex_single_f
#    define CHECK_CORRECTNESS_CHOLESKY check_correctness_cholesky_complex_single_f
#    define CHECK_CORRECTNESS_HERMITIAN_MULTIPLY check_correctness_hermitian_multiply_complex_single_f
#    define CHECK_CORRECTNESS_ANALYTIC check_correctness_analytic_complex_single_f
#    define CHECK_CORRECTNESS_EIGENVALUES_TOEPLITZ check_correctness_eigenvalues_toeplitz_complex_single_f
#  endif
#else
#  define EV_TYPE double
#  ifdef TEST_REAL
#    define MATRIX_TYPE double
#    define PREPARE_MATRIX_RANDOM prepare_matrix_random_real_double_f
#    define PREPARE_MATRIX_RANDOM_SPD prepare_matrix_random_spd_real_double_f
#    define PREPARE_MATRIX_ANALYTIC prepare_matrix_analytic_real_double_f
#    define PREPARE_MATRIX_TOEPLITZ prepare_matrix_toeplitz_real_double_f
#    define CHECK_CORRECTNESS_EVP_NUMERIC_RESIDUALS check_correctness_evp_numeric_residuals_real_double_f
#    define CHECK_CORRECTNESS_EVP_GEN_NUMERIC_RESIDUALS check_correctness_evp_gen_numeric_residuals_real_double_f
#    define CHECK_CORRECTNESS_CHOLESKY check_correctness_cholesky_real_double_f
#    define CHECK_CORRECTNESS_HERMITIAN_MULTIPLY check_correctness_hermitian_multiply_real_double_f
#    define CHECK_CORRECTNESS_ANALYTIC check_correctness_analytic_real_double_f
#    define CHECK_CORRECTNESS_EIGENVALUES_TOEPLITZ check_correctness_eigenvalues_toeplitz_real_double_f
#  else
#    define MATRIX_TYPE double_complex
#    define PREPARE_MATRIX_RANDOM prepare_matrix_random_complex_double_f
#    define PREPARE_MATRIX_RANDOM_SPD prepare_matrix_random_spd_complex_double_f
#    define PREPARE_MATRIX_ANALYTIC prepare_matrix_analytic_complex_double_f
#    define PREPARE_MATRIX_TOEPLITZ prepare_matrix_toeplitz_complex_double_f
#    define CHECK_CORRECTNESS_EVP_NUMERIC_RESIDUALS check_correctness_evp_numeric_residuals_complex_double_f
#    define CHECK_CORRECTNESS_EVP_GEN_NUMERIC_RESIDUALS check_correctness_evp_gen_numeric_residuals_complex_double_f
#    define CHECK_CORRECTNESS_CHOLESKY check_correctness_cholesky_complex_double_f
#    define CHECK_CORRECTNESS_HERMITIAN_MULTIPLY check_correctness_hermitian_multiply_complex_double_f
#    define CHECK_CORRECTNESS_ANALYTIC check_correctness_analytic_complex_double_f
#    define CHECK_CORRECTNESS_EIGENVALUES_TOEPLITZ check_correctness_eigenvalues_toeplitz_complex_double_f
#  endif
#endif

#define assert_elpa_ok(x) assert(x == ELPA_OK)

#ifdef HAVE_64BIT_INTEGER_MATH_SUPPORT
#define TEST_C_INT_TYPE_PTR long int*
#define C_INT_TYPE_PTR long int*
#define TEST_C_INT_TYPE long int
#define C_INT_TYPE long int
#else
#define TEST_C_INT_TYPE_PTR int*
#define C_INT_TYPE_PTR int*
#define TEST_C_INT_TYPE int
#define C_INT_TYPE int
#endif

#ifdef HAVE_64BIT_INTEGER_MPI_SUPPORT
#define TEST_C_INT_MPI_TYPE_PTR long int*
#define C_INT_MPI_TYPE_PTR long int*
#define TEST_C_INT_MPI_TYPE long int
#define C_INT_MPI_TYPE long int
#else
#define TEST_C_INT_MPI_TYPE_PTR int*
#define C_INT_MPI_TYPE_PTR int*
#define TEST_C_INT_MPI_TYPE int
#define C_INT_MPI_TYPE int
#endif

#define TEST_GPU  0
#if (TEST_NVIDIA_GPU == 1) || (TEST_AMD_GPU == 1) || (TEST_INTEL_GPU == 1) || (TEST_INTEL_GPU_OPENMP == 1) || (TEST_INTEL_GPU_SYCL == 1)
#undef TEST_GPU
#define TEST_GPU  1
#endif

#if (TEST_GPU == 1)
#include "../shared/GPU/test_gpu_vendor_agnostic_layerFunctions.h"
#include "../shared/GPU/test_gpu_vendor_agnostic_layerVariables.h"
#endif

#include "test/shared/generated.h"


int main(int argc, char** argv) {
   /* matrix dimensions */
   C_INT_TYPE na, nev, nblk;

   /* mpi */
   C_INT_TYPE myid, nprocs;
   C_INT_MPI_TYPE myidMPI, nprocsMPI;
   C_INT_TYPE na_cols, na_rows; // local matrix size
   C_INT_TYPE np_cols, np_rows; // number of MPI processes per column/row
   C_INT_TYPE my_prow, my_pcol; // local MPI task position (my_prow, my_pcol) in the grid (0..np_cols -1, 0..np_rows -1)
   C_INT_TYPE mpi_comm;
   C_INT_MPI_TYPE provided_mpi_thread_level;

   /* blacs */
   C_INT_TYPE my_blacs_ctxt, sc_desc[9], info, blacs_ok;

   /* gpu */
   C_INT_TYPE successGPU;
#if TEST_GPU_SET_ID == 1
   C_INT_TYPE  gpuID = 0;
#endif

   /* The Matrix */
   MATRIX_TYPE *a, *as;
#if defined(TEST_HERMITIAN_MULTIPLY_FULL)
   MATRIX_TYPE *b, *c;
#endif
#if defined(TEST_GENERALIZED_EIGENPROBLEM)
   MATRIX_TYPE *b, *bs;
#endif
   /* eigenvectors */
   MATRIX_TYPE *z;
   /* eigenvalues */
   EV_TYPE *ev;

#if TEST_GPU_DEVICE_POINTER_API == 1
   MATRIX_TYPE *a_dev, *z_dev, *b_dev, *c_dev;
   EV_TYPE *ev_dev;
#endif

#if defined(TEST_MATRIX_TOEPLITZ) // only for solve_triadiagonal
   EV_TYPE *d, *sd, *ds, *sds;
   EV_TYPE diagonalElement, subdiagonalElement;
#endif

   C_INT_TYPE error, status;
   int error_elpa;

   elpa_t handle;

   int  value;
   int is_skewsymmetric;

#ifdef WITH_MPI
#ifndef WITH_OPENMP_TRADITIONAL
   MPI_Init(&argc, &argv);
#else
   MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_mpi_thread_level);

   if (provided_mpi_thread_level != MPI_THREAD_MULTIPLE) {
     fprintf(stderr, "MPI ERROR: MPI_THREAD_MULTIPLE is not provided on this system\n");
     MPI_Finalize();
     exit(77);
   }
#endif

   MPI_Comm_size(MPI_COMM_WORLD, &nprocsMPI);
   nprocs = (C_INT_TYPE) nprocsMPI;
   MPI_Comm_rank(MPI_COMM_WORLD, &myidMPI);
   myid = (C_INT_TYPE) myidMPI;

#else
   nprocs = 1;
   myid = 0;
#endif

#if defined(HAVE_64BIT_INTEGER_MPI_SUPPORT) || defined(HAVE_64BIT_INTEGER_MATH_SUPPORT) || defined(HAVE_64BIT_INTEGER_SUPPORT)
#ifdef WITH_MPI
   MPI_Finalize();
#endif
   return 77;
#endif

#ifdef WITH_CUDA_AWARE_MPI
#if TEST_NVIDIA_GPU != 1
#ifdef WITH_MPI
   MPI_Finalize();
#endif
   return 77;
#endif
#ifdef TEST_COMPLEX
#ifdef WITH_MPI
   MPI_Finalize();
#endif
   return 77;
#endif
#endif

// pointer API is tested for NVIDIA, AMD, and INTEL
#if TEST_GPU_DEVICE_POINTER_API == 1 && TEST_NVIDIA_GPU == 0 && TEST_AMD_GPU == 0 && TEST_INTEL_GPU_OPENMP == 0 && TEST_INTEL_GPU_SYCL == 0
#ifdef WITH_MPI
   MPI_Finalize();
#endif
   return 77;
#endif

// we are switching off testing explicit name in gpu version in C-tests, since SYCL on CPU can't discriminate between a and a_dev
#if TEST_INTEL_GPU_SYCL==1 && defined(TEST_EXPLICIT_NAME) && TEST_GPU_DEVICE_POINTER_API==1
#ifdef WITH_MPI
   MPI_Finalize();
#endif
   return 77;
#endif

   if (argc == 4) {
     na = atoi(argv[1]);
     nev = atoi(argv[2]);
     nblk = atoi(argv[3]);
   } else {
#ifdef __cplusplus
     na = 100;
     nev = 50;
     nblk = 4;
#else
     na = 500;
     nev = 250;
     nblk = 16;
#endif
   }

   for (np_cols = (C_INT_TYPE) sqrt((double) nprocs); np_cols > 1; np_cols--) {
     if (nprocs % np_cols == 0) {
       break;
     }
   }

   np_rows = nprocs/np_cols;

   if (myid == 0){
       printf("Matrix size: %i\n", na);
       printf("Num eigenvectors: %i\n", nev);
       printf("Blocksize: %i\n", nblk);
#ifdef WITH_MPI
       printf("Num MPI proc: %i\n", nprocs);
       printf("Number of processor rows=%i, cols=%i, total=%i\n", np_rows, np_cols, nprocs);
#endif
   }

   /* set up blacs */
   /* convert communicators before */
#ifdef WITH_MPI
   mpi_comm = MPI_Comm_c2f(MPI_COMM_WORLD);
#else
   mpi_comm = 0;
#endif
   set_up_blacsgrid_f(mpi_comm, np_rows, np_cols, 'C', &my_blacs_ctxt, &my_prow, &my_pcol);
   set_up_blacs_descriptor_f(na, nblk, my_prow, my_pcol, np_rows, np_cols, &na_rows, &na_cols, sc_desc, my_blacs_ctxt, &info, &blacs_ok);

   if (blacs_ok == 0) {
     if (myid == 0) {
       printf("Setting up the blacsgrid failed. Aborting...");
     }
#ifdef WITH_MPI
     MPI_Finalize();
#endif
     abort();
   }


   /* Allocate the matrices needed for elpa */

   a  = (MATRIX_TYPE *) calloc(na_rows*na_cols, sizeof(MATRIX_TYPE));
   z  = (MATRIX_TYPE *) calloc(na_rows*na_cols, sizeof(MATRIX_TYPE));
   as = (MATRIX_TYPE *) calloc(na_rows*na_cols, sizeof(MATRIX_TYPE));
   ev = (EV_TYPE *) calloc(na, sizeof(EV_TYPE));

   is_skewsymmetric=0;
   PREPARE_MATRIX_RANDOM(na, myid, na_rows, na_cols, sc_desc, a, z, as, is_skewsymmetric);

#ifdef TEST_HERMITIAN_MULTIPLY_FULL
	b  = (MATRIX_TYPE *) calloc(na_rows*na_cols, sizeof(MATRIX_TYPE));
	c  = (MATRIX_TYPE *) calloc(na_rows*na_cols, sizeof(MATRIX_TYPE));
	PREPARE_MATRIX_RANDOM(na, myid, na_rows, na_cols, sc_desc, b, z, c, is_skewsymmetric); // b=c
#endif

#if defined(TEST_GENERALIZED_EIGENPROBLEM)
	b  = (MATRIX_TYPE *) calloc(na_rows*na_cols, sizeof(MATRIX_TYPE));
	bs = (MATRIX_TYPE *) calloc(na_rows*na_cols, sizeof(MATRIX_TYPE));
	PREPARE_MATRIX_RANDOM_SPD(na, myid, na_rows, na_cols, sc_desc, b, z, bs, nblk, np_rows, np_cols, my_prow, my_pcol);
#endif

#if defined(TEST_CHOLESKY)
	PREPARE_MATRIX_RANDOM_SPD(na, myid, na_rows, na_cols, sc_desc, a, z, as, nblk, np_rows, np_cols, my_prow, my_pcol);
#endif

#if defined(TEST_EIGENVALUES)
   PREPARE_MATRIX_ANALYTIC(na, a, na_rows, na_cols, nblk, myid, np_rows, np_cols, my_prow, my_pcol);
   memcpy(as, a, na_rows * na_cols * sizeof(MATRIX_TYPE));
#endif

#if defined(TEST_SOLVE_TRIDIAGONAL)
   d   = (EV_TYPE *) calloc(na, sizeof(EV_TYPE));
   ds  = (EV_TYPE *) calloc(na, sizeof(EV_TYPE));
   sd  = (EV_TYPE *) calloc(na, sizeof(EV_TYPE));
   sds = (EV_TYPE *) calloc(na, sizeof(EV_TYPE));

   diagonalElement = 0.45;
   subdiagonalElement = 0.78;
   PREPARE_MATRIX_TOEPLITZ(na, diagonalElement, subdiagonalElement,
               d, sd, ds, sds, a, as, nblk, na_rows, na_cols, np_rows,
               np_cols, my_prow, my_pcol);
#endif

   if (elpa_init(CURRENT_API_VERSION) != ELPA_OK) {
     fprintf(stderr, "Error: ELPA API version not supported");
     exit(1);
   }

   handle = elpa_allocate(&error_elpa);
   //assert_elpa_ok(error_elpa);

   /* Set parameters */

   elpa_set(handle, "na", (int) na, &error_elpa);
   assert_elpa_ok(error_elpa);

   elpa_set(handle, "nev", (int) nev, &error_elpa);
   assert_elpa_ok(error_elpa);

   if (myid == 0) {
     printf("Setting the matrix parameters na=%d, nev=%d \n",na,nev);
   }
   elpa_set(handle, "local_nrows", (int) na_rows, &error_elpa);
   assert_elpa_ok(error_elpa);

   elpa_set(handle, "local_ncols", (int) na_cols, &error_elpa);
   assert_elpa_ok(error_elpa);

   elpa_set(handle, "nblk", (int) nblk, &error_elpa);
   assert_elpa_ok(error_elpa);

#ifdef WITH_MPI
   elpa_set(handle, "mpi_comm_parent", (int) (MPI_Comm_c2f(MPI_COMM_WORLD)), &error_elpa);
   assert_elpa_ok(error_elpa);

   elpa_set(handle, "process_row", (int) my_prow, &error_elpa);
   assert_elpa_ok(error_elpa);

   elpa_set(handle, "process_col", (int) my_pcol, &error_elpa);
   assert_elpa_ok(error_elpa);
#endif

#ifdef TEST_GENERALIZED_EIGENPROBLEM
   elpa_set(handle, "blacs_context", (int) my_blacs_ctxt, &error_elpa);
   assert_elpa_ok(error_elpa);
#endif
   elpa_set(handle, "timings", 1, &error_elpa);
   assert_elpa_ok(error_elpa);

   /* Setup */
   assert_elpa_ok(elpa_setup(handle));

   /* Set solver and ELPA2 kernel */

#ifdef TEST_SOLVER_1STAGE
   elpa_set(handle, "solver", ELPA_SOLVER_1STAGE, &error_elpa);
#else
   elpa_set(handle, "solver", ELPA_SOLVER_2STAGE, &error_elpa);
#endif
   assert_elpa_ok(error_elpa);

#if TEST_NVIDIA_GPU == 1
   elpa_set(handle, "nvidia-gpu", TEST_GPU, &error_elpa);
   assert_elpa_ok(error_elpa);
#endif

#if TEST_AMD_GPU == 1
   elpa_set(handle, "amd-gpu", TEST_GPU, &error_elpa);
   assert_elpa_ok(error_elpa);
#endif

#if TEST_INTEL_GPU == 1 || TEST_INTEL_GPU_OPENMP == 1  || TEST_INTEL_GPU_SYCL == 1
   elpa_set(handle, "intel-gpu", TEST_GPU, &error_elpa);
   assert_elpa_ok(error_elpa);
#endif

#if defined(TEST_SOLVE_2STAGE) && defined(TEST_KERNEL)
   kernel = TEST_KERNEL
#ifdef TEST_COMPLEX
   elpa_set(handle, "complex_kernel", kernel, &error_elpa);
#else
   elpa_set(handle, "real_kernel", kernel, &error_elpa);
#endif

#ifdef TEST_REAL
#if (TEST_NVIDIA_GPU == 1)
#if WITH_NVIDIA_SM80_GPU_KERNEL == 1
     kernel = ELPA_2STAGE_REAL_NVIDIA_SM80_GPU
#else
     kernel = ELPA_2STAGE_REAL_NVIDIA_GPU
#endif
#endif /* TEST_NVIDIA_GPU */

#if (TEST_AMD_GPU == 1)
     kernel = ELPA_2STAGE_REAL_AMD_GPU
#endif

#if (TEST_INTEL_GPU == 1) || (TEST_INTEL_GPU_OPENMP == 1) || (TEST_INTEL_GPU_SYCL == 1)
     kernel = ELPA_2STAGE_REAL_INTEL_GPU_SYCL
#endif
#endif /* TEST_REAL */
#ifdef TEST_COMPLEX
#if (TEST_NVIDIA_GPU == 1)
     kernel = ELPA_2STAGE_COMPLEX_NVIDIA_GPU
#endif
#if (TEST_AMD_GPU == 1)
     kernel = ELPA_2STAGE_COMPLEX_AMD_GPU
#endif
#if (TEST_INTEL_GPU == 1) || (TEST_INTEL_GPU_OPENMP == 1) || (TEST_INTEL_GPU_SYCL == 1)
     kernel = ELPA_2STAGE_COMPLEX_INTEL_GPU_SYCL
#endif
#endif /* TEST_COMPLEX */

#ifdef TEST_COMPLEX
   elpa_set(handle, "complex_kernel", kernel, &error_elpa);
#else
   elpa_set(handle, "real_kernel", kernel, &error_elpa);
#endif

   assert_elpa_ok(error_elpa);
#endif /* defined(TEST_SOLVE_2STAGE) && defined(TEST_KERNEL) */

#if (TEST_GPU_SET_ID == 1) && (TEST_INTEL_GPU == 0) && (TEST_INTEL_GPU_OPENMP == 0) && (TEST_INTEL_GPU_SYCL == 0)
   int numberOfDevices;
   gpuGetDeviceCount(&numberOfDevices);
   printf("Number of Devices found: %d\n\n", numberOfDevices);
   gpuID = myid%numberOfDevices;
   printf("gpuID: %i\n", gpuID);
   elpa_set(handle, "use_gpu_id", gpuID, &error_elpa);
   assert_elpa_ok(error_elpa);
#endif

#if (TEST_NVIDIA_GPU == 1) || (TEST_AMD_GPU == 1) || (TEST_INTEL_GPU == 1) || (TEST_INTEL_GPU_OPENMP == 1) || (TEST_INTEL_GPU_SYCL == 1)
   assert_elpa_ok(elpa_setup_gpu(handle));
#endif

#if TEST_GPU_DEVICE_POINTER_API == 1
   set_gpu_parameters();

#if TEST_INTEL_GPU_SYCL == 1 /* temporary fix for SYCL on CPU */
   int numberOfDevices=0;
   successGPU = syclGetCpuCount(numberOfDevices);
   if (!successGPU){
      printf("Error in syclGetCpuCount\n");
      exit(1);
      }
#endif

   // Set device
   successGPU = gpuSetDevice(gpuID);
   if (!successGPU){
      printf("Error in gpuSetDevice\n");
      exit(1);
      }

#if defined(TEST_EIGENVECTORS)
   // malloc
   successGPU = gpuMalloc((intptr_t *) &a_dev , na_rows*na_cols*sizeof(MATRIX_TYPE));
   if (!successGPU){
      fprintf(stderr, "Error in gpuMalloc(a_dev)\n");
      exit(1);
      }

   successGPU = gpuMalloc((intptr_t *) &z_dev , na_rows*na_cols*sizeof(MATRIX_TYPE));
   if (!successGPU){
      fprintf(stderr, "Error in gpuMalloc(z_dev)\n");
      exit(1);
      }

   successGPU = gpuMalloc((intptr_t *) &ev_dev, na*sizeof(EV_TYPE));
   if (!successGPU){
      fprintf(stderr, "Error in gpuMalloc(ev_dev)\n");
      exit(1);
      }

   // copy
   successGPU = gpuMemcpy((intptr_t *) a_dev, (intptr_t *) a, na_rows*na_cols*sizeof(MATRIX_TYPE), gpuMemcpyHostToDevice);
   if (!successGPU){
      fprintf(stderr, "Error in gpuMemcpy(a_dev, a)\n");
      exit(1);
      }
#endif /* TEST_EIGENVECTORS */

#if defined(TEST_EIGENVALUES)
   successGPU = gpuMalloc((intptr_t *) &a_dev , na_rows*na_cols*sizeof(MATRIX_TYPE));
   if (!successGPU){
      fprintf(stderr, "Error in gpuMalloc(a_dev)\n");
      exit(1);
      }

   successGPU = gpuMalloc((intptr_t *) &ev_dev, na*sizeof(EV_TYPE));
   if (!successGPU){
      fprintf(stderr, "Error in gpuMalloc(ev_dev)\n");
      exit(1);
      }

   // copy
   successGPU = gpuMemcpy((intptr_t *) a_dev, (intptr_t *) a, na_rows*na_cols*sizeof(MATRIX_TYPE), gpuMemcpyHostToDevice);
   if (!successGPU){
      fprintf(stderr, "Error in gpuMemcpy(a_dev, a)\n");
      exit(1);
      }
#endif /* TEST_EIGENVALUES */

#if defined(TEST_CHOLESKY)
   elpa_set(handle, "gpu_cholesky", 1, &error_elpa);
   assert_elpa_ok(error_elpa);

   successGPU = gpuMalloc((intptr_t *) &a_dev , na_rows*na_cols*sizeof(MATRIX_TYPE));
   if (!successGPU){
      fprintf(stderr, "Error in gpuMalloc(a_dev)\n");
      exit(1);
      }

   // copy
   successGPU = gpuMemcpy((intptr_t *) a_dev, (intptr_t *) a, na_rows*na_cols*sizeof(MATRIX_TYPE), gpuMemcpyHostToDevice);
   if (!successGPU){
      fprintf(stderr, "Error in gpuMemcpy(a_dev, a)\n");
      exit(1);
      }
#endif /* TEST_CHOLESKY */

#if defined(TEST_HERMITIAN_MULTIPLY_FULL)
   elpa_set(handle, "gpu_hermitian_multiply", 1, &error_elpa);
   assert_elpa_ok(error_elpa);

   successGPU = gpuMalloc((intptr_t *) &a_dev , na_rows*na_cols*sizeof(MATRIX_TYPE));
   if (!successGPU){
      fprintf(stderr, "Error in gpuMalloc(a_dev)\n");
      exit(1);
      }

   successGPU = gpuMalloc((intptr_t *) &b_dev , na_rows*na_cols*sizeof(MATRIX_TYPE));
   if (!successGPU){
      fprintf(stderr, "Error in gpuMalloc(b_dev)\n");
      exit(1);
      }

   successGPU = gpuMalloc((intptr_t *) &c_dev , na_rows*na_cols*sizeof(MATRIX_TYPE));
   if (!successGPU){
      fprintf(stderr, "Error in gpuMalloc(c_dev)\n");
      exit(1);
      }

   // copy
   successGPU = gpuMemcpy((intptr_t *) a_dev, (intptr_t *) a, na_rows*na_cols*sizeof(MATRIX_TYPE), gpuMemcpyHostToDevice);
   if (!successGPU){
      fprintf(stderr, "Error in gpuMemcpy(a_dev, a)\n");
      exit(1);
      }

   successGPU = gpuMemcpy((intptr_t *) b_dev, (intptr_t *) b, na_rows*na_cols*sizeof(MATRIX_TYPE), gpuMemcpyHostToDevice);
   if (!successGPU){
      fprintf(stderr, "Error in gpuMemcpy(b_dev, b)\n");
      exit(1);
      }

   successGPU = gpuMemcpy((intptr_t *) c_dev, (intptr_t *) c, na_rows*na_cols*sizeof(MATRIX_TYPE), gpuMemcpyHostToDevice);
   if (!successGPU){
      fprintf(stderr, "Error in gpuMemcpy(c_dev, c)\n");
      exit(1);
      }
#endif /* TEST_HERMITIAN_MULTIPLY_FULL */

#endif /* TEST_GPU_DEVICE_POINTER_API == 1 && TEST_NVIDIA_GPU == 1 */

   elpa_get(handle, "solver", &value, &error_elpa);
   if (myid == 0) {
     printf("Solver is set to %d \n", value);
   }

  //_____________________________________________________________________________________________________________________
	/* The actual solve step */

#if defined(TEST_EIGENVECTORS)
#if TEST_QR_DECOMPOSITION == 1
     elpa_timer_start(handle, (char*) "elpa_eigenvectors_qr()");
#else
     elpa_timer_start(handle, (char*) "elpa_eigenvectors()");
#endif
#endif /* TEST_EIGENVECTORS */

#if defined(TEST_EIGENVECTORS)
#if defined(TEST_EXPLICIT_NAME)

#if defined(TEST_REAL)
#if defined(TEST_DOUBLE)
#if TEST_GPU_DEVICE_POINTER_API == 1
     elpa_eigenvectors_double(handle, a_dev, ev_dev, z_dev, &error_elpa);
     assert_elpa_ok(error_elpa);
#else
     elpa_eigenvectors_double(handle, a    , ev    , z    , &error_elpa);
     assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_DOUBLE */
#if defined(TEST_SINGLE)
#if TEST_GPU_DEVICE_POINTER_API == 1
     elpa_eigenvectors_float (handle, a_dev, ev_dev, z_dev, &error_elpa);
     assert_elpa_ok(error_elpa);
#else
     elpa_eigenvectors_float (handle, a    , ev    , z    , &error_elpa);
     assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_SINGLE */
#endif /* TEST_REAL */

#if defined(TEST_COMPLEX)
#if defined(TEST_DOUBLE)
#if TEST_GPU_DEVICE_POINTER_API == 1
     elpa_eigenvectors_double_complex(handle, a_dev, ev_dev, z_dev, &error_elpa);
     assert_elpa_ok(error_elpa);
#else
     elpa_eigenvectors_double_complex(handle, a    , ev    , z    , &error_elpa);
     assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_DOUBLE */
#if defined(TEST_SINGLE)
#if TEST_GPU_DEVICE_POINTER_API == 1
     elpa_eigenvectors_float_complex (handle, a_dev, ev_dev, z_dev, &error_elpa);
     assert_elpa_ok(error_elpa);
#else
     elpa_eigenvectors_float_complex (handle, a    , ev    , z    , &error_elpa);
     assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_SINGLE */
#endif /* TEST_COMPLEX */

#else /* TEST_EXPLICIT_NAME */
	   elpa_eigenvectors(handle, a, ev, z, &error_elpa);
     assert_elpa_ok(error_elpa);
#endif /* TEST_EXPLICIT_NAME */
#if TEST_QR_DECOMPOSITION == 1
     elpa_timer_stop(handle, (char*) "elpa_eigenvectors_qr()");
#else
     elpa_timer_stop(handle, (char*) "elpa_eigenvectors()");
#endif
#endif /* TEST_EIGENVECTORS */


#if defined(TEST_EIGENVALUES)
   elpa_timer_start(handle, (char*) "elpa_eigenvalues()");
#if defined(TEST_EXPLICIT_NAME)

#if defined(TEST_REAL)
#if defined(TEST_DOUBLE)
#if TEST_GPU_DEVICE_POINTER_API == 1
   elpa_eigenvalues_double(handle, a_dev, ev_dev, &error_elpa);
   assert_elpa_ok(error_elpa);
#else
   elpa_eigenvalues_double(handle, a    , ev    , &error_elpa);
   assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_DOUBLE */
#if defined(TEST_SINGLE)
#if TEST_GPU_DEVICE_POINTER_API == 1
     elpa_eigenvalues_float (handle, a_dev, ev_dev, &error_elpa);
     assert_elpa_ok(error_elpa);
#else
     elpa_eigenvalues_float (handle, a    , ev    , &error_elpa);
     assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_SINGLE */
#endif /* TEST_REAL */

#if defined(TEST_COMPLEX)
#if defined(TEST_DOUBLE)
#if TEST_GPU_DEVICE_POINTER_API == 1
     elpa_eigenvalues_double_complex(handle, a_dev, ev_dev, &error_elpa);
     assert_elpa_ok(error_elpa);
#else
     elpa_eigenvalues_double_complex(handle, a    , ev    , &error_elpa);
     assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_DOUBLE */
#if defined(TEST_SINGLE)
#if TEST_GPU_DEVICE_POINTER_API == 1
     elpa_eigenvalues_float_complex (handle, a_dev, ev_dev, &error_elpa);
     assert_elpa_ok(error_elpa);
#else
     elpa_eigenvalues_float_complex (handle, a    , ev    , &error_elpa);
     assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_SINGLE */
#endif /* TEST_COMPLEX */

#else /* TEST_EXPLICIT_NAME */
	 elpa_eigenvalues(handle, a, ev, &error_elpa);
   assert_elpa_ok(error_elpa);
#endif /* TEST_EXPLICIT_NAME */
   elpa_timer_stop(handle, (char*) "elpa_eigenvalues()");
#endif /* TEST_EIGENVALUES */

#if defined(TEST_CHOLESKY)
   elpa_timer_start(handle, (char*) "elpa_cholesky()");
#if defined(TEST_EXPLICIT_NAME)

#if defined(TEST_REAL)
#if defined(TEST_DOUBLE)
#if TEST_GPU_DEVICE_POINTER_API == 1
   elpa_cholesky_double(handle, a_dev, &error_elpa);
   assert_elpa_ok(error_elpa);
#else
   elpa_cholesky_double(handle, a    , &error_elpa);
   assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_DOUBLE */
#if defined(TEST_SINGLE)
#if TEST_GPU_DEVICE_POINTER_API == 1
     elpa_cholesky_float (handle, a_dev, &error_elpa);
     assert_elpa_ok(error_elpa);
#else
     elpa_cholesky_float (handle, a    , &error_elpa);
     assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_SINGLE */
#endif /* TEST_REAL */

#if defined(TEST_COMPLEX)
#if defined(TEST_DOUBLE)
#if TEST_GPU_DEVICE_POINTER_API == 1
     elpa_cholesky_double_complex(handle, a_dev, &error_elpa);
     assert_elpa_ok(error_elpa);
#else
     elpa_cholesky_double_complex(handle, a    , &error_elpa);
     assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_DOUBLE */
#if defined(TEST_SINGLE)
#if TEST_GPU_DEVICE_POINTER_API == 1
     elpa_cholesky_float_complex (handle, a_dev, &error_elpa);
     assert_elpa_ok(error_elpa);
#else
     elpa_cholesky_float_complex (handle, a    , &error_elpa);
     assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_SINGLE */
#endif /* TEST_COMPLEX */

#else /* TEST_EXPLICIT_NAME */
   elpa_cholesky(handle, a, &error_elpa);
   assert_elpa_ok(error_elpa);

#endif /* TEST_EXPLICIT_NAME */
   elpa_timer_stop(handle, (char*) "elpa_cholesky()");
#endif /* TEST_CHOLESKY */

#if defined(TEST_HERMITIAN_MULTIPLY_FULL)
   elpa_timer_start(handle, (char*) "elpa_hermitian_multiply()");
#if defined(TEST_EXPLICIT_NAME)

#if defined(TEST_REAL)
#if defined(TEST_DOUBLE)
#if TEST_GPU_DEVICE_POINTER_API == 1
	 elpa_hermitian_multiply_double(handle, 'F', 'F', na, a_dev, b_dev, na_rows, na_cols, c_dev, na_rows, na_cols, &error_elpa);
   assert_elpa_ok(error_elpa);
#else
	 elpa_hermitian_multiply_double(handle, 'F', 'F', na, a, b, na_rows, na_cols, c, na_rows, na_cols, &error_elpa);
   assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_DOUBLE */
#if defined(TEST_SINGLE)
#if TEST_GPU_DEVICE_POINTER_API == 1
	 elpa_hermitian_multiply_float(handle, 'F', 'F', na, a_dev, b_dev, na_rows, na_cols, c_dev, na_rows, na_cols, &error_elpa);
   assert_elpa_ok(error_elpa);
#else
	 elpa_hermitian_multiply_float(handle, 'F', 'F', na, a, b, na_rows, na_cols, c, na_rows, na_cols, &error_elpa);
   assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_SINGLE */
#endif /* TEST_REAL */

#if defined(TEST_COMPLEX)
#if defined(TEST_DOUBLE)
#if TEST_GPU_DEVICE_POINTER_API == 1
	 elpa_hermitian_multiply_double_complex(handle, 'F', 'F', na, a_dev, b_dev, na_rows, na_cols, c_dev, na_rows, na_cols, &error_elpa);
   assert_elpa_ok(error_elpa);
#else
	 elpa_hermitian_multiply_double_complex(handle, 'F', 'F', na, a, b, na_rows, na_cols, c, na_rows, na_cols, &error_elpa);
   assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_DOUBLE */
#if defined(TEST_SINGLE)
#if TEST_GPU_DEVICE_POINTER_API == 1
	 elpa_hermitian_multiply_float_complex(handle, 'F', 'F', na, a_dev, b_dev, na_rows, na_cols, c_dev, na_rows, na_cols, &error_elpa);
   assert_elpa_ok(error_elpa);
#else
	 elpa_hermitian_multiply_float_complex(handle, 'F', 'F', na, a, b, na_rows, na_cols, c, na_rows, na_cols, &error_elpa);
   assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_SINGLE */
#endif /* TEST_COMPLEX */

#else /* TEST_EXPLICIT_NAME */
	 elpa_hermitian_multiply(handle, 'F', 'F', na, a, b, na_rows, na_cols, c, na_rows, na_cols, &error_elpa);
   assert_elpa_ok(error_elpa);
#endif /* TEST_EXPLICIT_NAME */
   elpa_timer_stop(handle, (char*) "elpa_hermitian_multiply()");
#endif /* TEST_HERMITIAN_MULTIPLY_FULL */

#if defined(TEST_GENERALIZED_EIGENPROBLEM)
     elpa_timer_start(handle, (char*) "elpa_generalized_eigenvectors()");
#if defined(TEST_GENERALIZED_DECOMP_EIGENPROBLEM)
     elpa_timer_start(handle, "is_already_decomposed=.false.");
#endif
     elpa_generalized_eigenvectors(handle, a, b, ev, z, 0, &error_elpa);
     assert_elpa_ok(error_elpa);
#if defined(TEST_GENERALIZED_DECOMP_EIGENPROBLEM)
     elpa_timer_stop(handle, (char*) "is_already_decomposed=.false.");
     memcpy(a, as, na_rows * na_cols * sizeof(MATRIX_TYPE)); // so that the problem can be solved again
     elpa_timer_start(handle, (char*) "is_already_decomposed=.true.");
     elpa_generalized_eigenvectors(handle, a, b, ev, z, 1, &error_elpa);
     assert_elpa_ok(error_elpa);
     elpa_timer_stop(handle, (char*) "is_already_decomposed=.true.");
#endif /* TEST_GENERALIZED_DECOMP_EIGENPROBLEM */
     elpa_timer_stop(handle, (char*) "elpa_generalized_eigenvectors()");
#endif /* TEST_GENERALIZED_EIGENPROBLEM */

#if defined(TEST_SOLVE_TRIDIAGONAL)
     elpa_solve_tridiagonal(handle, d, sd, z, &error_elpa);
     assert_elpa_ok(error_elpa);
     memcpy(ev, d, na*sizeof(EV_TYPE));
#endif

#if defined(TEST_EIGENVECTORS)
     if (myid == 0) {
#if TEST_QR_DECOMPOSITION == 1
       elpa_print_times(handle, (char*) "elpa_eigenvectors_qr()");
#else
       elpa_print_times(handle, (char*) "elpa_eigenvectors()");
#endif
     }
#endif /* TEST_EIGENVECTORS */

//_____________________________________________________________________________________________________________________
   /* TEST_GPU_DEVICE_POINTER_API case: copy for testing from device to host, deallocate device pointers */

#if TEST_GPU_DEVICE_POINTER_API == 1

#if defined(TEST_EIGENVECTORS)
   // copy for testing
   successGPU = gpuMemcpy((intptr_t *) z , (intptr_t *) z_dev , na_rows*na_cols*sizeof(MATRIX_TYPE), gpuMemcpyDeviceToHost);
   if (!successGPU){
      fprintf(stderr, "Error in gpuMemcpy(z, z_dev)\n");
      exit(1);
      }

   successGPU = gpuMemcpy((intptr_t *) ev, (intptr_t *) ev_dev, na*sizeof(EV_TYPE)    , gpuMemcpyDeviceToHost);
   if (!successGPU){
      fprintf(stderr, "Error in gpuMemcpy(ev, ev_dev)\n");
      exit(1);
      }

   // and deallocate device pointer
   //successGPU = gpuFree((void *) a_dev);
   successGPU = gpuFree((intptr_t *) a_dev);
   if (!successGPU){
      fprintf(stderr, "Error in gpuFree(a_dev)\n");
      exit(1);
      }

   successGPU = gpuFree((intptr_t *) z_dev);
   if (!successGPU){
      fprintf(stderr, "Error in gpuFree(z_dev)\n");
      exit(1);
      }

   successGPU = gpuFree((intptr_t *) ev_dev);
   if (!successGPU){
      fprintf(stderr, "Error in gpuFree(ev_dev)\n");
      exit(1);
      }
#endif /* defined(TEST_EIGENVECTORS) */

#if defined(TEST_EIGENVALUES)
   // copy for testing
   successGPU = gpuMemcpy((intptr_t *) ev, (intptr_t *) ev_dev, na*sizeof(EV_TYPE)    , gpuMemcpyDeviceToHost);
   if (!successGPU){
      fprintf(stderr, "Error in gpuMemcpy(ev, ev_dev)\n");
      exit(1);
      }

   // and deallocate device pointer
   successGPU = gpuFree((intptr_t *) a_dev);
   if (!successGPU){
      fprintf(stderr, "Error in gpuFree(a_dev)\n");
      exit(1);
      }

   successGPU = gpuFree((intptr_t *) ev_dev);
   if (!successGPU){
      fprintf(stderr, "Error in gpuFree(ev_dev)\n");
      exit(1);
      }
#endif /* TEST_EIGENVALUES */

#if defined(TEST_CHOLESKY)
   // copy for testing
   successGPU = gpuMemcpy((intptr_t *) a , (intptr_t *) a_dev , na_rows*na_cols*sizeof(MATRIX_TYPE), gpuMemcpyDeviceToHost);
   if (!successGPU){
      fprintf(stderr, "Error in gpuMemcpy(a, a_dev)\n");
      exit(1);
      }

   // and deallocate device pointer
   successGPU = gpuFree((intptr_t *) a_dev);
   if (!successGPU){
      fprintf(stderr, "Error in gpuFree(a_dev)\n");
      exit(1);
      }
#endif /* TEST_CHOLESKY */

#if defined(TEST_HERMITIAN_MULTIPLY_FULL)
   // copy for testing
   successGPU = gpuMemcpy((intptr_t *) a, (intptr_t *) a_dev , na_rows*na_cols*sizeof(MATRIX_TYPE), gpuMemcpyDeviceToHost);
   if (!successGPU){
      fprintf(stderr, "Error in gpuMemcpy(a, a_dev)\n");
      exit(1);
      }

   successGPU = gpuMemcpy((intptr_t *) b, (intptr_t *) b_dev , na_rows*na_cols*sizeof(MATRIX_TYPE), gpuMemcpyDeviceToHost);
   if (!successGPU){
      fprintf(stderr, "Error in gpuMemcpy(b, b_dev)\n");
      exit(1);
      }

   successGPU = gpuMemcpy((intptr_t *) c, (intptr_t *) c_dev , na_rows*na_cols*sizeof(MATRIX_TYPE), gpuMemcpyDeviceToHost);
   if (!successGPU){
      fprintf(stderr, "Error in gpuMemcpy(c, c_dev)\n");
      exit(1);
      }

   // and deallocate device pointers
   successGPU = gpuFree((intptr_t *) a_dev);
   if (!successGPU){
      fprintf(stderr, "Error in gpuFree(a_dev)\n");
      exit(1);
      }

   successGPU = gpuFree((intptr_t *) b_dev);
   if (!successGPU){
      fprintf(stderr, "Error in gpuFree(b_dev)\n");
      exit(1);
      }

   successGPU = gpuFree((intptr_t *) c_dev);
   if (!successGPU){
      fprintf(stderr, "Error in gpuFree(c_dev)\n");
      exit(1);
      }
#endif /* TEST_HERMITIAN_MULTIPLY_FULL */

#endif /* TEST_GPU_DEVICE_POINTER_API == 1 */

   //_____________________________________________________________________________________________________________________
   /* Check the results */

#if defined(TEST_CHOLESKY)
	status = CHECK_CORRECTNESS_CHOLESKY(na, a, as, na_rows, na_cols, sc_desc, myid);
#endif

#if defined(TEST_HERMITIAN_MULTIPLY_FULL)
   status = CHECK_CORRECTNESS_HERMITIAN_MULTIPLY('H', na, a, b, c, na_rows, na_cols, sc_desc, myid);
#endif

#if defined(TEST_EIGENVALUES)
   status = CHECK_CORRECTNESS_ANALYTIC (na, nev, ev, z, na_rows, na_cols, nblk, myid, np_rows, np_cols,
                                           my_prow, my_pcol, 1, 0);
#endif

#if defined(TEST_EIGENVECTORS)
   status = CHECK_CORRECTNESS_EVP_NUMERIC_RESIDUALS(na, nev, na_rows, na_cols, as, z, ev,
                                sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol);
#endif

#if defined(TEST_GENERALIZED_EIGENPROBLEM)
   status = CHECK_CORRECTNESS_EVP_GEN_NUMERIC_RESIDUALS(na, nev, na_rows, na_cols, as, z, ev,
                                sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol, bs);
#endif

#if defined(TEST_SOLVE_TRIDIAGONAL)
   status = CHECK_CORRECTNESS_EIGENVALUES_TOEPLITZ(na, na_rows, na_cols, diagonalElement,
                                subdiagonalElement, ev, z, myid);
#endif

   if (status !=0){
     printf("Test produced an error!\n");
   }
   if (status ==0){
     printf("All ok!\n");
   }

  //_____________________________________________________________________________________________________________________
   /* Deallocate */

   elpa_deallocate(handle, &error_elpa);
   assert_elpa_ok(error_elpa);
   elpa_uninit(&error_elpa);
   assert_elpa_ok(error_elpa);

   free(a);
   free(z);
   free(as);
   free(ev);

#ifdef TEST_HERMITIAN_MULTIPLY_FULL
	free(b);
	free(c);
#endif

#if defined(TEST_GENERALIZED_EIGENPROBLEM)
   free(b);
   free(bs);
#endif

#if defined(TEST_MATRIX_TOEPLITZ)
   free(d);
   free(ds);
   free(sd);
   free(sds);
#endif

#ifdef WITH_MPI
   MPI_Finalize();
#endif

   return status;
}
