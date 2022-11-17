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
#include <complex.h>

#if !(defined(TEST_REAL) ^ defined(TEST_COMPLEX))
//#error "define exactly one of TEST_REAL or TEST_COMPLEX"
#endif

#if !(defined(TEST_SINGLE) ^ defined(TEST_DOUBLE))
//#error "define exactly one of TEST_SINGLE or TEST_DOUBLE"
#endif

#if !(defined(TEST_SOLVER_1STAGE) ^ defined(TEST_SOLVER_2STAGE))
//#error "define exactly one of TEST_SOLVER_1STAGE or TEST_SOLVER_2STAGE"
#endif

#ifdef __cplusplus
#define double_complex std::complex<double>
#define float_complex std::complex<float>
#define Complex_I std::complex<EV_TYPE> (0.0,1.0); 
#else
#define double_complex double complex
#define float_complex float complex
#define Complex_I _Complex_I
#endif

#ifdef TEST_SINGLE
#  define EV_TYPE float
#  ifdef TEST_REAL
#    define MATRIX_TYPE float
#    define PREPARE_MATRIX_RANDOM prepare_matrix_random_real_single_f
#    define PREPARE_MATRIX_RANDOM_SPD prepare_matrix_random_spd_real_single_f
#    define CHECK_CORRECTNESS_EVP_NUMERIC_RESIDUALS check_correctness_evp_numeric_residuals_real_single_f
#    define CHECK_CORRECTNESS_EVP_GEN_NUMERIC_RESIDUALS check_correctness_evp_gen_numeric_residuals_real_single_f
#  else
#    define MATRIX_TYPE float_complex
#    define PREPARE_MATRIX_RANDOM prepare_matrix_random_complex_single_f
#    define PREPARE_MATRIX_RANDOM_SPD prepare_matrix_random_spd_complex_single_f
#    define CHECK_CORRECTNESS_EVP_NUMERIC_RESIDUALS check_correctness_evp_numeric_residuals_complex_single_f
#    define CHECK_CORRECTNESS_EVP_GEN_NUMERIC_RESIDUALS check_correctness_evp_gen_numeric_residuals_complex_single_f
#  endif
#else
#  define EV_TYPE double
#  ifdef TEST_REAL
#    define MATRIX_TYPE double
#    define PREPARE_MATRIX_RANDOM prepare_matrix_random_real_double_f
#    define PREPARE_MATRIX_RANDOM_SPD prepare_matrix_random_spd_real_double_f
#    define CHECK_CORRECTNESS_EVP_NUMERIC_RESIDUALS check_correctness_evp_numeric_residuals_real_double_f
#    define CHECK_CORRECTNESS_EVP_GEN_NUMERIC_RESIDUALS check_correctness_evp_gen_numeric_residuals_real_double_f
#  else
#    define MATRIX_TYPE double_complex
#    define PREPARE_MATRIX_RANDOM prepare_matrix_random_complex_double_f
#    define PREPARE_MATRIX_RANDOM_SPD prepare_matrix_random_spd_complex_double_f
#    define CHECK_CORRECTNESS_EVP_NUMERIC_RESIDUALS check_correctness_evp_numeric_residuals_complex_double_f
#    define CHECK_CORRECTNESS_EVP_GEN_NUMERIC_RESIDUALS check_correctness_evp_gen_numeric_residuals_complex_double_f
#  endif
#endif

#ifdef TEST_SINGLE
#define MATRIX_TYPE_COMPLEX float_complex
#else
#define MATRIX_TYPE_COMPLEX double_complex
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
#if (TEST_NVIDIA_GPU == 1) || (TEST_AMD_GPU == 1) || (TEST_INTEL_GPU == 1)
#undef TEST_GPU
#define TEST_GPU  1
#endif


#include "test/shared/generated.h"

int main(int argc, char** argv) {
   /* matrix dimensions */
   C_INT_TYPE na, nev, nblk;

   /* mpi */
   C_INT_TYPE myid, nprocs;
   C_INT_MPI_TYPE myidMPI, nprocsMPI;
   C_INT_TYPE na_cols, na_rows;
   C_INT_TYPE np_cols, np_rows;
   C_INT_TYPE my_prow, my_pcol;
   C_INT_TYPE mpi_comm;
   C_INT_MPI_TYPE provided_mpi_thread_level;

   /* blacs */
   C_INT_TYPE my_blacs_ctxt, sc_desc[9], info, blacs_ok;

   /* The Matrix */
   MATRIX_TYPE *a_skewsymmetric, *as_skewsymmetric, *z_skewsymmetric, *z_skewsymmetric_prepare;
   EV_TYPE *ev_skewsymmetric;
   MATRIX_TYPE_COMPLEX *a_complex, *as_complex, *z_complex;
   EV_TYPE *ev_complex;

   C_INT_TYPE error, status;
   int error_elpa;

   elpa_t handle_skewsymmetric, handle_complex;

   int  value;
   int is_skewsymmetric;
   long int _elements, _nrows, _ncols, i;

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


   /* allocate the matrices needed for elpa */
   a_skewsymmetric  = (MATRIX_TYPE *) calloc(na_rows*na_cols, sizeof(MATRIX_TYPE));
   z_skewsymmetric  = (MATRIX_TYPE *) calloc(na_rows*2*na_cols, sizeof(MATRIX_TYPE));
   z_skewsymmetric_prepare  = (MATRIX_TYPE *) calloc(na_rows*na_cols, sizeof(MATRIX_TYPE));
   as_skewsymmetric = (MATRIX_TYPE *) calloc(na_rows*na_cols, sizeof(MATRIX_TYPE));
   ev_skewsymmetric = (EV_TYPE *) calloc(na, sizeof(EV_TYPE));

   for (_elements=0;_elements<na_rows*na_cols;_elements++){
     a_skewsymmetric[_elements] = 0.;
     z_skewsymmetric_prepare[_elements] = 0.;
     as_skewsymmetric[_elements] = 0.;
   }
   for (_elements=0;_elements<na_rows*2*na_cols;_elements++){
     z_skewsymmetric[_elements] = 0.;
   }
   for (_elements=0;_elements<na;_elements++){
     ev_skewsymmetric[_elements] = 0.;
   }

   is_skewsymmetric=1;
   PREPARE_MATRIX_RANDOM(na, myid, na_rows, na_cols, sc_desc, a_skewsymmetric, z_skewsymmetric_prepare, as_skewsymmetric, is_skewsymmetric);

   //copy to z FORTRAN DATA LAYOUT
   for (_nrows=0; _nrows<na_rows; _nrows++) {
     for (_ncols=0; _ncols<na_cols; _ncols++) {
       z_skewsymmetric[_nrows + na_rows*_ncols] = z_skewsymmetric_prepare[_nrows + na_rows*_ncols];
     }
   }

   free(z_skewsymmetric_prepare);

   // prepare the complex matrix for the "brute force" case
   a_complex  = (MATRIX_TYPE_COMPLEX *) calloc(na_rows*na_cols, sizeof(MATRIX_TYPE_COMPLEX));
   z_complex  = (MATRIX_TYPE_COMPLEX *) calloc(na_rows*na_cols, sizeof(MATRIX_TYPE_COMPLEX));
   as_complex = (MATRIX_TYPE_COMPLEX *) calloc(na_rows*na_cols, sizeof(MATRIX_TYPE_COMPLEX));
   ev_complex = (EV_TYPE *) calloc(na, sizeof(EV_TYPE));

   //for (_elements=0;_elements<na_rows*na_cols;_elements++){
   //  a_complex[_elements] = 0.;
   //  z_complex[_elements] = 0.;
   //  as_complex[_elements] = 0.;
   //}  
   //for (_elements=0;_elements<na;_elements++){
   //  ev_complex[_elements] = 0.;
   //}

   for (_ncols=0;_ncols<na_cols;_ncols++) {
     for (_nrows=0;_nrows<na_rows;_nrows++) {
        a_complex[_nrows + na_rows*_ncols] = 0.0 + a_skewsymmetric[_nrows + na_rows*_ncols]*Complex_I;
     }
   }

   for (_ncols=0;_ncols<na_cols;_ncols++) {
     for (_nrows=0;_nrows<na_rows;_nrows++) {
        z_complex[_nrows + na_rows*_ncols] = a_complex[_nrows + na_rows*_ncols];
        as_complex[_nrows + na_rows*_ncols] = a_complex[_nrows + na_rows*_ncols];
     }
   }


   if (elpa_init(CURRENT_API_VERSION) != ELPA_OK) {
     fprintf(stderr, "Error: ELPA API version not supported");
     exit(1);
   }


   //first set up and solve the brute force problem

   handle_complex = elpa_allocate(&error_elpa);

   /* Set parameters */
   elpa_set(handle_complex, "na", (int) na, &error_elpa);
   assert_elpa_ok(error_elpa);

   elpa_set(handle_complex, "nev", (int) nev, &error_elpa);
   assert_elpa_ok(error_elpa);

   if (myid == 0) {
     printf("Setting the matrix parameters na=%d, nev=%d \n",na,nev);
   }
   elpa_set(handle_complex, "local_nrows", (int) na_rows, &error_elpa);
   assert_elpa_ok(error_elpa);

   elpa_set(handle_complex, "local_ncols", (int) na_cols, &error_elpa);
   assert_elpa_ok(error_elpa);

   elpa_set(handle_complex, "nblk", (int) nblk, &error_elpa);
   assert_elpa_ok(error_elpa);

#ifdef WITH_MPI
   elpa_set(handle_complex, "mpi_comm_parent", (int) (MPI_Comm_c2f(MPI_COMM_WORLD)), &error_elpa);
   assert_elpa_ok(error_elpa);

   elpa_set(handle_complex, "process_row", (int) my_prow, &error_elpa);
   assert_elpa_ok(error_elpa);

   elpa_set(handle_complex, "process_col", (int) my_pcol, &error_elpa);
   assert_elpa_ok(error_elpa);
#endif

   /* Setup */
   assert_elpa_ok(elpa_setup(handle_complex));

   elpa_get(handle_complex, "solver", &value, &error_elpa);
   if (myid == 0) {
     printf("Solver is set to %d \n", value);
   }

   /* Solve EV problem */
   elpa_eigenvectors_a_h_a_dc(handle_complex, a_complex, ev_complex, z_complex, &error_elpa);
   assert_elpa_ok(error_elpa);


#ifdef WITH_MPI
   /* barrier */
   MPI_Barrier(MPI_COMM_WORLD);
#endif


   /* check the results */

#ifdef TEST_SINGLE
   status = check_correctness_evp_numeric_residuals_complex_single_f(na, nev, na_rows, na_cols, as_complex, z_complex, ev_complex, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol);
#else
   status = check_correctness_evp_numeric_residuals_complex_double_f(na, nev, na_rows, na_cols, as_complex, z_complex, ev_complex, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol);
#endif


   if (status !=0){
     printf("The computed EVs are not correct !\n");
   }
   if (status ==0){
     printf("All ok!\n");
   }


   free(a_complex);
   //free(z_complex);
   free(as_complex);
   //free(ev_complex);

#ifdef WITH_MPI
   /* barrier */
   MPI_Barrier(MPI_COMM_WORLD);
#endif
   // now run the skewsymmetric case

   handle_skewsymmetric = elpa_allocate(&error_elpa);

   /* Set parameters */
   elpa_set(handle_skewsymmetric, "na", (int) na, &error_elpa);
   assert_elpa_ok(error_elpa);

   elpa_set(handle_skewsymmetric, "nev", (int) nev, &error_elpa);
   assert_elpa_ok(error_elpa);

   if (myid == 0) {
     printf("Setting the matrix parameters na=%d, nev=%d \n",na,nev);
   }
   elpa_set(handle_skewsymmetric, "local_nrows", (int) na_rows, &error_elpa);
   assert_elpa_ok(error_elpa);

   elpa_set(handle_skewsymmetric, "local_ncols", (int) na_cols, &error_elpa);
   assert_elpa_ok(error_elpa);

   elpa_set(handle_skewsymmetric, "nblk", (int) nblk, &error_elpa);
   assert_elpa_ok(error_elpa);

#ifdef WITH_MPI
   elpa_set(handle_skewsymmetric, "mpi_comm_parent", (int) (MPI_Comm_c2f(MPI_COMM_WORLD)), &error_elpa);
   assert_elpa_ok(error_elpa);

   elpa_set(handle_skewsymmetric, "process_row", (int) my_prow, &error_elpa);
   assert_elpa_ok(error_elpa);

   elpa_set(handle_skewsymmetric, "process_col", (int) my_pcol, &error_elpa);
   assert_elpa_ok(error_elpa);
#endif


   /* Setup */
   assert_elpa_ok(elpa_setup(handle_skewsymmetric));

   elpa_get(handle_skewsymmetric, "solver", &value, &error_elpa);
   if (myid == 0) {
     printf("Solver is set to %d \n", value);
   }
   /* Solve EV problem */
   elpa_skew_eigenvectors(handle_skewsymmetric, a_skewsymmetric, ev_skewsymmetric, z_skewsymmetric, &error_elpa);
   assert_elpa_ok(error_elpa);


#ifdef WITH_MPI
   /* barrier */
   MPI_Barrier(MPI_COMM_WORLD);
#endif


   /* check the results */

   // check eigenvalues
   for (i=0;i<na; i++) {
     if (myid == 0) {
#ifdef TEST_DOUBLE
       if (fabs(ev_complex[i]-ev_skewsymmetric[i])/fabs(ev_complex[i]) > 1e-10) {
#endif
#ifdef TEST_SINGLE
       if (fabs(ev_complex[i]-ev_skewsymmetric[i])/fabs(ev_complex[i]) > 1e-4) {
#endif
	 printf("ev: i= %d,%f,%f\n",i,ev_complex[i],ev_skewsymmetric[i]);
         status = 1;
       }
     }
   }

   for (_elements=0;_elements<na_rows*na_cols;_elements++) {
     z_complex[_elements] = 0.;
   }

   for (_ncols=0;_ncols<na_cols;_ncols++) {
     for (_nrows=0;_nrows<na_rows;_nrows++) {
       z_complex[_nrows + na_rows*_ncols] = z_skewsymmetric[_nrows + na_rows*_ncols] + z_skewsymmetric[_nrows+na_rows*(na_cols+_ncols)]*Complex_I;
     }
   }
#ifdef WITH_MPI
   MPI_Barrier(MPI_COMM_WORLD);
#endif

   status = check_correctness_evp_numeric_residuals_ss_real_double_f(na, nev, na_rows, na_cols, as_skewsymmetric, z_complex, ev_skewsymmetric, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol);

#ifdef WITH_MPI
   MPI_Barrier(MPI_COMM_WORLD);
#endif

   elpa_deallocate(handle_complex, &error_elpa);
   elpa_deallocate(handle_skewsymmetric, &error_elpa);
   elpa_uninit(&error_elpa);

   free(z_complex);
   free(ev_complex);

   free(a_skewsymmetric);
   free(z_skewsymmetric);
   free(as_skewsymmetric);
   free(ev_skewsymmetric);



#ifdef WITH_MPI
   MPI_Finalize();
#endif

   return status;
}
