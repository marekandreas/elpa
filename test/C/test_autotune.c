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

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef WITH_MPI
#include <mpi.h>
#endif
#include <math.h>

#include <elpa/elpa.h>
#include <assert.h>

#if !(defined(TEST_REAL) ^ defined(TEST_COMPLEX))
//#error "define exactly one of TEST_REAL or TEST_COMPLEX"
#endif

#if !(defined(TEST_SINGLE) ^ defined(TEST_DOUBLE))
//#error "define exactly one of TEST_SINGLE or TEST_DOUBLE"
#endif

#if !(defined(TEST_SOLVER_1STAGE) ^ defined(TEST_SOLVER_2STAGE))
//#error "define exactly one of TEST_SOLVER_1STAGE or TEST_SOLVER_2STAGE"
#endif

#ifdef TEST_SINGLE
#  define EV_TYPE float
#  ifdef TEST_REAL
#    define MATRIX_TYPE float
#  else
#    define MATRIX_TYPE complex float
#  endif
#else
#  define EV_TYPE double
#  ifdef TEST_REAL
#    define MATRIX_TYPE double
#  else
#    define MATRIX_TYPE complex double
#  endif
#endif

#define assert_elpa_ok(x) assert(x == ELPA_OK)

#ifdef HAVE_64BIT_INTEGER_SUPPORT
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

#include "test/shared/generated.h"

int main(int argc, char** argv) {
   /* matrix dimensions */
   C_INT_TYPE na, nev, nblk;

   /* mpi */
   C_INT_TYPE myid, nprocs;
   C_INT_TYPE na_cols, na_rows;
   C_INT_TYPE np_cols, np_rows;
   C_INT_TYPE my_prow, my_pcol;
   C_INT_TYPE mpi_comm;

   /* blacs */
   C_INT_TYPE my_blacs_ctxt, sc_desc[9], info;

   /* The Matrix */
   MATRIX_TYPE *a, *as, *z;
   EV_TYPE *ev;

   C_INT_TYPE status;
   int error_elpa; 
   elpa_t handle;

   elpa_autotune_t autotune_handle;
   C_INT_TYPE i, unfinished;

   C_INT_TYPE value;
#ifdef WITH_MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
#else
   nprocs = 1;
   myid = 0;
#endif

   if (argc == 4) {
     na = atoi(argv[1]);
     nev = atoi(argv[2]);
     nblk = atoi(argv[3]);
   } else {
     na = 500;
     nev = 250;
     nblk = 16;
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
   set_up_blacs_descriptor_f(na, nblk, my_prow, my_pcol, np_rows, np_cols, &na_rows, &na_cols, sc_desc, my_blacs_ctxt, &info);

   /* allocate the matrices needed for elpa */
   a  = calloc(na_rows*na_cols, sizeof(MATRIX_TYPE));
   z  = calloc(na_rows*na_cols, sizeof(MATRIX_TYPE));
   as = calloc(na_rows*na_cols, sizeof(MATRIX_TYPE));
   ev = calloc(na, sizeof(EV_TYPE));

#ifdef TEST_REAL
#ifdef TEST_DOUBLE
   prepare_matrix_random_real_double_f(na, myid, na_rows, na_cols, sc_desc, a, z, as);
#else
   prepare_matrix_random_real_single_f(na, myid, na_rows, na_cols, sc_desc, a, z, as);
#endif
#else
#ifdef TEST_DOUBLE
   prepare_matrix_random_complex_double_f(na, myid, na_rows, na_cols, sc_desc, a, z, as);
#else
   prepare_matrix_random_complex_single_f(na, myid, na_rows, na_cols, sc_desc, a, z, as);
#endif
#endif

   if (elpa_init(CURRENT_API_VERSION) != ELPA_OK) {
     fprintf(stderr, "Error: ELPA API version not supported");
     exit(1);
   }

#ifdef OPTIONAL_C_ERROR_ARGUMENT
   handle = elpa_allocate();
#else
   handle = elpa_allocate(&error_elpa);
   assert_elpa_ok(error_elpa);
#endif
   assert_elpa_ok(error_elpa);

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

   /* Setup */
   assert_elpa_ok(elpa_setup(handle));

   elpa_set(handle, "nvidia-gpu", 0, &error_elpa);
   assert_elpa_ok(error_elpa);

   elpa_set(handle, "intel-gpu", 0, &error_elpa);
   assert_elpa_ok(error_elpa);
 
   autotune_handle = elpa_autotune_setup(handle, ELPA_AUTOTUNE_FAST, ELPA_AUTOTUNE_DOMAIN_REAL, &error_elpa);
   assert_elpa_ok(error_elpa);
   /* mimic 20 scf steps */

   for (i=0; i < 20; i++) {

      unfinished = elpa_autotune_step(handle, autotune_handle, &error_elpa);

      if (unfinished == 0) {
        if (myid == 0) {
       	  printf("ELPA autotuning finished in the %d th scf step \n",i);
        }
	break;
      }
      if (myid == 0) {
	printf("The current setting of the ELPA object: \n");
        elpa_print_settings(handle, &error_elpa);

	printf("The state of the autotuning: \n");
        elpa_autotune_print_state(handle, autotune_handle, &error_elpa);
      }


      /* Solve EV problem */
      elpa_eigenvectors(handle, a, ev, z, &error_elpa);
      assert_elpa_ok(error_elpa);

      /* check the results */
#ifdef TEST_REAL
#ifdef TEST_DOUBLE
      status = check_correctness_evp_numeric_residuals_real_double_f(na, nev, na_rows, na_cols, as, z, ev,
                                sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol);
      memcpy(a, as, na_rows*na_cols*sizeof(double));

#else
      status = check_correctness_evp_numeric_residuals_real_single_f(na, nev, na_rows, na_cols, as, z, ev,
                                sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol);
      memcpy(a, as, na_rows*na_cols*sizeof(float));
#endif
#else
#ifdef TEST_DOUBLE
      status = check_correctness_evp_numeric_residuals_complex_double_f(na, nev, na_rows, na_cols, as, z, ev,
                                sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol);
      memcpy(a, as, na_rows*na_cols*sizeof(complex double));
#else
      status = check_correctness_evp_numeric_residuals_complex_single_f(na, nev, na_rows, na_cols, as, z, ev,
                                sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol);
      memcpy(a, as, na_rows*na_cols*sizeof(complex float));
#endif
#endif

      if (status !=0){
        printf("The computed EVs are not correct !\n");
	break;
      }
      printf("hier %d \n",myid);
   }

   if (unfinished == 1) {
     if (myid == 0) {
        printf("ELPA autotuning did not finished during %d scf cycles\n",i);

     }	     

   }
   elpa_autotune_set_best(handle, autotune_handle, &error_elpa);

   if (myid == 0) {
     printf("The best combination found by the autotuning:\n");
     elpa_autotune_print_best(handle, autotune_handle, &error_elpa);
   }

#ifdef OPTIONAL_C_ERROR_ARGUMENT
   elpa_autotune_deallocate(autotune_handle);
   elpa_deallocate(handle);
#else
   elpa_autotune_deallocate(autotune_handle, &error_elpa);
   elpa_deallocate(handle, &error_elpa);
#endif
   elpa_uninit(&error_elpa);

   if (myid == 0) {
     printf("\n");
     printf("2stage ELPA real solver complete\n");
     printf("\n");
   }

   if (status ==0){
     if (myid ==0) {
       printf("All ok!\n");
     }
   }

   free(a);
   free(z);
   free(as);
   free(ev);

#ifdef WITH_MPI
   MPI_Finalize();
#endif

   return !!status;
}
