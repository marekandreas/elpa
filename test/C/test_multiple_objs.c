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

#include "test/shared/generated.h"

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

void set_basic_parameters(elpa_t *handle, int na, int nev, int na_rows, int na_cols, int nblk, int my_prow, int my_pcol){
   int error;
   elpa_set(*handle, "na", na, &error);
   assert_elpa_ok(error);

   elpa_set(*handle, "nev", nev, &error);
   assert_elpa_ok(error);

   elpa_set(*handle, "local_nrows", na_rows, &error);
   assert_elpa_ok(error);

   elpa_set(*handle, "local_ncols", na_cols, &error);
   assert_elpa_ok(error);

   elpa_set(*handle, "nblk", nblk, &error);
   assert_elpa_ok(error);

#ifdef WITH_MPI
   elpa_set(*handle, "mpi_comm_parent", MPI_Comm_c2f(MPI_COMM_WORLD), &error);
   assert_elpa_ok(error);

   elpa_set(*handle, "process_row", my_prow, &error);
   assert_elpa_ok(error);

   elpa_set(*handle, "process_col", my_pcol, &error);
   assert_elpa_ok(error);
#endif
}


int main(int argc, char** argv) {
   /* matrix dimensions */
   int na, nev, nblk;

   /* mpi */
   int myid, nprocs;
   int na_cols, na_rows;
   int np_cols, np_rows;
   int my_prow, my_pcol;
   int mpi_comm;

   /* blacs */
   int my_blacs_ctxt, sc_desc[9], info;

   /* The Matrix */
   MATRIX_TYPE *a, *as, *z;
   EV_TYPE *ev;

   int error, status;
   int gpu, debug, timings;

   char str[400];

   elpa_t elpa_handle_1, elpa_handle_2, *elpa_handle_ptr;

   elpa_autotune_t autotune_handle;
   int i, unfinished;

   int value;
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

   for (np_cols = (int) sqrt((double) nprocs); np_cols > 1; np_cols--) {
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

   elpa_handle_1 = elpa_allocate(&error);
   assert_elpa_ok(error);

   set_basic_parameters(&elpa_handle_1, na, nev, na_rows, na_cols, nblk, my_prow, my_pcol);
   /* Setup */
   assert_elpa_ok(elpa_setup(elpa_handle_1));

   elpa_set(elpa_handle_1, "gpu", 0, &error);
   assert_elpa_ok(error);

   elpa_set(elpa_handle_1, "timings", 1, &error);
   assert_elpa_ok(error);

   elpa_set(elpa_handle_1, "debug", 1, &error);
   assert_elpa_ok(error);

   elpa_store_settings(elpa_handle_1, "initial_parameters.txt", &error);
   assert_elpa_ok(error);

#ifdef WITH_MPI
     // barrier after store settings, file created from one MPI rank only, but loaded everywhere
     MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef OPTIONAL_C_ERROR_ARGUMENT
   elpa_handle_2 = elpa_allocate();
#else
   elpa_handle_2 = elpa_allocate(&error);
#endif
   assert_elpa_ok(error);

   set_basic_parameters(&elpa_handle_2, na, nev, na_rows, na_cols, nblk, my_prow, my_pcol);
   /* Setup */
   assert_elpa_ok(elpa_setup(elpa_handle_2));

   elpa_load_settings(elpa_handle_2, "initial_parameters.txt", &error);

   elpa_get(elpa_handle_2, "gpu", &gpu, &error);
   assert_elpa_ok(error);

   elpa_get(elpa_handle_2, "timings", &timings, &error);
   assert_elpa_ok(error);

   elpa_get(elpa_handle_2, "debug", &debug, &error);
   assert_elpa_ok(error);

   if ((timings != 1) || (debug != 1) || (gpu != 0)){
     printf("Parameters not stored or loaded correctly. Aborting... %d, %d, %d\n", timings, debug, gpu);
     exit(1);
   }

   elpa_handle_ptr = &elpa_handle_2;

   autotune_handle = elpa_autotune_setup(*elpa_handle_ptr, ELPA_AUTOTUNE_FAST, ELPA_AUTOTUNE_DOMAIN_REAL, &error);
   assert_elpa_ok(error);
   /* mimic 20 scf steps */

   for (i=0; i < 20; i++) {

      unfinished = elpa_autotune_step(*elpa_handle_ptr, autotune_handle, &error);

      if (unfinished == 0) {
        if (myid == 0) {
          printf("ELPA autotuning finished in the %d th scf step \n",i);
        }
        break;
      }

      elpa_print_settings(*elpa_handle_ptr, &error);
      elpa_autotune_print_state(*elpa_handle_ptr, autotune_handle, &error);

      sprintf(str, "saved_parameters_%d.txt", i);
      elpa_store_settings(*elpa_handle_ptr, str, &error);
      assert_elpa_ok(error);

      /* Solve EV problem */
      elpa_eigenvectors(*elpa_handle_ptr, a, ev, z, &error);
      assert_elpa_ok(error);

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

     elpa_autotune_print_state(*elpa_handle_ptr, autotune_handle, &error);
     assert_elpa_ok(error);

     sprintf(str, "saved_state_%d.txt", i);
     elpa_autotune_save_state(*elpa_handle_ptr, autotune_handle, str, &error);
     assert_elpa_ok(error);

#ifdef WITH_MPI
     //barrier after save state, file created from one MPI rank only, but loaded everywhere
     MPI_Barrier(MPI_COMM_WORLD);
#endif

     elpa_autotune_load_state(*elpa_handle_ptr, autotune_handle, str, &error);
     assert_elpa_ok(error);

     if (unfinished == 1) {
       if (myid == 0) {
          printf("ELPA autotuning did not finished during %d scf cycles\n",i);
       }
     }

   }
   elpa_autotune_set_best(*elpa_handle_ptr, autotune_handle, &error);

   if (myid == 0) {
     printf("The best combination found by the autotuning:\n");
     elpa_autotune_print_best(*elpa_handle_ptr, autotune_handle, &error);
   }

   elpa_autotune_deallocate(autotune_handle, &error);
   elpa_deallocate(elpa_handle_1, &error);
#ifdef OPTIONAL_C_ERROR_ARGUMENT
   elpa_deallocate(elpa_handle_2);
#else
   elpa_deallocate(elpa_handle_2, &error);
#endif
   elpa_uninit(&error);

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
