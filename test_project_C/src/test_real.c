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
#include <mpi.h>
#include <math.h>
#include <string.h>

#include <elpa/elpa.h>
#include <assert.h>

#define assert_elpa_ok(x) assert(x == ELPA_OK)

int main(int argc, char** argv) {
   /* matrix dimensions */
   const int na = 1000;
   const int nev = 500;
   const int nblk = 16;

   /* mpi */
   int myid, nprocs;
   int na_cols, na_rows;
   int np_cols, np_rows;
   int my_prow, my_pcol;
   int mpi_comm;

   /* blacs */
   int my_blacs_ctxt, sc_desc[9], info;

   /* The Matrix */
   double *a, *as, *z;
   double *ev;

   int error, status;

   elpa_t handle;

   int value;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);


   for (np_cols = (int) sqrt((double) nprocs); np_cols > 1; np_cols--) {
     if (nprocs % np_cols == 0) {
       break;
     }
   }

   np_rows = nprocs/np_cols;

   /* set up blacs */
   /* convert communicators before */
   mpi_comm = MPI_Comm_c2f(MPI_COMM_WORLD);
   set_up_blacsgrid_f(mpi_comm, np_rows, np_cols, 'C', &my_blacs_ctxt, &my_prow, &my_pcol);
   set_up_blacs_descriptor_f(na, nblk, my_prow, my_pcol, np_rows, np_cols, &na_rows, &na_cols, sc_desc, my_blacs_ctxt, &info);

   /* allocate the matrices needed for elpa */
   a  = calloc(na_rows*na_cols, sizeof(double));
   z  = calloc(na_rows*na_cols, sizeof(double));
   as = calloc(na_rows*na_cols, sizeof(double));
   ev = calloc(na, sizeof(double));

   // TODO: prepare properly
   memset(a, 0, na_rows * na_cols * sizeof(double));
   //prepare_matrix_real_double_f(na, myid, na_rows, na_cols, sc_desc, a, z, as);

   if (elpa_init(20170403) != ELPA_OK) {
     fprintf(stderr, "Error: ELPA API version not supported");
     exit(1);
   }

   if(myid == 0) printf("init done\n");

   handle = elpa_allocate(&error);
   assert_elpa_ok(error);

   /* Set parameters */
   elpa_set(handle, "na", na, &error);
   assert_elpa_ok(error);

   elpa_set(handle, "nev", nev, &error);
   assert_elpa_ok(error);

   elpa_set(handle, "local_nrows", na_rows, &error);
   assert_elpa_ok(error);

   elpa_set(handle, "local_ncols", na_cols, &error);
   assert_elpa_ok(error);

   elpa_set(handle, "nblk", nblk, &error);
   assert_elpa_ok(error);

   elpa_set(handle, "mpi_comm_parent", MPI_Comm_c2f(MPI_COMM_WORLD), &error);
   assert_elpa_ok(error);

   elpa_set(handle, "process_row", my_prow, &error);
   assert_elpa_ok(error);

   elpa_set(handle, "process_col", my_pcol, &error);
   assert_elpa_ok(error);
//
   /* Setup */
   assert_elpa_ok(elpa_setup(handle));

   if(myid == 0) printf("setup done\n");
   /* Set tunables */
   elpa_set(handle, "solver", ELPA_SOLVER_1STAGE, &error);
  // elpa_set(handle, "solver", ELPA_SOLVER_2STAGE, &error);
   assert_elpa_ok(error);

//   elpa_set(handle, "real_kernel", TEST_KERNEL, &error);
//   assert_elpa_ok(error);

   if(myid == 0) printf("solve..\n");

   /* Solve EV problem */
   elpa_eigenvectors(handle, a, ev, z, &error);
   assert_elpa_ok(error);
   if(myid == 0) printf("solve done \n");

//   for(int i = 0; i < na; i++)
//       printf("%lf, ", ev[i]);
//   printf("\n");

   elpa_deallocate(handle);
   elpa_uninit();


   /* check the results */
//   status = check_correctness_real_double_f(na, nev, na_rows, na_cols, as, z, ev, sc_desc, myid);

//   if (status !=0){
//     printf("The computed EVs are not correct !\n");
//   }
//   if (status ==0){
//     printf("All ok!\n");
//   }

   free(a);
   free(z);
   free(as);
   free(ev);

   MPI_Finalize();

   return !!status;
}
