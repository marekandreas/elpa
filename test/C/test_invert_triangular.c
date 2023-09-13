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
#    define PRINT_MATRIX print_matrix_real_single_f
#    define PREPARE_MATRIX_RANDOM_TRIANGULAR prepare_matrix_random_triangular_real_single_f
#    define PREPARE_MATRIX_UNIT prepare_matrix_unit_real_single_f
#    define CHECK_CORRECTNESS_HERMITIAN_MULTIPLY check_correctness_hermitian_multiply_real_single_f
#  else
#    define MATRIX_TYPE float_complex
#    define PRINT_MATRIX print_matrix_complex_single_f
#    define PREPARE_MATRIX_RANDOM_TRIANGULAR prepare_matrix_random_triangular_complex_single_f
#    define PREPARE_MATRIX_UNIT prepare_matrix_unit_complex_single_f
#    define CHECK_CORRECTNESS_HERMITIAN_MULTIPLY check_correctness_hermitian_multiply_complex_single_f
#  endif
#else
#  define EV_TYPE double
#  ifdef TEST_REAL
#    define MATRIX_TYPE double
#    define PRINT_MATRIX print_matrix_real_double_f
#    define PREPARE_MATRIX_RANDOM_TRIANGULAR prepare_matrix_random_triangular_real_double_f
#    define PREPARE_MATRIX_UNIT prepare_matrix_unit_real_double_f
#    define CHECK_CORRECTNESS_HERMITIAN_MULTIPLY check_correctness_hermitian_multiply_real_double_f
#  else
//#    define MATRIX_TYPE std::complex<double>
#    define MATRIX_TYPE double_complex
#    define PRINT_MATRIX print_matrix_complex_double_f
#    define PREPARE_MATRIX_RANDOM_TRIANGULAR prepare_matrix_random_triangular_complex_double_f
#    define PREPARE_MATRIX_UNIT prepare_matrix_unit_complex_double_f
#    define CHECK_CORRECTNESS_HERMITIAN_MULTIPLY check_correctness_hermitian_multiply_complex_double_f
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
   C_INT_TYPE na_cols, na_rows;
   C_INT_TYPE np_cols, np_rows;
   C_INT_TYPE my_prow, my_pcol;
   C_INT_TYPE mpi_comm;

   /* blacs */
   C_INT_TYPE my_blacs_ctxt, sc_desc[9], info, blacs_ok;

   /* The Matrix */
   MATRIX_TYPE *a, *as, *c;

#if TEST_GPU == 1
   MATRIX_TYPE *a_dev;
   C_INT_TYPE gpuID = 0;
   C_INT_TYPE numberOfDevices;
   C_INT_TYPE successGPU;
#endif
   
   C_INT_TYPE status;
   int error_elpa;
  
   elpa_t handle;

#ifdef WITH_MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
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

// pointer API is tested for NVIDIA, AMD, and INTEL_SYCL
#if TEST_GPU_DEVICE_POINTER_API == 1 && TEST_NVIDIA_GPU == 0 && TEST_AMD_GPU == 0 && TEST_INTEL_GPU_SYCL == 0
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
   a  = (MATRIX_TYPE *) calloc(na_rows*na_cols, sizeof(MATRIX_TYPE));
   as = (MATRIX_TYPE *) calloc(na_rows*na_cols, sizeof(MATRIX_TYPE));
   c  = (MATRIX_TYPE *) calloc(na_rows*na_cols, sizeof(MATRIX_TYPE));
   
   PREPARE_MATRIX_RANDOM_TRIANGULAR (na, a, nblk, myid, na_rows, na_cols, np_rows, np_cols, my_prow, my_pcol);
   memcpy(as, a, na_rows*na_cols*sizeof(MATRIX_TYPE));
   //PRINT_MATRIX(myid, na_rows, a, "a");
   
   PREPARE_MATRIX_UNIT (na, c, nblk, myid, na_rows, na_cols, np_rows, np_cols, my_prow, my_pcol);
   //PRINT_MATRIX(myid, na_rows, c, "c");
   
   if (elpa_init(CURRENT_API_VERSION) != ELPA_OK) {
     fprintf(stderr, "Error: ELPA API version not supported");
     exit(1);
   }
   
   handle = elpa_allocate(&error_elpa);
   assert_elpa_ok(error_elpa);

   /* Set parameters */
   
   elpa_set(handle, "na", (int) na, &error_elpa);
   assert_elpa_ok(error_elpa);

   elpa_set(handle, "nev", (int) nev, &error_elpa);
   assert_elpa_ok(error_elpa);

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
   
   elpa_set(handle, "debug", 1, &error_elpa);
   assert_elpa_ok(error_elpa);

   /* Setup */
   assert_elpa_ok(elpa_setup(handle));

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


#if (TEST_GPU == 1)

#if (TEST_INTEL_GPU == 0) && (TEST_INTEL_GPU_OPENMP == 0) && (TEST_INTEL_GPU_SYCL == 0)
   gpuGetDeviceCount(&numberOfDevices);
   printf("Number of Devices found: %d\n\n", numberOfDevices);
   gpuID = myid%numberOfDevices; 
   printf("gpuID: %i\n", gpuID);
   elpa_set(handle, "use_gpu_id", gpuID, &error_elpa);
   assert_elpa_ok(error_elpa);
#endif

   set_gpu_parameters();

#if TEST_INTEL_GPU_SYCL == 1 /* temporary fix for SYCL on CPU */
   successGPU = syclGetCpuCount(numberOfDevices); 
   if (!successGPU){    
      printf("Error in syclGetCpuCount\n");
      exit(1);
      }
#endif

   successGPU = gpuSetDevice(gpuID);
   if (!successGPU){    
      printf("Error in gpuSetDevice\n");
      exit(1);
      }

   elpa_set(handle, "gpu_invert_trm", 1, &error_elpa);
   assert_elpa_ok(error_elpa); 
#endif /* TEST_GPU */


   //-----------------------------------------------------------------------------------------------------------------------------
   // TEST_GPU == 1: create device pointer for a_dev; copy a -> a_dev
#if TEST_GPU == 1

   // malloc
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
#endif /* TEST_GPU */   
   
   //-----------------------------------------------------------------------------------------------------------------------------
   // The actual solve step

#if defined(TEST_EXPLICIT_NAME)
     printf("Inverting with TEST_EXPLICIT_NAME\n");
   
#if defined(TEST_REAL)
#if defined(TEST_DOUBLE)
#if TEST_GPU == 1
     printf("Inverting with device API\n");
     elpa_invert_triangular_double(handle, a_dev, &error_elpa);
     assert_elpa_ok(error_elpa);
#else
     printf("Inverting without device API\n");
     elpa_invert_triangular_double(handle, a, &error_elpa);
     assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_DOUBLE */
#if defined(TEST_SINGLE)
#if TEST_GPU == 1
     printf("Inverting with device API\n");
     elpa_invert_triangular_float(handle, a_dev, &error_elpa);
     assert_elpa_ok(error_elpa);
#else
     printf("Inverting without device API\n");
     elpa_invert_triangular_float(handle, a, &error_elpa);
     assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_SINGLE */
#endif /* TEST_REAL */

#if defined(TEST_COMPLEX)
#if defined(TEST_DOUBLE)
#if TEST_GPU == 1
     printf("Inverting with device API\n");   
     elpa_invert_triangular_double_complex(handle, a_dev, &error_elpa);
     assert_elpa_ok(error_elpa);
#else
     printf("Inverting without device API\n");
     elpa_invert_triangular_double_complex(handle, a, &error_elpa);
     assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_DOUBLE */
#if defined(TEST_SINGLE)
#if TEST_GPU == 1
     printf("Inverting with device API\n");
     elpa_invert_triangular_float_complex(handle, a_dev, &error_elpa);
     assert_elpa_ok(error_elpa);
#else
     printf("Inverting without device API\n");
     elpa_invert_triangular_float_complex(handle, a, &error_elpa);
     assert_elpa_ok(error_elpa);
#endif
#endif /* TEST_SINGLE */
#endif /* TEST_COMPLEX */

#else /* TEST_EXPLICIT_NAME */
   printf("Inverting without TEST_EXPLICIT_NAME\n");
   elpa_invert_triangular (handle, a, &error_elpa);
   assert_elpa_ok(error_elpa);
#endif /* TEST_EXPLICIT_NAME */
   
   //PRINT_MATRIX(myid, na_rows, a, "a_inverted");

   //-----------------------------------------------------------------------------------------------------------------------------     
   // TEST_GPU == 1: copy for testing from device to host, deallocate device pointers
#if TEST_GPU == 1
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
#endif /* TEST_GPU */
   
   //-----------------------------------------------------------------------------------------------------------------------------
   // Check the results
   
   status = CHECK_CORRECTNESS_HERMITIAN_MULTIPLY('N', na, a, as, c, na_rows, na_cols, sc_desc, myid);
   
   if (myid==0) {
      if (status !=0) {
         printf("Test produced an error!\n");
         printf("Check whether the matrix is well-conditioned\n");
      }
      if (status ==0) printf("All ok!\n");
   }
   

   //-----------------------------------------------------------------------------------------------------------------------------
   // Deallocate

   elpa_deallocate(handle, &error_elpa);

   elpa_uninit(&error_elpa);

   free(a);
   free(as);
   free(c);
	
	
#ifdef WITH_MPI
   MPI_Finalize();
#endif
	
	return status;
}
