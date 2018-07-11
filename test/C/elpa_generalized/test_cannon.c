#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#ifdef WITH_MPI
#include <mpi.h>
#endif
#include <math.h>

#include <elpa/elpa.h>
#include <elpa/elpa_generated.h>
#include <elpa/elpa_constants.h>
#include <elpa/elpa_generated_legacy.h>
#include <elpa/elpa_generic.h>
#include <elpa/elpa_legacy.h>


const double TOL = 1e-10;

int main(int argc, char** argv) {
   int myid;
   int nprocs;
   int status;
#ifndef WITH_MPI
   int MPI_COMM_WORLD;
#endif
   int my_mpi_comm_world, mpi_comm_rows, mpi_comm_cols;
   int na, nev, nblk, np_cols, np_rows, np_colsStart, my_blacs_ctxt, nprow, npcol, my_prow, my_pcol;

   int mpierr;

   int info, i, j, na_rows, na_cols, rep, new_f, nev_f, Liwork, Lwork_find, LocC; 
   
   double startVal;

   double *a, *b, *EigenVectors, *EigValues_elpa, *a_copy, *b_copy, *c, *AUinv, *EigVectors_gen, *work_find;
   int *a_desc, *b_desc, *AUinv_desc, *c_desc, *EigenVectors_desc; 
      
   double startTime, endTime, localTime, AverageTime, MaxTime, diff, diff_max, start_in, end_in, time_invert, time_mult_from_left, time_mult_from_left2, time_mult_1, time_mult_2;
   double time_transpose, back_transform_time, back_average, back_max, overall_reduce_time, overall_reduce_av, overall_reduce_max;
   double reduce_time, reduce_av, reduce_max, time_invert_av, time_invert_max;
   double time_mult_1_av, time_mult_2_av, time_mult_1_max, time_mult_2_max; 
      
   int useQr, THIS_REAL_ELPA_KERNEL_API, success; 
   double value; 
   
   double done = 1.0; 
   int one = 1; 
   double dzero = 0.0; 
   int zero = 0;
   
   double *Ax, *Bx, *lambdaBx, *lambda_Matr;
   int *Ax_desc, *Bx_desc, *lambdaBx_desc, *a_copy_desc, *b_copy_desc, *lambda_Matr_desc;
   int reallevel;
   double BuffLevel;
   
   status = 0;
#ifdef WITH_MPI
   MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &reallevel);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   if(myid == 0)
     printf("Threading level = %d \n", reallevel);
#else
   nprocs = 1;
   myid=0;
   MPI_COMM_WORLD=1;
#endif
   
   if(argc < 4 || argc > 5)
     return 1;

   na = atoi(argv[1]);
   nev = atoi(argv[2]);
   //nev = (int)na*0.33;
   nblk = atoi(argv[3]);
   if(argc == 5)
     BuffLevel = atof(argv[4]);
   else
     BuffLevel = 1;

   if (myid == 0)
      printf("Number of eigenvalues: %d\nBuffLevel: %lf\n", nev, BuffLevel);
   Liwork = 20*na;
    
   ///////////// procs grids and communicators ///////////////////////////////////////////////
   if (myid == 0)
     printf("Matrix size: %d, blocksize: %d\n\n", na, nblk);

   startVal = sqrt((double) nprocs);
   np_colsStart = (int) round(startVal);
   for (np_rows=np_colsStart;np_rows>1;np_rows--){
     if (nprocs %np_rows ==0)
     break;
     }
   np_cols = nprocs/np_rows;
   if (myid == 0)
     printf("Number of processor rows %d, cols %d, total %d \n\n",np_rows,np_cols,nprocs);
   
   /* set up blacs */
   /* convert communicators before */
#ifdef WITH_MPI
   my_mpi_comm_world = MPI_Comm_c2f(MPI_COMM_WORLD);
#else
  my_mpi_comm_world = 1;
#endif
//set_up_blacsgrid_f1(my_mpi_comm_world, &my_blacs_ctxt, &np_rows, &np_cols, &nprow, &npcol, &my_prow, &my_pcol);
set_up_blacsgrid_f(my_mpi_comm_world, np_rows, np_cols, 'R', &my_blacs_ctxt, &my_prow, &my_pcol);

   /* get the ELPA row and col communicators. */
   /* These are NOT usable in C without calling the MPI_Comm_f2c function on them !! */
#ifdef WITH_MPI
   my_mpi_comm_world = MPI_Comm_c2f(MPI_COMM_WORLD);
#endif
   mpierr = elpa_get_communicators(my_mpi_comm_world, my_prow, my_pcol, &mpi_comm_rows, &mpi_comm_cols);

   ////////////////////// descriptors area ///////////////////////////////////////////////
   a_desc = malloc(9*sizeof(int));
   b_desc = malloc(9*sizeof(int));
   AUinv_desc = malloc(9*sizeof(int));
   c_desc = malloc(9*sizeof(int));
   EigenVectors_desc = malloc(9*sizeof(int));
   int *EigenVectors_desc1 = malloc(9*sizeof(int));
   Ax_desc = malloc(9*sizeof(int));
   Bx_desc = malloc(9*sizeof(int));
   lambdaBx_desc = malloc(9*sizeof(int));
   a_copy_desc = malloc(9*sizeof(int));
   b_copy_desc = malloc(9*sizeof(int));
   lambda_Matr_desc = malloc(9*sizeof(int));
   
   na_rows = numroc_(&na, &nblk, &my_prow, &zero, &np_rows);
   na_cols = numroc_(&na, &nblk, &my_pcol, &zero, &np_cols);
   
   descinit_(a_desc, &na, &na, &nblk, &nblk, &zero, &zero, &my_blacs_ctxt, &na_rows, &info);
   descinit_(b_desc, &na, &na, &nblk, &nblk, &zero, &zero, &my_blacs_ctxt, &na_rows, &info);
   descinit_(a_copy_desc, &na, &na, &nblk, &nblk, &zero, &zero, &my_blacs_ctxt, &na_rows, &info);
   descinit_(b_copy_desc, &na, &na, &nblk, &nblk, &zero, &zero, &my_blacs_ctxt, &na_rows, &info);
   descinit_(AUinv_desc, &na, &na, &nblk, &nblk, &zero, &zero, &my_blacs_ctxt, &na_rows, &info);
   descinit_(c_desc, &na, &na, &nblk, &nblk, &zero, &zero, &my_blacs_ctxt, &na_rows, &info);

   LocC = numroc_(&nev, &nblk, &my_pcol, &zero, &np_cols);
   int LocR_1 = numroc_(&nev, &nblk, &my_prow, &zero, &np_rows);
   descinit_(EigenVectors_desc, &na, &na, &nblk, &nblk, &zero, &zero, &my_blacs_ctxt, &na_rows, &info);
   descinit_(EigenVectors_desc1, &na, &nev, &nblk, &nblk, &zero, &zero, &my_blacs_ctxt, &na_rows, &info);
   descinit_(Ax_desc, &na, &nev, &nblk, &nblk, &zero, &zero, &my_blacs_ctxt, &na_rows, &info);
   descinit_(Bx_desc, &na, &nev, &nblk, &nblk, &zero, &zero, &my_blacs_ctxt, &na_rows, &info);
   descinit_(lambdaBx_desc, &na, &nev, &nblk, &nblk, &zero, &zero, &my_blacs_ctxt, &na_rows, &info);
   descinit_(lambda_Matr_desc, &nev, &nev, &nblk, &nblk, &zero, &zero, &my_blacs_ctxt, &LocR_1, &info);

   if ((na_rows*na_cols + 2*nblk*nblk) > 18*na)
      Lwork_find = (na_rows*na_cols + 2*nblk*nblk + 5*na + (nev/(nprocs) + 4)*na)*10;
   else
      Lwork_find = (25*na + (nev/(nprocs) + 4)*na)*10;
   
   /////////////////////////memory allocation area//////////////////////////////////////////////////////////////
   a  = malloc(na_rows*na_cols*sizeof(double));
   b  = malloc(na_rows*na_cols*sizeof(double));
   EigValues_elpa = malloc(na*sizeof(double)); 
   EigenVectors = malloc(na_rows*na_cols*sizeof(double));
   a_copy = malloc(na_rows*na_cols*sizeof(double));
   b_copy = malloc(na_rows*na_cols*sizeof(double));
   c = malloc(na_rows*na_cols*sizeof(double));
   work_find = malloc(Lwork_find*sizeof(double));
   AUinv = malloc(na_rows*na_cols*sizeof(double));
   int* Iwork = malloc(Liwork*sizeof(int));
   Ax = malloc(na_rows*LocC*sizeof(double));
   Bx = malloc(na_rows*LocC*sizeof(double));
   lambdaBx = malloc(na_rows*LocC*sizeof(double));
   lambda_Matr = malloc(LocR_1*LocC*sizeof(double));
   EigVectors_gen = malloc(na_rows*na_cols*sizeof(double));

   //////////////////////////generate matrices//////////////////////////////////////////////////////////////////////////////
   int i_global, j_global;
   for(i = 0; i < na_rows; i++)
      for(j = 0; j < na_cols; j++)
      {
         i_global = np_rows*nblk*(i/nblk) + i%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1; 
         j_global = np_cols*nblk*(j/nblk) + j%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
         a[i + j*na_rows] = (double)cos(i_global)*cos(j_global) + (double)sin(i_global)*sin(j_global);
         if(i_global == j_global)
            a[i + j*na_rows] = a[i + j*na_rows] + (double)(i_global + j_global)/na;
         b[i + j*na_rows] = (double)sin(i_global)*(double)sin(j_global);   
         if(i_global == j_global)
            b[i + j*na_rows] = b[i + j*na_rows] + 1;  
      }
   
   //make copies of a and b
   for(i = 0; i < na_rows*na_cols; i++)
   {
      a_copy[i] = a[i];
      b_copy[i] = b[i];
   }
  
   new_f = 0; 
   nev_f = 0; 
  
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   
   //////////////////////////////////////////////////////////////////////////// Test of our algorithm ////////////////////////////////////////////////////////////////////////////////////////////////////
   
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   for(rep = 0; rep < 2; rep++)
   {
      //restore a and b
      for(i = 0; i < na_rows*na_cols; i++)
      {
         a[i] = a_copy[i];
         b[i] = b_copy[i];
         AUinv[i] = 0;
      }  
      if(myid == 0)
         printf("My algorithm \n\n ");
   
      ///////////////////////////////////////////////////////// Cholesky for B /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      elpa_cholesky_real_double(na, b, na_rows, nblk, na_cols, mpi_comm_rows, mpi_comm_cols, 1);   // now b = U
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();
      localTime = endTime - startTime; 
      MPI_Reduce(&localTime, &AverageTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localTime, &MaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
 
      if (myid == 0)
         printf("Cholesky is done, 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, AverageTime/nprocs, MaxTime);

      int BuffLevelInt = BuffLevel*(np_rows-1);
   
      ///////////////////////////////////////////////////////////////////////////////////// Reduction ///////////////////////////////////////////////////////////////////////////////////////////////////////////
      MPI_Barrier(MPI_COMM_WORLD);
      start_in = MPI_Wtime();
      elpa_invert_trm_real_double(na, b, na_rows, nblk, na_cols, mpi_comm_rows, mpi_comm_cols, 1);    // now b = U(-1)  
      MPI_Barrier(MPI_COMM_WORLD);
      end_in = MPI_Wtime();
      time_invert = end_in - start_in;
      
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      d_cannons_reduction(a, b, np_rows, np_cols, my_prow, my_pcol, a_desc, AUinv, BuffLevelInt, 
                          MPI_Comm_f2c(mpi_comm_cols), MPI_Comm_f2c(mpi_comm_rows));
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();
      reduce_time = endTime - startTime; 
      
      MPI_Reduce(&reduce_time, &reduce_av, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&reduce_time, &reduce_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      reduce_av = reduce_av/nprocs; 
      
      MPI_Reduce(&time_invert, &time_invert_av, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&time_invert, &time_invert_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      time_invert_av = time_invert_av/nprocs; 
   
      if(myid == 0)
      {
         printf("Time for reduction: on 0 proc %lf, average over procs = %lf, max = %lf\n\n", reduce_time, reduce_av, reduce_max);
         printf("Time for invertion: on 0 proc %lf, average over procs = %lf, max = %lf\n\n", time_invert, time_invert_av, time_invert_max);
      }
  
////////////////////////////////////////////////////////////////////// Solution area ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
      useQr = 0;
      THIS_REAL_ELPA_KERNEL_API = ELPA_2STAGE_REAL_GENERIC;
      for(i = 0; i < na; i++)
         EigValues_elpa[i] = 0;
      AverageTime = 0;
      MaxTime = 0;
            
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      int useGPU = 0;
      success = elpa_solve_evp_real_2stage_double_precision(na, na, AUinv, na_rows, EigValues_elpa, EigenVectors, na_rows, nblk, na_cols, mpi_comm_rows, mpi_comm_cols, my_mpi_comm_world, THIS_REAL_ELPA_KERNEL_API, useQr, useGPU);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();
      localTime = endTime - startTime; 
      MPI_Reduce(&localTime, &AverageTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localTime, &MaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   
      if(myid == 0)
         printf("\n ELPA Solution is done, 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, AverageTime/nprocs, MaxTime);
      
      if (success != 1) {
         printf("error in ELPA solve \n");
#ifdef WITH_MPI
        mpierr = MPI_Abort(MPI_COMM_WORLD, 99);
#endif
      }
      
      // create matrix of the EigenValues for the later check 
      for (i = 0; i < LocR_1*LocC; i++)
         lambda_Matr[i] = 0;
      for (i = 1; i <= nev; i++)
      {
         value = EigValues_elpa[i-1];
         pdelset_(lambda_Matr, &i, &i, lambda_Matr_desc, &value);
      }
            
//////////////////////////////////////////////////////////////////////////////////////////////// back transform /////////////////////////////////////////////////////////////////////// 
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      d_cannons_triang_rectangular(b, EigenVectors, np_rows, np_cols, my_prow, my_pcol, b_desc, EigenVectors_desc1, EigVectors_gen, 
                                   MPI_Comm_f2c(mpi_comm_cols), MPI_Comm_f2c(mpi_comm_rows));
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();
      back_transform_time = endTime - startTime; 
      MPI_Reduce(&back_transform_time, &back_average, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&back_transform_time, &back_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      
      if(myid == 0)
         printf("\n Cannons back transform is done, 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", back_transform_time, back_average/nprocs, back_max);
   
      //////////////////////////////////////// Check the results //////////////////////////////////////////////////
      
      pdsymm_("L", "L", &na, &nev, &done, a_copy, &one, &one, a_copy_desc, EigVectors_gen, &one, &one, EigenVectors_desc1, &dzero, Ax, &one, &one, Ax_desc);
      pdsymm_("L", "L", &na, &nev, &done, b_copy, &one, &one, b_copy_desc, EigVectors_gen, &one, &one, EigenVectors_desc1, &dzero, Bx, &one, &one, Bx_desc);
      pdsymm_("R", "L", &na, &nev, &done, lambda_Matr, &one, &one, lambda_Matr_desc, Bx, &one, &one, Bx_desc, &dzero, lambdaBx, &one, &one, lambdaBx_desc);
      
      diff = 0;
      for (i = 0; i < na_rows; i++)
         for (j = 0; j < LocC; j++)
            diff = diff + fabs(Ax[i + j*na_rows] - lambdaBx[i + j*na_rows]);  

      MPI_Reduce(&diff, &diff_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      if(myid == 0){
         printf("max accumulated diff of the Ax-lamBx = %.15e \n", diff_max);
         if(diff_max > TOL){
           printf("Results incorrect, difference exceeds tolerance\n");
           status = 1;
         }
         printf("_______________________________________________________________________________________________________\n");
      }
   }
   
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////   

//////////////////////////////////////////////// Test of ELPA //////////////////////////////////////////////////////////////////////////////////////////////// 

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   for(rep = 0; rep < 2; rep++)
   {
      if (myid == 0)
         printf("\n ELPA\n\n");
   
      //restore a and b from copies
      for(i = 0; i < na_rows*na_cols; i++)
      {
         a[i] = a_copy[i];
         b[i] = b_copy[i]; 
      }
  
      /////////////////////////////////////////////// Cholesky for B /////////////////////////////////////////////////////////
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      elpa_cholesky_real_double(na, b, na_rows, nblk, na_cols, mpi_comm_rows, mpi_comm_cols, 1);         // Upper part; Lower is 0
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();
      localTime = endTime - startTime; 
      MPI_Reduce(&localTime, &AverageTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localTime, &MaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
 
      if (myid == 0)
         printf("\n Cholesky is done, 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, AverageTime/nprocs, MaxTime);
      
      ////////////////////////////////////////////////////////////// Reduction ///////////////////////////////////////////////////////////////////////
      MPI_Barrier(MPI_COMM_WORLD);
      start_in = MPI_Wtime();
      elpa_invert_trm_real_double(na, b, na_rows, nblk, na_cols, mpi_comm_rows, mpi_comm_cols, 1);    // now b = U(-1)  
      MPI_Barrier(MPI_COMM_WORLD);
      end_in = MPI_Wtime();
      time_invert = end_in - start_in;
      
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      elpa_mult_at_b_real_double('U', 'L', na, na, b, na_rows, na_cols, a, na_rows, na_cols, nblk, mpi_comm_rows, mpi_comm_cols, c, na_rows, na_cols);  // now c = U(-H)*A
      pdtran_(&na, &na, &done, c, &one, &one, c_desc, &dzero, AUinv, &one, &one, AUinv_desc);  //AUinv = A*U(-1)
      elpa_mult_at_b_real_double('U', 'U', na, na, b, na_rows, na_cols, AUinv, na_rows, na_cols, nblk, mpi_comm_rows, mpi_comm_cols, c, na_rows, na_cols);  // now c = U(-H)*A*U(-1)
      pdtran_(&na, &na, &done, c, &one, &one, a_desc, &dzero, AUinv, &one, &one, AUinv_desc);
      pdlacpy_("L", &na, &na, AUinv, &one, &one, AUinv_desc, c, &one, &one, a_desc);
      MPI_Barrier(MPI_COMM_WORLD);   
      endTime = MPI_Wtime();
      reduce_time = endTime-startTime; 
      
      MPI_Reduce(&reduce_time, &reduce_av, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&reduce_time, &reduce_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      reduce_av = reduce_av/nprocs;
      
      MPI_Reduce(&time_invert, &time_invert_av, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&time_invert, &time_invert_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      time_invert_av = time_invert_av/nprocs;
   
      if (myid == 0)
      {
         printf("Reduce from ELPA My is done, 0 proc time is %lf, average = %lf, max = %lf \n", reduce_time, reduce_av, reduce_max);
         printf("Time for triangular invert of U (ELPA function): %lf, average = %lf, max = %lf \n\n", time_invert, time_invert_av, time_invert_max);
      }
   
      ///////////// Solution area ////////////////////////////////////////////////////////////// 
      useQr = 0;
  
      for(i = 0; i < na; i++)
         EigValues_elpa[i] = 0;
   
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      int useGPU = 0;
      success = elpa_solve_evp_real_2stage_double_precision(na, nev, c, na_rows, EigValues_elpa, EigenVectors, na_rows, nblk, na_cols, mpi_comm_rows, mpi_comm_cols, my_mpi_comm_world, THIS_REAL_ELPA_KERNEL_API, useQr, useGPU);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();
      localTime = endTime - startTime; 

      if (success != 1) {
         printf("error in ELPA solve \n");
#ifdef WITH_MPI
        mpierr = MPI_Abort(MPI_COMM_WORLD, 99);
#endif
      }

      MPI_Reduce(&localTime, &AverageTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localTime, &MaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
 
      if (myid == 0)
         printf("Solution ELPA is done, 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, AverageTime/nprocs, MaxTime);
      
      // create matrix of the EigenValues for the later check 
      for (i = 0; i < LocR_1*LocC; i++)
         lambda_Matr[i] = 0;
      for (i = 1; i <= nev; i++)
      {
         value = EigValues_elpa[i-1];
         pdelset_(lambda_Matr, &i, &i, lambda_Matr_desc, &value);
      }
////////////////////////////////////////////////////////////////// back transform //////////////////////////////////////////////////////////////////////////
      MPI_Barrier(MPI_COMM_WORLD);
      start_in = MPI_Wtime();
      pdtran_(&na, &na, &done, b, &one, &one, b_desc, &dzero, AUinv, &one, &one, AUinv_desc);
      elpa_mult_at_b_real_double('L', 'F', na, nev, AUinv, na_rows, na_cols, EigenVectors, na_rows, na_cols, nblk, mpi_comm_rows, mpi_comm_cols, EigVectors_gen, na_rows, na_cols);
      MPI_Barrier(MPI_COMM_WORLD);
      end_in = MPI_Wtime();
      back_transform_time = end_in - start_in; 
      MPI_Reduce(&back_transform_time, &back_average, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&back_transform_time, &back_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      
      if(myid == 0)
         printf("\n Transpose + ELPA A_TB back transform is done, 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", back_transform_time, back_average/nprocs, back_max);
   
//////////////////////////////////////// Check the results //////////////////////////////////////////////////

      pdsymm_("L", "L", &na, &nev, &done, a_copy, &one, &one, a_copy_desc, EigVectors_gen, &one, &one, EigenVectors_desc, &dzero, Ax, &one, &one, Ax_desc);
      pdsymm_("L", "L", &na, &nev, &done, b_copy, &one, &one, b_copy_desc, EigVectors_gen, &one, &one, EigenVectors_desc, &dzero, Bx, &one, &one, Bx_desc);
      pdsymm_("R", "L", &na, &nev, &done, lambda_Matr, &one, &one, lambda_Matr_desc, Bx, &one, &one, Bx_desc, &dzero, lambdaBx, &one, &one, lambdaBx_desc);
      
      diff = 0;
      for (i = 0; i < na_rows; i++)
         for (j = 0; j < LocC; j++)
            diff = diff + fabs(Ax[i + j*na_rows] - lambdaBx[i + j*na_rows]);   

      MPI_Reduce(&diff, &diff_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      if(myid == 0) {
         printf("max accumulated diff of the Ax-lamBx = %.15e \n", diff_max);
         if(diff_max > TOL){
           printf("Results incorrect, difference exceeds tolerance\n");
           status = 1;
         }
         printf("_______________________________________________________________________________________________________\n");
      }
   }
   
   free(c); 
   free(c_desc);
   free(AUinv);
   free(AUinv_desc);
   free(EigVectors_gen);
   MPI_Barrier(MPI_COMM_WORLD);
   
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////   

//////////////////////////////////////////////// Test of  ScaLAPACK /////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   
   int IBYTYPE = 1;
   double Scale; 
   int NP0, NQ0, Lwork; 
   NP0 = numroc_(&na, &nblk, &zero, &zero, &np_rows);
   NQ0 = numroc_(&na, &nblk, &zero, &zero, &np_cols);
   Lwork = 2*NP0*nblk + NQ0*nblk + nblk*nblk + 1000; 
   double* work1 = malloc(Lwork*sizeof(double));
   for(rep = 0; rep < 2; rep++)
   {
      if (myid == 0)
         printf("\n ScaLAPACK\n\n");
   
      //restore a and b from copies
      for(i = 0; i < na_rows*na_cols; i++)
      {
         a[i] = a_copy[i];
         b[i] = b_copy[i]; 
      }
   
      /////////////////////////////////////////////////////////// Cholesky for B /////////////////////////////////////////////////////////////////////////////////////////
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      pdpotrf_("L", &na, b, &one, &one, b_desc, &info);           // rewrites only lower triang part of b
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();
      localTime = endTime - startTime; 
      MPI_Reduce(&localTime, &AverageTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localTime, &MaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
 
      if (myid == 0)
         printf("\n Cholesky is done, 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, AverageTime/nprocs, MaxTime);
   
      /////////////////////////////////////////////////////// Reduction //////////////////////////////////////////////////////////////////////
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      pdsyngst_(&IBYTYPE, "L", &na, a, &one, &one, a_desc, b, &one, &one, b_desc, &Scale, work1, &Lwork, &info);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();
      reduce_time = endTime-startTime; 
      MPI_Reduce(&reduce_time, &reduce_av, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&reduce_time, &reduce_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      if (myid == 0)
         printf("Reduce from ScaLAPACK is done, 0 proc time is %lf, average = %lf, max = %lf \n\n", reduce_time, reduce_av/nprocs, reduce_max);

      //////////////////////////////////////////////////////////////////////////////// Solution area ////////////////////////////////////////////////////////////// 
      useQr = 0;
      for(i = 0; i < na; i++)
         EigValues_elpa[i] = 0;
      
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      if(nprocs < 8000)
         pdsyevr_("V", "I", "L", &na, a, &one, &one, a_desc, &na, &na, &one, &nev, &new_f, &nev_f, EigValues_elpa, EigenVectors, &one, &one, EigenVectors_desc, work_find, &Lwork_find, Iwork, &Liwork, &info);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();
      localTime = endTime - startTime;

      MPI_Reduce(&localTime, &AverageTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localTime, &MaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
 
      if (myid == 0)
         printf("Solution ScaLAPACK is done, 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, AverageTime/nprocs, MaxTime);
      
      // create matrix of the EigenValues for the later check 
      for (i = 0; i < LocR_1*LocC; i++)
         lambda_Matr[i] = 0;
      for (i = 1; i <= nev; i++)
      {
         value = EigValues_elpa[i-1];
         pdelset_(lambda_Matr, &i, &i, lambda_Matr_desc, &value);
      }
            
      ////////////////////////////////////////////////////////////////////// back transform ///////////////////////////////////////////////////////////////////////
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      pdtrtrs_("L", "T", "N", &na, &nev, b, &one, &one, b_desc, EigenVectors, &one, &one, EigenVectors_desc, &info);  // now EigenVectors = L(-H)*x
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();
      back_transform_time = endTime - startTime; 
      MPI_Reduce(&back_transform_time, &back_average, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&back_transform_time, &back_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      
      if(myid == 0)
         printf("\n 1 step ScaLAPACK back transform is done, 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", back_transform_time, back_average/nprocs, back_max);

      //////////////////////////////////////// Check the results //////////////////////////////////////////////////
      
      pdsymm_("L", "L", &na, &nev, &done, a_copy, &one, &one, a_copy_desc, EigenVectors, &one, &one, EigenVectors_desc, &dzero, Ax, &one, &one, Ax_desc);
      pdsymm_("L", "L", &na, &nev, &done, b_copy, &one, &one, b_copy_desc, EigenVectors, &one, &one, EigenVectors_desc, &dzero, Bx, &one, &one, Bx_desc);
      pdsymm_("R", "L", &na, &nev, &done, lambda_Matr, &one, &one, lambda_Matr_desc, Bx, &one, &one, Bx_desc, &dzero, lambdaBx, &one, &one, lambdaBx_desc);
      
      diff = 0;
      for (i = 0; i < na_rows; i++)
         for (j = 0; j < LocC; j++)
            diff = diff + fabs(Ax[i + j*na_rows] - lambdaBx[i + j*na_rows]);   

      MPI_Reduce(&diff, &diff_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      if(myid == 0){
         printf("max accumulated diff of the Ax-lamBx = %.15e \n", diff_max);
         if(diff_max > TOL){
           printf("Results incorrect, difference exceeds tolerance\n");
           status = 1;
         }
         printf("_______________________________________________________________________________________________________\n");
      }
   }
   
////////////////////////////////////////////////////////////////////////////////////// free memory ///////////////////////////////////////////////////
   free(a);
   free(a_desc);
   free(b);
   free(b_desc);
   free(EigenVectors);
   free(EigenVectors_desc);
   free(EigValues_elpa);
   free(a_copy);
   free(a_copy_desc);
   free(b_copy_desc);
   free(b_copy);
   free(work1);
   free(work_find);
   free(Iwork);
   
   free(Ax);
   free(Ax_desc);
   free(Bx);
   free(Bx_desc);
   free(lambdaBx);
   free(lambdaBx_desc);
   free(lambda_Matr);
   free(lambda_Matr_desc);
 
#ifdef WITH_MPI
   MPI_Finalize();
#endif
   return status;
}

