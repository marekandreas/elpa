#include "config-f90.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// most of the file is not compiled if not using MPI
#ifdef WITH_MPI
#include <mpi.h>

//#include <elpa/elpa.h>
//#include <elpa/elpa_generated.h>
//#include <elpa/elpa_constants.h>
//#include <elpa/elpa_generated_legacy.h>
//#include <elpa/elpa_generic.h>
//#include <elpa/elpa_legacy.h>
//
void pdlacpy_(char*, int*, int*, double*, int*, int*, int*, double*, int*, int*, int*);
void dlacpy_(char*, int*, int*, double*, int*, double*, int*);
void dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*); 
void pdtran_(int*, int*, double*, double*, int*, int*, int*, double*, double*, int*, int*, int*);
//void pdelset_(double*, int*, int*, int*, double*);
//void pdsymm_(char*, char*, int*, int*, double*, double*, int*, int*, int*, double*, int*, int*, int*, double*, double*, int*, int*, int*);
//void pdpotrf_(char*, int*, double*, int*, int*, int*, int*);
//void pdsyngst_(int*, char*, int*, double*, int*, int*, int*, double*, int*, int*, int*, double*, double*, int*, int*);
//void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
int numroc_(int*, int*, int*, int*, int*);
//void set_up_blacsgrid_f1(int, int*, int*, int*, int*, int*, int*, int*);
//void pdtrtrs_(char*, char*, char*, int*, int*, double*, int*, int*, int*, double*, int*, int*, int*, int*);
//void pdsyevr_(char*, char*, char*, int*, double*, int*, int*, int*, int*, int*, int*, int*, int*, int*, double*, double*, int*, int*, int*, double*, int*, int*, int*, int*);

void d_cannons_triang_rectangular(double* U, double* B, int np_rows, int np_cols, int my_prow, int my_pcol, int* U_desc, int*b_desc, double *Res, MPI_Comm row_comm, MPI_Comm col_comm)
{
   // Cannons algorithm, Non-blocking version
   // Input: 
   //    - U is upper triangular matrix
   //    - B is rectangular matrix
   // Output: 
   //    - Res is a full rectangular matrix Res = U*B
   // row_comm: communicator along rows
   // col_comm: communicator along columns
   // This function will be used for a backtransformation
  
   int na, nb, nblk, width, na_rows, na_cols, nb_cols, cols_in_buffer_U_my_initial, cols_in_buffer_U, rows_in_buffer_U, Size_receive_U_now, rows_in_buffer_U_now, cols_in_buffer_U_now, rows_in_buffer_U_my_initial;

   int i, j, Size_send_U, Size_receive_U, Size_send_B, Size_receive_B, intNumber, Buf_rows, Buf_cols_U, Buf_cols_B, curr_rows, num_of_iters, cols_in_buffer, rows_in_block, curr_col_loc, cols_in_block, num_of_blocks_in_U_buffer, col_of_origin_U, b_rows_mult, b_cols_mult; 
   
   double *Buf_to_send_U, *Buf_to_receive_U, *Buf_to_send_B, *Buf_to_receive_B, *Buf_U, *PosBuff;
  
   int where_to_send_U, from_where_to_receive_U, where_to_send_B, from_where_to_receive_B, last_proc_col_B, last_proc_row_B, n, Size_U_stored, proc_col_min; 
   
   double *U_local_start, *Buf_pos, *B_local_start, *double_ptr, *CopyTo, *CopyFrom;
   
   int ratio;
   
   MPI_Status status;

   int one = 1;
   int zero = 0; 
   double done = 1.0;
   double dzero = 0.0;
      
   na = U_desc[2];
   nblk = U_desc[4]; 
   nb = b_desc[3];
   
   na_rows = numroc_(&na, &nblk, &my_prow, &zero, &np_rows);
   na_cols = numroc_(&na, &nblk, &my_pcol, &zero, &np_cols);
   nb_cols = numroc_(&nb, &nblk, &my_pcol, &zero, &np_cols);
   
   MPI_Request request_U_Recv; 
   MPI_Request request_U_Send;
   MPI_Request request_B_Recv; 
   MPI_Request request_B_Send;
   
   ///////////////////////////////////////////////////////////////// Start of algorithm ///////////////////////////////////////////////////////////////////////////////////////////////
   last_proc_col_B = ((nb-1)/nblk) % np_cols;
   last_proc_row_B = ((na-1)/nblk) % np_rows;
   
   /////////////////////////memory allocation area//////////////////////////////////////////////////////////////
   
    if(nb%nblk == 0)
      if(my_pcol <= last_proc_col_B)
         Buf_cols_B = nb_cols;
      else
         Buf_cols_B = nb_cols + nblk;      
   else
      if(my_pcol < last_proc_col_B)
         Buf_cols_B = nb_cols;
      else if(my_pcol > last_proc_col_B)
         Buf_cols_B = nb_cols + nblk; 
      else  // if my_pcol == last_proc_col_B
         Buf_cols_B = nb_cols + nblk - nb_cols%nblk;     
   
   if(na%nblk == 0)
      if(my_prow <= last_proc_row_B)
         Buf_rows = na_rows;
      else
         Buf_rows = na_rows + nblk;      
   else
      if(my_prow < last_proc_row_B)
         Buf_rows = na_rows;
      else if(my_prow > last_proc_row_B)
         Buf_rows = na_rows + nblk; 
      else  // if my_prow == last_proc_row_B
         Buf_rows = na_rows + nblk - na_rows%nblk;  
   
   ratio = np_cols/np_rows; 
   
   intNumber = ceil((double)na/(double)(np_cols*nblk));   // max. possible number of the local block columns of U
   Size_U_stored = ratio*nblk*nblk*intNumber*(intNumber+1)/2 + 2;   // number of local elements from the upper triangular part that every proc. has (max. possible value among all the procs.)
   
   Buf_to_send_U = malloc(ratio*Size_U_stored*sizeof(double));
   Buf_to_receive_U = malloc(ratio*Size_U_stored*sizeof(double));
   Buf_to_send_B = malloc(Buf_cols_B*Buf_rows*sizeof(double));
   Buf_to_receive_B = malloc(Buf_cols_B*Buf_rows*sizeof(double));
   if(ratio != 1)
      Buf_U = malloc(Size_U_stored*sizeof(double));   // in this case we will receive data into initial buffer and after place block-rows to the needed positions of buffer for calculation
    
   for(i = 0; i < na_rows*nb_cols; i++)
     Res[i] = 0; 
    
   /////////////////////////////////////////////////////////////// initial reordering of U ///////////////////////////////////////////////////////////////////////////////////////// 
      
   // here we assume, that np_rows < np_cols; then I will send to the number of processors equal to <ratio> with the "leap" equal to np_rows; the same holds for receive  
   if((ratio != 1)||(my_prow != 0))   // if grid is rectangular or my_prow is not 0
      Buf_pos = Buf_to_send_U;     // I will copy to the send buffer
   else
      Buf_pos = Buf_to_receive_U;  // if grid is square and my_prow is 0, then I will copy to the received buffer
      
   // form array to send by block-columns; we need only upper triangular part
   // find the first local block belonging to the upper part of matrix U
   if(my_pcol >= my_prow)  // if I am in the upper part of proc. grid
      curr_col_loc = 0;    // my first local block-column has block from the upper part of matrix
   else
      curr_col_loc = 1;   //ceil((double)(((double)my_prow - (double)my_pcol)/(double)np_cols)) always will give 1 since np_cols > np_rows 
      
   num_of_iters = ceil((double)na_cols/(double)nblk);             // number my of block-columns
   num_of_iters = num_of_iters - curr_col_loc;   // I will exclude the first <curr_col_loc> block-columns since they do not have blocks from the upper part of matrix U
   curr_col_loc = curr_col_loc*nblk;             // local index of the found block-column

   if(my_pcol >= my_prow )
      rows_in_block = ceil(((double)(my_pcol + 1) - (double)my_prow)/(double)np_rows)*nblk;
   else
      rows_in_block = ratio*nblk;
   cols_in_buffer_U_my_initial = 0;
   Size_send_U = 0; 
   for(i = 0; i < num_of_iters; i++)       // loop over my block-columns, which have blocks in the upepr part of U
   {      
      if(rows_in_block > na_rows)
         rows_in_block = na_rows; 

      if ((na_cols - curr_col_loc) < nblk)
         cols_in_block = na_cols - curr_col_loc;     // how many columns do I have in the current block-column
      else
         cols_in_block = nblk; 
      
      if((rows_in_block > 0)&&(cols_in_block > 0))
      {
         double_ptr = &U[curr_col_loc*na_rows];   // pointer to start of the current block-column to be copied to buffer
         dlacpy_("A", &rows_in_block, &cols_in_block, double_ptr, &na_rows, Buf_pos, &rows_in_block);     // copy upper part of block-column in the buffer with LDA = length of the upper part of block-column 
         Buf_pos = Buf_pos + rows_in_block*cols_in_block;                         // go to the position where the next block-column will be copied                                             
         Size_send_U = Size_send_U + rows_in_block*cols_in_block; 
         cols_in_buffer_U_my_initial = cols_in_buffer_U_my_initial + cols_in_block; 
      }
      curr_col_loc = curr_col_loc + nblk;      // go to the next local block-column of my local array U 
      rows_in_block = rows_in_block + ratio*nblk;
   }
   rows_in_buffer_U_my_initial = rows_in_block - ratio*nblk;    // remove redundant addition from the previous loop 
   *Buf_pos = (double)cols_in_buffer_U_my_initial; // write number of the columns at the end of the buffer; we will need this for furhter multiplications on the other processors
   Buf_pos = Buf_pos + 1; 
   *Buf_pos = (double)rows_in_buffer_U_my_initial; // write number of the rows at the end of the buffer; we will need this for furhter multiplications on the other processors
   Size_send_U = Size_send_U + 2;
   
   // now we have the local buffer to send
   // find the lowest processor column among those who will send me
   proc_col_min = np_cols; 
   for(i = 0; i < ratio; i++)
   {
      from_where_to_receive_U = (my_pcol + my_prow + i*np_rows)%np_cols;
      if(from_where_to_receive_U < proc_col_min)
         proc_col_min = from_where_to_receive_U;
   }
   
   // do communications and form local buffers for calculations
   Size_receive_U = 0;       // size of the accumulated buffer
   cols_in_buffer_U = 0;     // number of columns in the accumulated buffer
   rows_in_buffer_U = 0;     // number of rows in the accumulated buffer
   for(i = 0; i < ratio; i++)
   {
      where_to_send_U = (my_pcol - my_prow - i*np_rows + np_cols)%np_cols;                
      from_where_to_receive_U = (my_pcol + my_prow + i*np_rows)%np_cols;
      
      // send and receive in the row_comm
      if(ratio != 1)   // if grid is not square
      {
         if(where_to_send_U != my_pcol)   // if I need to send and receive on this step
         {
            MPI_Sendrecv(Buf_to_send_U, Size_send_U, MPI_DOUBLE, where_to_send_U, 0, Buf_U, Size_U_stored, MPI_DOUBLE, from_where_to_receive_U, 0, row_comm, &status);
            MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_U_now);
            Size_receive_U = Size_receive_U + Size_receive_U_now - 2; // we need only number of elements, so exclude information about cols_in_buffer_U and rows_in_buffer_U
            
            cols_in_buffer_U_now = Buf_U[Size_receive_U_now - 2];
            cols_in_buffer_U = cols_in_buffer_U + cols_in_buffer_U_now;
            rows_in_buffer_U_now = Buf_U[Size_receive_U_now - 1];
            
            if(rows_in_buffer_U < rows_in_buffer_U_now)
               rows_in_buffer_U = rows_in_buffer_U_now; 

            intNumber = from_where_to_receive_U/np_rows; // how many processors will send me blocks, such that they will be placed before the current blocks  
            if(proc_col_min >= my_prow)   // if among procs who will send me there is one with the full sets of block-rows in the upper part
               CopyTo = &Buf_to_receive_U[nblk*nblk*intNumber*(intNumber + 1)/2];  // here I will copy to; formula based on arithm. progression
            else                         // if among procs who will send me there is one from the lower part of grid
               if(from_where_to_receive_U < my_prow)   // if I have just received from this processor from the lower part
                  CopyTo = &Buf_to_receive_U[nblk*nblk*ratio*(ratio - 1)/2];  // copy the first block of this processor after the first blocks from the others procs. that will send me later (the first block-column of this proc. is in the lower part of matrix)
               else
                  CopyTo = &Buf_to_receive_U[nblk*nblk*intNumber*(intNumber - 1)/2];
            CopyFrom = Buf_U; 
         }
         else  // if I need to send to myself on this step, then I will copy from Buf_to_send_U to Buf_to_receive_U
         {
            cols_in_buffer_U_now = cols_in_buffer_U_my_initial;
            cols_in_buffer_U = cols_in_buffer_U + cols_in_buffer_U_now; 
            
            rows_in_buffer_U_now = rows_in_buffer_U_my_initial;
            if(rows_in_buffer_U < rows_in_buffer_U_now)
               rows_in_buffer_U = rows_in_buffer_U_now; 

            intNumber = my_pcol/np_rows; // how many processors will send me blocks, such that they will be placed before the current blocks  
            if(proc_col_min >= my_prow)   // if among procs who will send me there is one with the full sets of block-rows in the upper part
               CopyTo = &Buf_to_receive_U[nblk*nblk*intNumber*(intNumber + 1)/2];  // here I will copy to; formula based on arithm. progression
            else                         // if among procs who will send me there is one from the lower part of grid
               if(my_pcol < my_prow)   // if I have just received from this processor from the lower part (in this case it is me)
                  CopyTo = &Buf_to_receive_U[nblk*nblk*ratio*(ratio - 1)/2];  // copy the first block of this processor after the first blocks from the others procs. that will send me later (the first block-column of this proc. is in the lower part of matrix)
               else
                  CopyTo = &Buf_to_receive_U[nblk*nblk*intNumber*(intNumber - 1)/2];
            CopyFrom = Buf_to_send_U;  
            Size_receive_U = Size_receive_U + Size_send_U - 2;
         }
            
         // copy by block-columns
         intNumber = ceil((double)cols_in_buffer_U_now/(double)nblk);  // how many block-columns I have received on this iteration
         if(from_where_to_receive_U >= my_prow)
            rows_in_block = ceil(((double)(from_where_to_receive_U + 1) - (double)my_prow)/(double)np_rows)*nblk;  // number of rows in the first block-column of U buffer
         else
            rows_in_block = ratio*nblk; 
         for(j = 0; j < intNumber; j++)
         {
            if((j+1)*nblk < cols_in_buffer_U_now)
               cols_in_block = nblk; 
            else
               cols_in_block = cols_in_buffer_U_now - j*nblk;
               
            dlacpy_("A", &rows_in_block, &cols_in_block, CopyFrom, &rows_in_block, CopyTo, &rows_in_block);

            CopyFrom = CopyFrom + rows_in_block*cols_in_block; 
            CopyTo = CopyTo + ratio*rows_in_block*nblk + nblk*nblk*ratio*(ratio-1)/2;  // I need to leave place for ratio block-columns of the other procs. of the lengths rows_in_block, (rows_in_block+nblk), (rows_in_block+2*nblk) and so on
            rows_in_block = rows_in_block + ratio*nblk;     // number of rows in the next block-columns
            if(rows_in_block > rows_in_buffer_U_now)
               rows_in_block = rows_in_buffer_U_now; 
         }
      }
      else    // if grid is square
      {
         if(my_prow > 0)
         {
            MPI_Sendrecv(Buf_to_send_U, Size_send_U, MPI_DOUBLE, where_to_send_U, 0, Buf_to_receive_U, Size_U_stored, MPI_DOUBLE, from_where_to_receive_U, 0, row_comm, &status);
            MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_U);
            cols_in_buffer_U = (int)Buf_to_receive_U[Size_receive_U-2];
            rows_in_buffer_U = (int)Buf_to_receive_U[Size_receive_U-1];
         }
         else    // if my_prow == 0, then I have already everything in my Buf_to_receive_U buffer
         {
            Size_receive_U = Size_send_U;
            rows_in_buffer_U = rows_in_buffer_U_my_initial;
            cols_in_buffer_U = cols_in_buffer_U_my_initial;
         }
      }
   }
   if(ratio != 1)
   {
      Buf_to_receive_U[Size_receive_U] = cols_in_buffer_U;
      Buf_to_receive_U[Size_receive_U + 1] = rows_in_buffer_U;
      Size_receive_U = Size_receive_U + 2;
   }
      
   ////////////////////////////////////////////////////////////// initial reordering of B ///////////////////////////////////////////////////////////////////////////////////////// 
   
   if(my_pcol > 0)
   {
      where_to_send_B = (my_prow - my_pcol + np_cols)%np_rows;                   // shift = my_pcol
      from_where_to_receive_B = (my_pcol + my_prow)%np_rows;

      // send and receive in the row_comm
      if(where_to_send_B != my_prow)                  // for the rectangular proc grids it may be possible that I need to "send to myself"; if it is not the case, then I send
      {
         // form array to send
         dlacpy_("A", &na_rows, &nb_cols, B, &na_rows, Buf_to_send_B, &na_rows);
         MPI_Sendrecv(Buf_to_send_B, nb_cols*na_rows, MPI_DOUBLE, where_to_send_B, 0, Buf_to_receive_B, nb_cols*Buf_rows, MPI_DOUBLE, from_where_to_receive_B, 0, col_comm, &status); 
         MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_B); // find out how many elements I have received 
         Size_receive_B = Size_receive_B/nb_cols;    // how many rows I have received
      }
      else
      {
         dlacpy_("A", &na_rows, &nb_cols, B, &na_rows, Buf_to_receive_B, &na_rows); // else I copy data like I have "received" it
         Size_receive_B = na_rows;
      }
   }
   else
   {
      dlacpy_("A", &na_rows, &nb_cols, B, &na_rows, Buf_to_receive_B, &na_rows);        // if I am in the 0 proc row, I need not to send; so copy data like I have "received" it
      Size_receive_B = na_rows; 
   }   
   
   //////////////////////////////////////////////////////////////////////// main loop ////////////////////////////////////////////////////////////////////////////////
   where_to_send_U = (my_pcol - 1 + np_cols)%np_cols;
   from_where_to_receive_U = (my_pcol + 1)%np_cols;
   where_to_send_B = (my_prow - 1 + np_rows)%np_rows;
   from_where_to_receive_B = (my_prow + 1)%np_rows;    

   for(i = 1; i < np_rows; i++)
   {
      // at this moment I need to send to neighbour what I have in the "received" arrays; that is why change pointers of the "received" and "send" arrays
      double_ptr = Buf_to_send_U; 
      Buf_to_send_U = Buf_to_receive_U; 
      Buf_to_receive_U = double_ptr; 
      
      double_ptr = Buf_to_send_B; 
      Buf_to_send_B = Buf_to_receive_B; 
      Buf_to_receive_B = double_ptr;
            
      Size_send_U = Size_receive_U;
      Size_send_B = Size_receive_B;                   
        
      ///// shift for U ////////////////////////////////////////////////////////////
      MPI_Isend(Buf_to_send_U, Size_send_U, MPI_DOUBLE, where_to_send_U, 0, row_comm, &request_U_Send); 
      MPI_Irecv(Buf_to_receive_U, ratio*Size_U_stored, MPI_DOUBLE, from_where_to_receive_U, 0, row_comm, &request_U_Recv);
         
      ///// shift for B /////////////////////////////////////////////      
      MPI_Isend(Buf_to_send_B, Size_send_B*nb_cols, MPI_DOUBLE, where_to_send_B, 0, col_comm, &request_B_Send); 
      MPI_Irecv(Buf_to_receive_B, Buf_rows*nb_cols, MPI_DOUBLE, from_where_to_receive_B, 0, col_comm, &request_B_Recv);
      
      ///// multiplication ////////////////////////////////////////////////////////////////////////////////////////////
      cols_in_buffer_U = (int)Buf_to_send_U[Size_receive_U-2];
      rows_in_buffer_U = (int)Buf_to_send_U[Size_receive_U-1];
      //find minimal proc. column among those procs. who contributed in the current U buffer
      proc_col_min = np_cols; 
      for(j = 0; j < ratio; j++)
      {
         col_of_origin_U = (my_pcol + my_prow + i - 1 + j*np_rows)%np_cols;
         if(col_of_origin_U < proc_col_min)
            proc_col_min = col_of_origin_U;
      }
      col_of_origin_U = proc_col_min;
      
      num_of_blocks_in_U_buffer = ceil((double)cols_in_buffer_U/(double)nblk); 
      
      if (col_of_origin_U >= my_prow)
         B_local_start = Buf_to_send_B;
      else 
         B_local_start = Buf_to_send_B + nblk;
      
      U_local_start = Buf_to_send_U;
      
      for(j = 0; j < num_of_blocks_in_U_buffer; j++)
      {
         curr_rows = (j+1)*nblk;
         if (curr_rows > rows_in_buffer_U)
            curr_rows = rows_in_buffer_U; 
         
         if((j+1)*nblk <= cols_in_buffer_U)
            b_rows_mult = nblk; 
         else
            b_rows_mult = cols_in_buffer_U - j*nblk;
         
         dgemm_("N", "N", &curr_rows, &nb_cols, &b_rows_mult, &done, U_local_start, &curr_rows, B_local_start, &Size_receive_B, &done, Res, &na_rows); 
  
         U_local_start = U_local_start + nblk*curr_rows; 
         B_local_start = B_local_start + nblk; 
      }
      
      MPI_Wait(&request_U_Send, &status);
      MPI_Wait(&request_U_Recv, &status);
      MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_U); // find out how many elements I have received 
      
      MPI_Wait(&request_B_Send, &status);
      MPI_Wait(&request_B_Recv, &status);
      MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_B); // find out how many elements I have received 
      Size_receive_B = Size_receive_B/nb_cols;    // how many rows I have received
   }         
   
   // last iteration 
   cols_in_buffer_U = (int)Buf_to_receive_U[Size_receive_U-2];
   rows_in_buffer_U = (int)Buf_to_receive_U[Size_receive_U-1];
   //find minimal proc. column among those procs. who contributed in the current U buffer
   proc_col_min = np_cols; 
   for(j = 0; j < ratio; j++)
   {
      col_of_origin_U = (my_pcol + my_prow + np_rows - 1 + j*np_rows)%np_cols;
      if(col_of_origin_U < proc_col_min)
         proc_col_min = col_of_origin_U;
   }
   col_of_origin_U = proc_col_min;
      
   num_of_blocks_in_U_buffer = ceil((double)cols_in_buffer_U/(double)nblk);
  
   if (col_of_origin_U >= my_prow)
      B_local_start = Buf_to_receive_B;
   else 
      B_local_start = Buf_to_receive_B + nblk;
      
   U_local_start = Buf_to_receive_U;  
   
   for(j = 0; j < num_of_blocks_in_U_buffer; j++)
   {
      curr_rows = (j+1)*nblk;
      if (curr_rows > rows_in_buffer_U)
         curr_rows = rows_in_buffer_U; 
      
      if((j+1)*nblk <= cols_in_buffer_U)
         b_rows_mult = nblk; 
      else
         b_rows_mult = cols_in_buffer_U - j*nblk;
      
      dgemm_("N", "N", &curr_rows, &nb_cols, &b_rows_mult, &done, U_local_start, &curr_rows, B_local_start, &Size_receive_B, &done, Res, &na_rows); 

      U_local_start = U_local_start + nblk*curr_rows; 
      B_local_start = B_local_start + nblk;
   }
   
   free(Buf_to_send_U);
   free(Buf_to_receive_U);
   free(Buf_to_send_B);
   free(Buf_to_receive_B);
   if(ratio != 1)
      free(Buf_U);
}

#endif

//***********************************************************************************************************
/*
!f> interface
!f>   subroutine cannons_triang_rectangular(U, B, local_rows, local_cols, np_rows, np_cols, my_prow, my_pcol, u_desc, b_desc, &
!f>                                Res, row_comm, col_comm) &
!f>                             bind(C, name="d_cannons_triang_rectangular_c")
!f>     use, intrinsic :: iso_c_binding
!f>     real(c_double)                        :: U(local_rows, local_cols), B(local_rows, local_cols), Res(local_rows, local_cols)
!f>     integer(kind=c_int)                   :: u_desc(9), b_desc(9)
!f>     integer(kind=c_int),value             :: local_rows, local_cols
!f>     integer(kind=c_int),value     :: np_rows, np_cols, my_prow, my_pcol, row_comm, col_comm
!f>   end subroutine
!f> end interface
*/
void d_cannons_triang_rectangular_c(double* U, double* B, int local_rows, int local_cols, int np_rows, int np_cols,
                                     int my_prow, int my_pcol, int* u_desc, int* b_desc, double *Res, int row_comm, int col_comm)
{
#ifdef WITH_MPI
  MPI_Comm c_row_comm = MPI_Comm_f2c(row_comm);
  MPI_Comm c_col_comm = MPI_Comm_f2c(col_comm);

//  int c_my_prow, c_my_pcol;
//  MPI_Comm_rank(c_row_comm, &c_my_prow);
//  MPI_Comm_rank(c_col_comm, &c_my_pcol);
//  printf("FORT<->C row: %d<->%d, col: %d<->%d\n", my_prow, c_my_prow, my_pcol, c_my_pcol);

  // BEWARE
  // in the cannons algorithm, column and row communicators are exchanged
  // What we usually call row_comm in elpa, is thus passed to col_comm parameter of the function and vice versa
  // (order is swapped in the following call)
  // It is a bit unfortunate, maybe it should be changed in the Cannon algorithm to comply with ELPA standard notation?
  d_cannons_triang_rectangular(U, B, np_rows, np_cols, my_prow, my_pcol, u_desc, b_desc, Res, c_col_comm, c_row_comm);
#else
  printf("Internal error: Cannons algorithm should not be called without MPI, stopping...\n");
  exit(1);
#endif
}

