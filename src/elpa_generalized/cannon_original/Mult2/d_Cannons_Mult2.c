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

void dlacpy_(char*, int*, int*, double*, int*, double*, int*);
void dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*); 
void pdtran_(int*, int*, double*, double*, int*, int*, int*, double*, double*, int*, int*, int*);
void pdtrmm_(char*, char*, char*, char*, int*, int*, double*, double*, int*, int*, int*, double*, int*, int*, int*);
void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
int numroc_(int*, int*, int*, int*, int*);
void set_up_blacsgrid_f1(int, int*, int*, int*, int*, int*, int*, int*);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////// My function for multiplication 2 //////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void d_Cannons_Mult2(double* L, double* U, int np_rows, int np_cols, int my_prow, int my_pcol, int* a_desc, double *Res, MPI_Comm row_comm, MPI_Comm col_comm)
{
   // Input matrices: 
   //   - L: lower triangular matrix
   //   - U: upper triangular matrix
   // Output matrix: 
   //   - Lower triangular part of L*U 
   // row_comm: communicator along rows
   // col_comm: communicator along columns

   int na, nblk;

   int i, j, ii, Size_send_L, Size_receive_L, Size_send_U, Size_receive_U, num_of_blocks_in_U_buffer, row_of_origin_U, col_of_origin_L, Nb, owner; 
   
   double *Buf_to_send_L, *Buf_to_receive_L, *Buf_to_send_U, *Buf_to_receive_U, *U_local_start_curr, *CopyFrom, *CopyTo;
  
   int curr_col_loc, where_to_send_L, from_where_to_receive_L, where_to_send_U, from_where_to_receive_U, rows_in_block, cols_in_block, cols_in_buffer_L, cols_in_buffer_L_my_initial, rows_in_buffer_L, rows_in_buffer_L_my_initial, cols_in_buffer_U, rows_in_buffer_U;
   
   double *L_local_start, *Buf_pos, *U_local_start, *double_ptr, *Res_ptr, *Buf_L;
   
   int LDA_L, rows_in_block_U_curr, ratio, rows_in_buffer, proc_col_min, num_of_iters, rows_in_block_U, curr_row_loc; 
   
   int curr_col_loc_res, curr_row_loc_res, curr_row_loc_L, curr_col_loc_U, curr_col_glob_res, L_local_index, LDA_L_new, index_row_L_for_LDA, Size_receive_L_now, cols_in_buffer_L_now, rows_in_buffer_L_now, intNumber, Size_U_stored; 
   
   MPI_Status status;

   int one = 1;
   int zero = 0; 
   double done = 1.0;
   double dzero = 0.0;
   int na_rows, na_cols;
      
   na = a_desc[2];
   nblk = a_desc[4]; 
   na_rows = numroc_(&na, &nblk, &my_prow, &zero, &np_rows);
   na_cols = numroc_(&na, &nblk, &my_pcol, &zero, &np_cols);
   
   MPI_Request request_L_Recv; 
   MPI_Request request_L_Send;
   MPI_Request request_U_Recv; 
   MPI_Request request_U_Send;
   
   if (np_cols%np_rows != 0)
   {
      if((my_prow == 0)&& (my_pcol ==0))
         printf("!!!!! np_cols must be a multiple of np_rows!!!!! I do nothing! \n");
      return;
   }
   if (np_cols < np_rows != 0)
   {
      if((my_prow == 0)&& (my_pcol ==0))
         printf("np_cols < np_rows \n");
      return;
   }
   
   ratio = np_cols/np_rows; 
   
   ///////////////////////////////////////////////////////////// Start of algorithm ///////////////////////////////////////////////////////////////////////////////////////////////
   
   /////////////////////////memory allocation area//////////////////////////////////////////////////////////////
   
   intNumber = ceil((double)na/(double)(np_cols*nblk));   // max. possible number of the local block columns of U
   Size_U_stored = ratio*nblk*nblk*intNumber*(intNumber+1)/2 + 2;   // number of local elements from the upper triangular part that every proc. has (max. possible value among all the procs.)
     
   Buf_to_send_L = malloc(ratio*Size_U_stored*sizeof(double));
   Buf_to_receive_L = malloc(ratio*Size_U_stored*sizeof(double));
   Buf_to_send_U = malloc(Size_U_stored*sizeof(double));
   Buf_to_receive_U = malloc(Size_U_stored*sizeof(double));
   if(ratio != 1)
      Buf_L = malloc(Size_U_stored*sizeof(double));   // in this case we will receive data into initial buffer and after place block-columns to the needed positions of buffer for calculation
                    
   /////////////////////////////////////////////////////////////// initial reordering of L ///////////////////////////////////////////////////////////////////////////////////////// 
   // here we assume, that np_rows < np_cols; then I will send to the number of processors equal to <ratio> with the "leap" equal to np_rows; the same holds for receive  
   if((ratio != 1)||(my_prow != 0))   // if grid is rectangular or my_prow is not 0
      Buf_pos = Buf_to_send_L;     // I will copy to the send buffer
   else
      Buf_pos = Buf_to_receive_L;  // if grid is square and my_prow is 0, then I will copy to the received buffer
   
   // form array to send by block-columns; we need only lower triangular part
   num_of_iters = ceil((double)na_cols/(double)nblk);             // number my of block-columns
   
   cols_in_buffer_L_my_initial = 0;
   Size_send_L = 0; 
   
   if(my_pcol <= my_prow)  // if I am from the lower part of grid
   {
      curr_row_loc = 0;     // I will copy all my block-rows
      rows_in_buffer_L_my_initial = na_rows;
   }
   else
   {
      curr_row_loc = ceil((double)(((double)my_pcol - (double)my_prow)/(double)np_rows))*nblk; // I will skip some of my block-rows
      rows_in_buffer_L_my_initial = na_rows - curr_row_loc;   
   }
       
   for(i = 0; i < num_of_iters; i++)       // loop over my block-columns
   {
      curr_col_loc = i*nblk;      // local index of start of the current block-column 
      rows_in_block = na_rows - curr_row_loc;    // how many rows do I have in the lower part of the current block-column
      
      if ((na_cols - curr_col_loc) < nblk)
         cols_in_block = na_cols - curr_col_loc;     // how many columns do I have in the block-column
      else
         cols_in_block = nblk; 
      
      if((rows_in_block > 0)&&(cols_in_block > 0))
      {
         L_local_start = &L[curr_col_loc*na_rows + curr_row_loc];
         dlacpy_("A", &rows_in_block, &cols_in_block, L_local_start, &na_rows, Buf_pos, &rows_in_block);     // copy lower part of block-column in the buffer with LDA = length of the lower part of block-column 
         Buf_pos = Buf_pos + rows_in_block*cols_in_block;
         Size_send_L = Size_send_L + rows_in_block*cols_in_block; 
         cols_in_buffer_L_my_initial = cols_in_buffer_L_my_initial + cols_in_block; 
      }
      curr_row_loc = curr_row_loc + ratio*nblk;
   }
   *Buf_pos = (double)cols_in_buffer_L_my_initial; // write number of the columns at the end of the buffer; we will need this for furhter multiplications on the other processors
   Size_send_L = Size_send_L + 1;
   
   // now we have the local buffer to send
   // find the lowest processor column among those who will send me
   proc_col_min = np_cols; 
   for(i = 0; i < ratio; i++)
   {
      from_where_to_receive_L = (my_pcol + my_prow + i*np_rows)%np_cols;
      if(from_where_to_receive_L < proc_col_min)
         proc_col_min = from_where_to_receive_L;
   }
   // do communications and form local buffers for calculations
   Size_receive_L = 0;       // size of the accumulated buffer
   cols_in_buffer_L = 0;     // number of columns in the accumulated buffer
   rows_in_buffer_L = 0;     // number of rows in the accumulated buffer
   for(i = 0; i < ratio; i++)
   {
      where_to_send_L = (my_pcol - my_prow - i*np_rows + np_cols)%np_cols;                
      from_where_to_receive_L = (my_pcol + my_prow + i*np_rows)%np_cols;
      
      // send and receive in the row_comm
      if(ratio != 1)   // if grid is not square
      {
         if(where_to_send_L != my_pcol)   // if I need to send and receive on this step
         {
            MPI_Sendrecv(Buf_to_send_L, Size_send_L, MPI_DOUBLE, where_to_send_L, 0, Buf_L, Size_U_stored, MPI_DOUBLE, from_where_to_receive_L, 0, row_comm, &status);
            MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_L_now);
            Size_receive_L = Size_receive_L + Size_receive_L_now - 1; // we need only number of elements, so exclude information about cols_in_buffer_L
            
            cols_in_buffer_L_now = Buf_L[Size_receive_L_now-1];
            cols_in_buffer_L = cols_in_buffer_L + cols_in_buffer_L_now; 
            
            // determine number of rows in the received buffer
            if(from_where_to_receive_L <= my_prow)  // if source is from the lower part of grid
            {
               rows_in_buffer_L_now = na_rows;
            }
            else
            {
               rows_in_buffer_L_now = na_rows - ceil((double)(((double)from_where_to_receive_L - (double)my_prow)/(double)np_rows))*nblk; // some of the block-rows have been skipped
            }
            if(rows_in_buffer_L < rows_in_buffer_L_now)
               rows_in_buffer_L = rows_in_buffer_L_now; 

            intNumber = from_where_to_receive_L/np_rows; // how many processors will send me blocks, such that they will be placed before the current blocks  
            if(proc_col_min <= my_prow)   // if among procs who will send me there is one with the full sets of block-rows in the lower part
               CopyTo = &Buf_to_receive_L[nblk*(na_rows*intNumber - nblk*(intNumber-1)*intNumber/2)];  // here I will copy to; formula based on arithm. progression
            else
               CopyTo = &Buf_to_receive_L[nblk*(na_rows*intNumber - nblk*intNumber*(intNumber+1)/2)];  // otherwise, the first block-column will be shorter by one block
            CopyFrom = Buf_L; 
         }
         else  // if I need to send to myself on this step, then I will copy from Buf_to_send_L to Buf_to_receive_L
         {
            cols_in_buffer_L_now = cols_in_buffer_L_my_initial;
            cols_in_buffer_L = cols_in_buffer_L + cols_in_buffer_L_now; 
            
            rows_in_buffer_L_now = rows_in_buffer_L_my_initial;
            if(rows_in_buffer_L < rows_in_buffer_L_now)
               rows_in_buffer_L = rows_in_buffer_L_now; 

            intNumber = my_pcol/np_rows; // how many processors will send me blocks, such that they will be placed before the current blocks  
            if(proc_col_min <= my_prow)   // if among procs who will send me there is one with the full sets of block-rows in the lower part
               CopyTo = &Buf_to_receive_L[nblk*(na_rows*intNumber - nblk*(intNumber-1)*intNumber/2)];  // here I will copy to; formula based on arithm. progression
            else
               CopyTo = &Buf_to_receive_L[nblk*(na_rows*intNumber - nblk*intNumber*(intNumber+1)/2)];  // otherwise, the first block-column will be shorter by one block
            CopyFrom = Buf_to_send_L;  

            Size_receive_L = Size_receive_L + Size_send_L - 1;
         }
            
         // copy by block-columns
         intNumber = ceil((double)cols_in_buffer_L_now/(double)nblk);  // how many block-columns I have received on this iteration
         rows_in_block = rows_in_buffer_L_now; 
         for(j = 0; j < intNumber; j++)
         {
            if((j+1)*nblk < cols_in_buffer_L_now)
               cols_in_block = nblk; 
            else
               cols_in_block = cols_in_buffer_L_now - j*nblk;
               
            dlacpy_("A", &rows_in_block, &cols_in_block, CopyFrom, &rows_in_block, CopyTo, &rows_in_block);

            CopyFrom = CopyFrom + rows_in_block*cols_in_block; 
            CopyTo = CopyTo + nblk*(ratio*rows_in_block - nblk*(ratio-1)*ratio/2);  // I need to leave place for ratio block-columns of the other procs. of the lengths rows_in_block, (rows_in_block-nblk), (rows_in_block-2*nblk) and so on
            rows_in_block = rows_in_block - ratio*nblk;     // number of rows in the next block-columns
         }
      }
      else    // if grid is square
      {
         if(my_prow > 0)
         {
            MPI_Sendrecv(Buf_to_send_L, Size_send_L, MPI_DOUBLE, where_to_send_L, 0, Buf_to_receive_L, Size_U_stored, MPI_DOUBLE, from_where_to_receive_L, 0, row_comm, &status);
            MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_L);
            cols_in_buffer_L = (int)Buf_to_receive_L[Size_receive_L-1];
            if(from_where_to_receive_L <= my_prow)  // if source is from the lower part of grid
            {
               rows_in_buffer_L = na_rows;
            }
            else
            {
               rows_in_buffer_L = na_rows - ceil((double)(((double)from_where_to_receive_L - (double)my_prow)/(double)np_rows))*nblk; // some of the block-rows have been skipped
            }
         }
         else    // if my_prow == 0, then I have already everything in my Buf_to_receive_L buffer
         {
            Size_receive_L = Size_send_L;
            rows_in_buffer_L = rows_in_buffer_L_my_initial;
            cols_in_buffer_L = cols_in_buffer_L_my_initial;
         }
      }
   }
   if(ratio != 1)
   {
      Buf_to_receive_L[Size_receive_L] = cols_in_buffer_L;
      Buf_to_receive_L[Size_receive_L + 1] = rows_in_buffer_L;
      Size_receive_L = Size_receive_L + 2;
   }
   else
   {
      Buf_to_receive_L[Size_receive_L] = rows_in_buffer_L;
      Size_receive_L = Size_receive_L + 1;
   }

   ////////////////////////////////////////////////////////////// initial reordering of U //////////////////////////////////////////////////////////////////////////////////////////
   // form array to send by block-columns
   num_of_iters = ceil((double)na_cols/(double)nblk);             // number my of block-columns
   
   where_to_send_U = (my_prow - my_pcol + np_cols)%np_rows;                 // shift = my_pcol; we assume that np_cols%np_rows = 0
   from_where_to_receive_U = (my_pcol + my_prow)%np_rows;
   
   if(where_to_send_U == my_prow)    // if I will not need to send my local part of U, then copy my local data to the "received" buffer
      Buf_pos = Buf_to_receive_U;
   else
      Buf_pos = Buf_to_send_U;         // else form the array to send
   Size_send_U = 0;    // we already have 1 element in the buffer
   
   // find the first local block belonging to the upper part of matrix U
   if(my_pcol >= my_prow)  // if I am in the upper part of proc. grid
      curr_col_loc = 0;    // my first local block-column has block from the upper part of matrix
   else
      curr_col_loc = 1;   //ceil((double)(((double)my_prow - (double)my_pcol)/(double)np_cols)) always will give 1 since np_cols > np_rows 
      
   num_of_iters = num_of_iters - curr_col_loc;   // I will exclude the first <curr_col_loc> block-columns since they do not have blocks from the upper part of matrix U
   curr_col_loc = curr_col_loc*nblk;             // local index of the found block-column

   if(my_pcol >= my_prow )
      rows_in_block = ceil(((double)(my_pcol + 1) - (double)my_prow)/(double)np_rows)*nblk;
   else
      rows_in_block = ratio*nblk;
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
      }
      curr_col_loc = curr_col_loc + nblk;      // go to the next local block-column of my local array U 
      rows_in_block = rows_in_block + ratio*nblk;
   }
   rows_in_buffer = rows_in_block - ratio*nblk;    // remove redundant addition from the previous loop 
   *Buf_pos = (double)rows_in_buffer; // write number of the rows at the end of the buffer; we will need this for furhter multiplications on the other processors
   Size_send_U = Size_send_U + 1;
   
   //send and receive
   if(where_to_send_U != my_prow)
   {   
      // send and receive in the col_comm
      MPI_Sendrecv(Buf_to_send_U, Size_send_U, MPI_DOUBLE, where_to_send_U, 0, Buf_to_receive_U, Size_U_stored, MPI_DOUBLE, from_where_to_receive_U, 0, col_comm, &status); 
      MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_U); // find out how many elements I have received 
   }
   else // if I do not need to send 
      Size_receive_U = Size_send_U;         // how many rows I "have received"; the needed data I have already copied to the "receive" buffer
   
   //////////////////////////////////////////////////////////////////////// main loop ////////////////////////////////////////////////////////////////////////////////
   where_to_send_L = (my_pcol - 1 + np_cols)%np_cols;
   from_where_to_receive_L = (my_pcol + 1)%np_cols;
   where_to_send_U = (my_prow - 1 + np_rows)%np_rows;
   from_where_to_receive_U = (my_prow + 1)%np_rows;
  
   for(j = 1; j < np_rows; j++)
   {
      // at this moment I need to send to neighbour what I have in the "received" arrays; that is why exchange pointers of the "received" and "send" arrays
      double_ptr = Buf_to_send_L; 
      Buf_to_send_L = Buf_to_receive_L; 
      Buf_to_receive_L = double_ptr; 
      
      double_ptr = Buf_to_send_U; 
      Buf_to_send_U = Buf_to_receive_U; 
      Buf_to_receive_U = double_ptr;
        
      ///// shift for L ////////////////////////////////////////////////////////////
      Size_send_L = Size_receive_L; 
      MPI_Isend(Buf_to_send_L, Size_send_L, MPI_DOUBLE, where_to_send_L, 0, row_comm, &request_L_Send); 
      MPI_Irecv(Buf_to_receive_L, ratio*Size_U_stored, MPI_DOUBLE, from_where_to_receive_L, 0, row_comm, &request_L_Recv);
         
      ///// shift for U /////////////////////////////////////////////
      Size_send_U = Size_receive_U; 
      MPI_Isend(Buf_to_send_U, Size_send_U, MPI_DOUBLE, where_to_send_U, 0, col_comm, &request_U_Send); 
      MPI_Irecv(Buf_to_receive_U, Size_U_stored, MPI_DOUBLE, from_where_to_receive_U, 0, col_comm, &request_U_Recv); 
      
      ///// multiplication ////////////////////////////////////////////////////////////////////////////////////////////
      rows_in_buffer_U = (int)Buf_to_send_U[Size_receive_U-1];
      row_of_origin_U = (my_pcol + my_prow + np_cols + j - 1)%np_rows;
      if(my_pcol >= row_of_origin_U)
         cols_in_buffer_U = na_cols;
      else
         cols_in_buffer_U = na_cols - nblk;
      
      cols_in_buffer_L = (int)Buf_to_send_L[Size_receive_L-2];
      rows_in_buffer_L = (int)Buf_to_send_L[Size_receive_L-1];
      // find the minimal pcol among those who have sent L for this iteration
      col_of_origin_L = np_cols; 
      for(i = 0; i < ratio; i++)
      {
         intNumber = (my_pcol + my_prow + i*np_rows + np_cols + j - 1)%np_cols;
         if(intNumber < col_of_origin_L)
            col_of_origin_L = intNumber;
      }
      
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // find block-column of the result to start update with
      if (my_pcol >= row_of_origin_U)   // if origin of U is from the upper part 
         curr_col_loc_res = 0;          // then I update all columns of Result    
      else
         curr_col_loc_res = nblk;       // the first block column of U corresponds to my second one and I do not need to update the first block-column
      
      num_of_blocks_in_U_buffer = ceil((double)((double)cols_in_buffer_U/(double)nblk)); 
      if(my_pcol >= row_of_origin_U)    // if origin of U is from the upper part
         rows_in_block_U = ceil(((double)(my_pcol + 1) - (double)row_of_origin_U)/(double)np_rows)*nblk;  // blocks in the first block-column of U buffer
      else
         rows_in_block_U = ratio*nblk;
      
      U_local_start = &Buf_to_send_U[0];
      
      for (i = 0; i < num_of_blocks_in_U_buffer; i++)
      { 
         // find block-row of the result to start update with; we need to update only lower triangular part of result
         curr_col_glob_res = np_cols*nblk*(curr_col_loc_res/nblk) + curr_col_loc_res%nblk + ((np_cols+my_pcol)%np_cols)*nblk;   // global index of the first column to be updated
         // now we need to find the smallest my local row index, such that the corresponding global index is larger of equal to <curr_col_glob_res>
         Nb = curr_col_glob_res/nblk;    // how many global block-rows are before the needed one
         owner = Nb%np_rows;             // proc. row index of the owner of row with the global index equal to <curr_col_glob_res> (it is not necessarily me)
         curr_row_loc_res = (Nb/np_rows)*nblk; 
         if(my_prow < owner)
            curr_row_loc_res = curr_row_loc_res + nblk; 
      
         curr_row_loc_L = curr_row_loc_res;     // it is impossible, that both col_of_origin_L and row_of_origin_U are from upper part
         if(col_of_origin_L > my_prow)
            curr_row_loc_L = curr_row_loc_L - nblk;  
        
         rows_in_block = rows_in_buffer_L - curr_row_loc_L;    // rows in current block of L 
              
         curr_col_loc_U = i*nblk;   // local index in the buffer U of the current column
      
         if((curr_col_loc_U + nblk) <= cols_in_buffer_U)
            cols_in_block = nblk;      // number columns in block of U which will take part in this calculation
         else
            cols_in_block = cols_in_buffer_U - curr_col_loc_U; 
      
         if(rows_in_block_U > rows_in_buffer_U)
            rows_in_block_U = rows_in_buffer_U;     // rows in current column of U; also a leading dimension for U
 
         L_local_index = curr_row_loc_L; 
         L_local_start = &Buf_to_send_L[L_local_index]; 
         Res_ptr = &Res[curr_col_loc_res*na_rows + curr_row_loc_res];

         LDA_L = rows_in_buffer_L; 
         LDA_L_new = LDA_L; 
         if ((rows_in_block > 0)&&(cols_in_block > 0))
         {
            U_local_start_curr = U_local_start; 
            if (my_prow >= col_of_origin_L)
               index_row_L_for_LDA = 0;
            else
               index_row_L_for_LDA = 1;
            // loop over block-columns of the "active" part of L buffer
            for (ii = 0; ii < ceil((double)rows_in_block_U/(double)nblk); ii++)
            {
               if((ii+1)*nblk <= cols_in_buffer_L)
                  rows_in_block_U_curr = nblk; 
               else
                  rows_in_block_U_curr = cols_in_buffer_L - ii*nblk;  

               if((j == 1)&&(ii == 0))
                  dgemm_("N", "N", &rows_in_block, &cols_in_block, &rows_in_block_U_curr, &done, L_local_start, &LDA_L, U_local_start_curr, &rows_in_block_U, &dzero, Res_ptr, &na_rows); 
               else 
                  dgemm_("N", "N", &rows_in_block, &cols_in_block, &rows_in_block_U_curr, &done, L_local_start, &LDA_L, U_local_start_curr, &rows_in_block_U, &done, Res_ptr, &na_rows);

               if(np_rows*nblk*index_row_L_for_LDA + ((np_rows+my_prow)%np_rows)*nblk < np_cols*nblk*(ii + 1) + ((np_cols+col_of_origin_L)%np_cols)*nblk)
               {
                  LDA_L_new = LDA_L_new - nblk;
                  index_row_L_for_LDA = index_row_L_for_LDA + 1; 
               }
      
               U_local_start_curr = U_local_start_curr + rows_in_block_U_curr; 
               L_local_index = L_local_index - LDA_L + LDA_L*nblk + LDA_L_new; 
               L_local_start = &Buf_to_send_L[L_local_index];
               LDA_L = LDA_L_new; 
            }
         }
      
         U_local_start = U_local_start + rows_in_block_U*cols_in_block;
         curr_col_loc_res = curr_col_loc_res + nblk; 
         rows_in_block_U = rows_in_block_U + ratio*nblk;
      }    
      
      MPI_Wait(&request_L_Send, &status);
      MPI_Wait(&request_L_Recv, &status);
      MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_L); // find out how many elements I have received 
      
      MPI_Wait(&request_U_Send, &status);
      MPI_Wait(&request_U_Recv, &status);
      MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_U); // find out how many elements I have received  
   }
   
   /////// do the last multiplication //////////////
   rows_in_buffer_U = (int)Buf_to_receive_U[Size_receive_U-1];
   row_of_origin_U = (my_pcol + my_prow + np_cols + j - 1)%np_rows;     
   if(my_pcol >= row_of_origin_U)
      cols_in_buffer_U = na_cols;
   else
      cols_in_buffer_U = na_cols - nblk;
      
   cols_in_buffer_L = (int)Buf_to_receive_L[Size_receive_L-2];
   rows_in_buffer_L = (int)Buf_to_receive_L[Size_receive_L-1];
   // find the minimal pcol among those who have sent L for this iteration
   col_of_origin_L = np_cols; 
   for(i = 0; i < ratio; i++)
   {
      intNumber = (my_pcol + my_prow + i*np_rows + np_cols + np_rows - 1)%np_cols;
      if(intNumber < col_of_origin_L)
         col_of_origin_L = intNumber;
   }
   
   // find block-column of the result to start update with
   if (my_pcol >= row_of_origin_U)   // if origin of U is from the upper part 
      curr_col_loc_res = 0;          // then I update all columns of Result    
   else
      curr_col_loc_res = nblk;       // the first block column of U corresponds to my second one and I do not need to update the first block-column
      
   num_of_blocks_in_U_buffer = ceil((double)((double)cols_in_buffer_U/(double)nblk));
   if(my_pcol >= row_of_origin_U)    // if origin of U is from the upper part
      rows_in_block_U = ceil(((double)(my_pcol + 1) - (double)row_of_origin_U)/(double)np_rows)*nblk;  // blocks in the first block-column of U buffer
   else
      rows_in_block_U = ratio*nblk;
      
   U_local_start = &Buf_to_receive_U[0];
      
   for (i = 0; i < num_of_blocks_in_U_buffer; i++)
   { 
      // find block-row of the result to start update with; we need to update only lower triangular part of result
      curr_col_glob_res = np_cols*nblk*(curr_col_loc_res/nblk) + curr_col_loc_res%nblk + ((np_cols+my_pcol)%np_cols)*nblk;   // global index of the first column to be updated
      // now we need to find the smallest my local row index, such that the corresponding global index is larger of equal to <curr_col_glob_res>
      Nb = curr_col_glob_res/nblk;    // how many global block-rows are before the needed one
      owner = Nb%np_rows;             // proc. row index of the owner of row with the global index equal to <curr_col_glob_res> (it is not necessarily me)
      curr_row_loc_res = (Nb/np_rows)*nblk; 
      if(my_prow < owner)
         curr_row_loc_res = curr_row_loc_res + nblk; 
      
      curr_row_loc_L = curr_row_loc_res;     // it is impossible, that both col_of_origin_L and row_of_origin_U are from upper part
      if(col_of_origin_L > my_prow)
         curr_row_loc_L = curr_row_loc_L - nblk;
      
      rows_in_block = rows_in_buffer_L - curr_row_loc_L;    //rows in current block of  
              
      curr_col_loc_U = i*nblk;   // local index in the buffer U of the current column
      
      if((curr_col_loc_U + nblk) <= cols_in_buffer_U)
         cols_in_block = nblk;      // number columns in block of U which will take part in this calculation
      else
         cols_in_block = cols_in_buffer_U - curr_col_loc_U; 
      
      if(rows_in_block_U > rows_in_buffer_U)
         rows_in_block_U = rows_in_buffer_U; 
 
      L_local_index = curr_row_loc_L; 
      L_local_start = &Buf_to_receive_L[L_local_index]; 
      Res_ptr = &Res[curr_col_loc_res*na_rows + curr_row_loc_res];
      LDA_L = rows_in_buffer_L; 
      LDA_L_new = LDA_L; 
      if ((rows_in_block > 0) &&(cols_in_block > 0))
      {
         U_local_start_curr = U_local_start; 
         if (my_prow >= col_of_origin_L)
            index_row_L_for_LDA = 0;
         else
           index_row_L_for_LDA = 1; 
         // loop over block-columns of the "active" part of L buffer
         for (ii = 0; ii < ceil((double)rows_in_block_U/(double)nblk); ii++)
         {
            if((ii+1)*nblk <= cols_in_buffer_L)
               rows_in_block_U_curr = nblk; 
            else
               rows_in_block_U_curr = cols_in_buffer_L - ii*nblk;  

            if((j == 1)&&(ii == 0))
               dgemm_("N", "N", &rows_in_block, &cols_in_block, &rows_in_block_U_curr, &done, L_local_start, &LDA_L, U_local_start_curr, &rows_in_block_U, &dzero, Res_ptr, &na_rows); 
            else 
               dgemm_("N", "N", &rows_in_block, &cols_in_block, &rows_in_block_U_curr, &done, L_local_start, &LDA_L, U_local_start_curr, &rows_in_block_U, &done, Res_ptr, &na_rows);

            if(np_rows*nblk*index_row_L_for_LDA + ((np_rows+my_prow)%np_rows)*nblk < np_cols*nblk*(ii + 1) + ((np_cols+col_of_origin_L)%np_cols)*nblk)
            {
               LDA_L_new = LDA_L_new - nblk;
               index_row_L_for_LDA = index_row_L_for_LDA + 1; 
            }
              
            U_local_start_curr = U_local_start_curr + rows_in_block_U_curr; 
            L_local_index = L_local_index - (LDA_L - rows_in_block) + LDA_L*nblk + LDA_L_new - rows_in_block; 
            L_local_start = &Buf_to_receive_L[L_local_index];
            LDA_L = LDA_L_new; 
         }
      }
      
      U_local_start = U_local_start + rows_in_block_U*cols_in_block;
      curr_col_loc_res = curr_col_loc_res + nblk; 
      rows_in_block_U = rows_in_block_U + ratio*nblk;
   }
      
   free(Buf_to_send_L);
   free(Buf_to_receive_L);
   free(Buf_to_send_U);
   free(Buf_to_receive_U);
   if(ratio != 1)
      free(Buf_L); 
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////// Start of main program //////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
   int myid;
   int nprocs;
#ifndef WITH_MPI
   int MPI_COMM_WORLD;
#endif
   int my_mpi_comm_world, mpi_comm_rows, mpi_comm_cols;
   int na, nblk, np_cols, np_rows, np_colsStart, my_blacs_ctxt, nprow, npcol, my_prow, my_pcol;

   int mpierr;

   int info, i, j, na_rows, na_cols; 
   
   double startVal;

   double *a, *b,  *c, *a_copy, *b_copy, *c1, *c2, *a_t, *work;
   int *a_desc, *b_desc, *c_desc; 
      
   double value, diff, diffSum; 
   
   double done = 1.0; 
   double dMinusOne = -1.0; 
   int one = 1; 
   double dzero = 0.0; 
   int zero = 0;
   double startTime, endTime, localTime, avTime, maxTime;
   int reallevel;
   
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
   
   na = atoi(argv[1]);
   nblk = atoi(argv[2]);

   ///////////// procs grids and communicators ///////////////////////////////////////////////
   if (myid == 0)
     printf("Matrix size: %d, blocksize: %d\n\n", na, nblk);

   startVal = sqrt((double) nprocs);
   np_colsStart = (int) round(startVal);
   for (np_rows=np_colsStart;np_rows>1;np_rows--){
     if (nprocs %np_rows ==0)
     break;
     }
   if (nprocs == 3200)
      np_rows = 40;
   if (nprocs == 800)
      np_rows = 20; 
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
set_up_blacsgrid_f1(my_mpi_comm_world, &my_blacs_ctxt, &np_rows, &np_cols, &nprow, &npcol, &my_prow, &my_pcol);

   /* get the ELPA row and col communicators. */
   /* These are NOT usable in C without calling the MPI_Comm_f2c function on them !! */
#ifdef WITH_MPI
   my_mpi_comm_world = MPI_Comm_c2f(MPI_COMM_WORLD);
#endif
   mpierr = elpa_get_communicators(my_mpi_comm_world, my_prow, my_pcol, &mpi_comm_rows, &mpi_comm_cols);

   ////////////////////// descriptors area ///////////////////////////////////////////////
   a_desc = malloc(9*sizeof(int));
   b_desc = malloc(9*sizeof(int));
   c_desc = malloc(9*sizeof(int));
   
   na_rows = numroc_(&na, &nblk, &my_prow, &zero, &np_rows);
   na_cols = numroc_(&na, &nblk, &my_pcol, &zero, &np_cols);
   
   descinit_(a_desc, &na, &na, &nblk, &nblk, &zero, &zero, &my_blacs_ctxt, &na_rows, &info);
   descinit_(b_desc, &na, &na, &nblk, &nblk, &zero, &zero, &my_blacs_ctxt, &na_rows, &info);
   descinit_(c_desc, &na, &na, &nblk, &nblk, &zero, &zero, &my_blacs_ctxt, &na_rows, &info);
   
   /////////////////////////memory allocation area//////////////////////////////////////////////////////////////
   a  = malloc(na_rows*na_cols*sizeof(double));
   b  = malloc(na_rows*na_cols*sizeof(double));
   c = malloc(na_rows*na_cols*sizeof(double));
   a_copy  = malloc(na_rows*na_cols*sizeof(double));
   b_copy  = malloc(na_rows*na_cols*sizeof(double));
   c1 = malloc(na_rows*na_cols*sizeof(double));
   c2 = malloc(na_rows*na_cols*sizeof(double));
   a_t  = malloc(na_rows*na_cols*sizeof(double));
   work = malloc(na_cols*na_rows*sizeof(double));

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
   
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   for(i = 0; i < na_rows*na_cols; i++)
     c[i] = 0;
   for(i = 0; i < na_rows*na_cols; i++)
     c2[i] = 0;
   
   elpa_cholesky_real_double(na, b, na_rows, nblk, na_cols, mpi_comm_rows, mpi_comm_cols, 1);   // now b = U
   for(i = 0; i < na_rows; i++)
      for(j = 0; j < na_cols; j++)
      {
         i_global = np_rows*nblk*(i/nblk) + i%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1; 
         j_global = np_cols*nblk*(j/nblk) + j%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
         if(i_global > j_global)
            b[i + j*na_rows] = 0;
         if(i_global < j_global)
            a[i + j*na_rows] = 0;
      }
     
   //make copies of a and b
   for(i = 0; i < na_rows*na_cols; i++)
   {
      a_copy[i] = a[i];
      b_copy[i] = b[i];
   }      
   
   for(i = 0; i < na_rows*na_cols; i++)
      c1[i] = a_copy[i];
   
   if(myid == 0)
      printf("\n\nTest1 ___________________________________________________________________ \n");
   ///// test Cannon's ///////////////////////////////////////////////////////////////////////////////
   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
   d_Cannons_Mult2(a, b, np_rows, np_cols, my_prow, my_pcol, a_desc, c, mpi_comm_cols, mpi_comm_rows);
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   
   if(myid == 0)
      printf("\n Cannon's algorithm. 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, avTime, maxTime);
   
   ///// test PDTRMM /////////////////////////////////////////////////////////////////////////////////////
   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
   pdtrmm_("R", "U", "N", "N", &na, &na, &done, b_copy, &one, &one, b_desc, c1, &one, &one, c_desc);
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("\n PDTRMM from ScaLAPACK. 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, avTime, maxTime);
   
   ///// test ELPA ///////////////////////////////////////////////////////////////////////////////////////////
   pdtran_(&na, &na, &done, a_copy, &one, &one, a_desc, &dzero, a_t, &one, &one, a_desc);
   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
   elpa_mult_at_b_real_double('U', 'U', na, na, b, na_rows, na_cols, a_t, na_rows, na_cols, nblk, mpi_comm_rows, mpi_comm_cols, work, na_rows, na_cols);   // work has upper part of b(H)*A(H)
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("\n elpa_mult_at_b_real_double(U,U). 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, avTime, maxTime);
   
   pdtran_(&na, &na, &done, work, &one, &one, a_desc, &dzero, c2, &one, &one, a_desc);   // c2 has lower part of A*b
      
   for(i = 0; i < na_rows; i++)
      for(j = 0; j < na_cols; j++)
      {
         i_global = np_rows*nblk*(i/nblk) + i%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1; 
         j_global = np_cols*nblk*(j/nblk) + j%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
         if(i_global < j_global)
         {
            c[i + j*na_rows] = 0; 
            c1[i + j*na_rows] = 0; 
            c2[i + j*na_rows] = 0;
         }        
      }
   
   /////check /////////////////////////////////////////////////////////////////////////////////////////////////
   diff = 0;
   diffSum = 0; 
   for(i = 0; i < na_rows*na_cols; i++)
      diff = diff + fabs(c1[i]-c[i]);
      
   MPI_Reduce(&diff, &diffSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between Cannon's and PDTRMM = %.15e, average = %.15e\n", diffSum, diffSum/(na*na));
   
   diff = 0;
   diffSum = 0; 
   for(i = 0; i < na_rows*na_cols; i++)
      diff = diff + fabs(c2[i]-c[i]);
   MPI_Reduce(&diff, &diffSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between Cannon's and ELPA = %.15e, average = %.15e\n", diffSum, diffSum/(na*na));
   
   diff = 0;
   diffSum = 0; 
   for(i = 0; i < na_rows*na_cols; i++)
      diff = diff + fabs(c2[i]-c1[i]);
   MPI_Reduce(&diff, &diffSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between ScaLAPACK and ELPA = %.15e, average = %.15e\n", diffSum, diffSum/(na*na));
   
   if(myid == 0)
      printf("\n\nTest2 ___________________________________________________________________ \n");
   
   for(i = 0; i < na_rows*na_cols; i++)
      c[i] = 0;
   for(i = 0; i < na_rows*na_cols; i++)
      c2[i] = 0;
   for(i = 0; i < na_rows*na_cols; i++)
      c1[i] = a_copy[i];
   
   ///// test PDTRMM /////////////////////////////////////////////////////////////////////////////////////
   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
   pdtrmm_("R", "U", "N", "N", &na, &na, &done, b_copy, &one, &one, b_desc, c1, &one, &one, c_desc);
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("\n PDTRMM from ScaLAPACK. 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, avTime, maxTime);
   
   ///// test Cannon's ///////////////////////////////////////////////////////////////////////////////
   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
   d_Cannons_Mult2(a, b, np_rows, np_cols, my_prow, my_pcol, a_desc, c, mpi_comm_cols, mpi_comm_rows);
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   
   if(myid == 0)
      printf("\n Cannon's algorithm. 0 proc in: %lf, average over procs = %lf, max = %lf\n\n ", localTime, avTime, maxTime);
   
   ///// test ELPA ///////////////////////////////////////////////////////////////////////////////////////////
   pdtran_(&na, &na, &done, a_copy, &one, &one, a_desc, &dzero, a_t, &one, &one, a_desc);
   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
   elpa_mult_at_b_real_double('U', 'U', na, na, b, na_rows, na_cols, a_t, na_rows, na_cols, nblk, mpi_comm_rows, mpi_comm_cols, work, na_rows, na_cols);
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("\n elpa_mult_at_b_real_double(U,U). 0 proc in: %lf, average over procs = %lf, max = %lf\n\n ", localTime, avTime, maxTime);
   
   pdtran_(&na, &na, &done, work, &one, &one, a_desc, &dzero, c2, &one, &one, a_desc);
      
   for(i = 0; i < na_rows; i++)
      for(j = 0; j < na_cols; j++)
      {
         i_global = np_rows*nblk*(i/nblk) + i%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1; 
         j_global = np_cols*nblk*(j/nblk) + j%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
         if(i_global < j_global)
         {
            c[i + j*na_rows] = 0; 
            c1[i + j*na_rows] = 0; 
            c2[i + j*na_rows] = 0;
         }        
      }
   
   /////check /////////////////////////////////////////////////////////////////////////////////////////////////
   diff = 0;
   diffSum = 0; 
   for(i = 0; i < na_rows*na_cols; i++)
      diff = diff + fabs(c1[i]-c[i]);
   MPI_Reduce(&diff, &diffSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between Cannon's and PDTRMM = %.15e, average = %.15e\n", diffSum, diffSum/(na*na));
   
   diff = 0;
   diffSum = 0; 
   for(i = 0; i < na_rows*na_cols; i++)
      diff = diff + fabs(c2[i]-c[i]);
   MPI_Reduce(&diff, &diffSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between Cannon's and ELPA = %.15e, average = %.15e\n", diffSum, diffSum/(na*na));
   
   diff = 0;
   diffSum = 0; 
   for(i = 0; i < na_rows*na_cols; i++)
      diff = diff + fabs(c2[i]-c1[i]);
   MPI_Reduce(&diff, &diffSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between ScaLAPACK and ELPA = %.15e, average = %.15e\n", diffSum, diffSum/(na*na));
   
   if(myid == 0)
      printf("\n\nTest3 ___________________________________________________________________ \n");
   
   for(i = 0; i < na_rows*na_cols; i++)
      c[i] = 0; 
   for(i = 0; i < na_rows*na_cols; i++)
      c2[i] = 0;
   for(i = 0; i < na_rows*na_cols; i++)
      c1[i] = a_copy[i];
   
   ///// test PDTRMM /////////////////////////////////////////////////////////////////////////////////////
   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
   pdtrmm_("R", "U", "N", "N", &na, &na, &done, b_copy, &one, &one, b_desc, c1, &one, &one, c_desc);
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("\n PDTRMM from ScaLAPACK. 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, avTime, maxTime);

   ///// test ELPA ///////////////////////////////////////////////////////////////////////////////////////////
   pdtran_(&na, &na, &done, a_copy, &one, &one, a_desc, &dzero, a_t, &one, &one, a_desc);
   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
   elpa_mult_at_b_real_double('U', 'U', na, na, b, na_rows, na_cols, a_t, na_rows, na_cols, nblk, mpi_comm_rows, mpi_comm_cols, work, na_rows, na_cols);
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("\n elpa_mult_at_b_real_double(U,U). 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, avTime, maxTime);
   
   ///// test Cannon's ///////////////////////////////////////////////////////////////////////////////
   for(i = 0; i < na_rows*na_cols; i++)
     c[i] = 0;
   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
   d_Cannons_Mult2(a, b, np_rows, np_cols, my_prow, my_pcol, a_desc, c, mpi_comm_cols, mpi_comm_rows);
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   
   if(myid == 0)
      printf("\n Cannon's algorithm. 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, avTime, maxTime);

   pdtran_(&na, &na, &done, work, &one, &one, a_desc, &dzero, c2, &one, &one, a_desc);
     
   for(i = 0; i < na_rows; i++)
      for(j = 0; j < na_cols; j++)
      {
         i_global = np_rows*nblk*(i/nblk) + i%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1; 
         j_global = np_cols*nblk*(j/nblk) + j%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
         if(i_global < j_global)
         {
            c[i + j*na_rows] = 0; 
            c1[i + j*na_rows] = 0; 
            c2[i + j*na_rows] = 0;
         }        
      }
   
   /////check /////////////////////////////////////////////////////////////////////////////////////////////////
   diff = 0;
   diffSum = 0; 
   for(i = 0; i < na_rows*na_cols; i++)
      diff = diff + fabs(c1[i]-c[i]);
   MPI_Reduce(&diff, &diffSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between Cannon's and PDTRMM = %.15e, average = %.15e\n", diffSum, diffSum/(na*na));
   
   diff = 0;
   diffSum = 0; 
   for(i = 0; i < na_rows*na_cols; i++)
      diff = diff + fabs(c2[i]-c[i]);
   MPI_Reduce(&diff, &diffSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between Cannon's and ELPA = %.15e, average = %.15e\n", diffSum, diffSum/(na*na));
   
   diff = 0;
   diffSum = 0; 
   for(i = 0; i < na_rows*na_cols; i++)
      diff = diff + fabs(c2[i]-c1[i]);
   MPI_Reduce(&diff, &diffSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between ScaLAPACK and ELPA = %.15e, average = %.15e\n", diffSum, diffSum/(na*na)); 

////////////////////////////////////////////////////////////////////////////////////// free memory ///////////////////////////////////////////////////
   free(a);
   free(a_desc);
   free(b);
   free(b_desc);
   free(c); 
   free(c_desc);
   free(work);
   free(a_copy);
   free(b_copy);
   free(c1);
   free(c2);
   free(a_t);
 
#ifdef WITH_MPI
   MPI_Finalize();
#endif
   return 0;
}
