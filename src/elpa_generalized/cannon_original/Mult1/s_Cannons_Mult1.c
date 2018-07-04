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

void slacpy_(char*, int*, int*, float*, int*, float*, int*);
void sgemm_(char*, char*, int*, int*, int*, float*, float*, int*, float*, int*, float*, float*, int*); 
void pstran_(int*, int*, float*, float*, int*, int*, int*, float*, float*, int*, int*, int*);
void pstrmm_(char*, char*, char*, char*, int*, int*, float*, float*, int*, int*, int*, float*, int*, int*, int*);
void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
int numroc_(int*, int*, int*, int*, int*);
void set_up_blacsgrid_f1(int, int*, int*, int*, int*, int*, int*, int*);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////// My function for multiplication 1 //////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void s_Cannons_Mult1(float* A, float* U, int np_rows, int np_cols, int my_prow, int my_pcol, int* a_desc, float *Res, MPI_Comm row_comm, MPI_Comm col_comm)
{
   // Input: 
   //    A is square matrix
   //    U is upper triangular
   //  Output:
   //     Res is an upper triangular part of A*B
   // row_comm: communicator along rows
   // col_comm: communicator along columns
  
   int na, nblk, i, j, Size_send_A, Size_receive_A, Size_send_U, Size_receive_U, Buf_rows, Buf_cols, where_to_send_A, from_where_to_receive_A, where_to_send_U, from_where_to_receive_U, last_proc_row, last_proc_col, Size_U_stored;
   float *Buf_to_send_A, *Buf_to_receive_A, *Buf_to_send_U, *Buf_to_receive_U, *float_ptr, *Buf_A, *Buf_pos, *U_local_start, *Res_ptr;
   int ratio, num_of_iters, cols_in_buffer, rows_in_block, rows_in_block_A, rows_in_buffer, curr_col_loc, cols_in_block, curr_col_glob;
   int rows_in_block_U, num_of_blocks_in_U_buffer, startPos, curr_col_loc_res, curr_col_loc_buf, proc_row_curr, Size_receive_A_now, intNumber, width, row_origin_U;
   float *CopyTo, *CopyFrom;
   float done = 1.0;
   float dzero = 0.0;
   int zero = 0; 
   int na_rows, na_cols;
        
   MPI_Status status;
   MPI_Request request_A_Recv; 
   MPI_Request request_A_Send;
   MPI_Request request_U_Recv; 
   MPI_Request request_U_Send;
      
   na = a_desc[2];
   nblk = a_desc[4];
   na_rows = numroc_(&na, &nblk, &my_prow, &zero, &np_rows);
   na_cols = numroc_(&na, &nblk, &my_pcol, &zero, &np_cols); 
   
//////////////////////////////////////////// Start of algorithm //////////////////////////////////////////////////////////////////////////////
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
   last_proc_row = ((na-1)/nblk) % np_rows;          // processor row having the last block-row of matrix
   last_proc_col = ((na-1)/nblk) % np_cols;          // processor column having the last block-column of matrix
   
   /////////////////////////memory allocation area//////////////////////////////////////////////////////////////
   if(na%nblk == 0)
      if(my_pcol <= last_proc_col)
         Buf_cols = na_cols;
      else
         Buf_cols = na_cols + nblk;      
   else
      if(my_pcol < last_proc_col)
         Buf_cols = na_cols;
      else if(my_pcol > last_proc_col)
         Buf_cols = na_cols + nblk; 
      else  // if my_pcol == last_proc_col
         Buf_cols = na_cols + nblk - na_cols%nblk;     
   
  if(na%nblk == 0)
      if(my_prow <= last_proc_row)
         Buf_rows = na_rows;
      else
         Buf_rows = na_rows + nblk;      
   else
      if(my_prow < last_proc_row)
         Buf_rows = na_rows;
      else if(my_prow > last_proc_row)
         Buf_rows = na_rows + nblk; 
      else  // if my_prow == last_proc_row
         Buf_rows = na_rows + nblk - na_rows%nblk;  
      
   intNumber = ceil((float)na/(float)(np_cols*nblk));   // max. possible number of the local block columns of U
   Size_U_stored = ratio*nblk*nblk*intNumber*(intNumber+1)/2 + 1;   // number of local elements from the upper triangular part that every proc. has (max. possible value among all the procs.)
        
   Buf_to_send_A = malloc(ratio*Buf_cols*Buf_rows*sizeof(float));
   Buf_to_receive_A = malloc(ratio*Buf_cols*Buf_rows*sizeof(float));
   Buf_to_send_U = malloc(Size_U_stored*sizeof(float));
   Buf_to_receive_U = malloc(Size_U_stored*sizeof(float));
   if(ratio != 1)
      Buf_A = malloc(Buf_cols*Buf_rows*sizeof(float));   // in this case we will receive data into initial buffer and after place block-columns to the needed positions of buffer for calculation
        
   ////////////////////////////////////////////////////////////// initial reordering of A //////////////////////////////////////////////////////////////////////////////
   // here we assume, that np_rows < np_cols; then I will send to the number of processors equal to <ratio> with the "leap" equal to np_rows; the same holds for receive  
   if(ratio != 1)
      slacpy_("A", &na_rows, &na_cols, A, &na_rows, Buf_to_send_A, &na_rows);   // copy my buffer to send
   Size_receive_A = 0; 
   
   // receive from different processors and place in my buffer for calculation;
   for(i = 0; i < ratio; i++)
   {
      where_to_send_A = (my_pcol - my_prow - i*np_rows + np_cols)%np_cols;  // here we have "+ np_cols" not to get negative values          
      from_where_to_receive_A = (my_pcol + my_prow + i*np_rows)%np_cols;
      
      // send and receive in the row_comm
      if(ratio != 1)   // if grid is not square
      {
         if(where_to_send_A != my_pcol)
         {
            MPI_Sendrecv(Buf_to_send_A, na_cols*na_rows, MPI_FLOAT, where_to_send_A, 0, Buf_A, na_rows*Buf_cols, MPI_FLOAT, from_where_to_receive_A, 0, row_comm, &status);
            MPI_Get_count(&status, MPI_FLOAT, &Size_receive_A_now);
            Size_receive_A_now = Size_receive_A_now/na_rows;       // how many columns of A I have received
         }
         else
            Size_receive_A_now = na_cols;
         Size_receive_A = Size_receive_A + Size_receive_A_now;  // here accumulate number of columns of A that I will receive

         // now I need to copy the received block to my buffer for A
         intNumber = from_where_to_receive_A/np_rows; // how many blocks I will receive, such that I will need to put them before the just received block         
         
         CopyTo = &Buf_to_receive_A[intNumber*na_rows*nblk];  // here I will start copying the received buffer
         if(where_to_send_A != my_pcol)
            CopyFrom = Buf_A; 
         else
            CopyFrom = A;
         
         intNumber = ceil((float)Size_receive_A_now/(float)nblk);   // how many block-columns I have received
         for(j = 0; j < intNumber; j++)
         {
            width = nblk; // width of the current block column
            if(nblk*(j+1) > Size_receive_A_now)
               width = Size_receive_A_now - nblk*j; 
            slacpy_("A", &na_rows, &width, CopyFrom, &na_rows, CopyTo, &na_rows);
            CopyTo = CopyTo + na_rows*nblk*ratio; 
            CopyFrom = CopyFrom + na_rows*nblk; 
         }
      }
      else  // if grid is square then simply receive from one processor to a calculation buffer
         if(my_prow > 0)
         {
            slacpy_("A", &na_rows, &na_cols, A, &na_rows, Buf_to_send_A, &na_rows);   // copy my buffer to send
            MPI_Sendrecv(Buf_to_send_A, na_cols*na_rows, MPI_FLOAT, where_to_send_A, 0, Buf_to_receive_A, na_rows*Buf_cols, MPI_FLOAT, from_where_to_receive_A, 0, row_comm, &status);
            MPI_Get_count(&status, MPI_FLOAT, &Size_receive_A);
            Size_receive_A = Size_receive_A/na_rows;       // how many columns of A I have received
         }
         else
         {
            slacpy_("A", &na_rows, &na_cols, A, &na_rows, Buf_to_receive_A, &na_rows);   // copy A to the received buffer if I do not need to send
            Size_receive_A = na_cols; 
         }
   }
      
   ////////////////////////////////////////////////////////////// initial reordering of U //////////////////////////////////////////////////////
   // form array to send by block-columns
   num_of_iters = ceil((float)na_cols/(float)nblk);       // number my of block-columns
   
   where_to_send_U = (my_prow - my_pcol + np_cols)%np_rows;                 // shift = my_pcol; we assume that np_cols%np_rows = 0
   from_where_to_receive_U = (my_pcol + my_prow)%np_rows;
   
   if(where_to_send_U == my_prow)    // if I will not need to send my local part of U, then copy my local data to the "received" buffer
      Buf_pos = Buf_to_receive_U;
   else
      Buf_pos = Buf_to_send_U;         // else form the array to send
   
   // find the first local block belonging to the upper part of matrix U
   if(my_pcol >= my_prow)  // if I am in the upper part of proc. grid
      curr_col_loc = 0;    // my first local block-column has block from the upper part of matrix
   else
      curr_col_loc = 1;   //ceil((float)(((float)my_prow - (float)my_pcol)/(float)np_cols)) always will give 1 since np_cols > np_rows 
      
   num_of_iters = num_of_iters - curr_col_loc;   // I will exclude the first <curr_col_loc> block-columns since they do not have blocks from the upper part of matrix U
   curr_col_loc = curr_col_loc*nblk;             // local index of the found block-column

   if(my_pcol >= my_prow )
      rows_in_block = ceil(((float)(my_pcol + 1) - (float)my_prow)/(float)np_rows)*nblk;  // number of rows in first block-column of U, s.t. this block-column has at least one block-row in the upper part
   else
      rows_in_block = ratio*nblk;
   
   Size_send_U = 0; 
   for(i = 0; i < num_of_iters; i++)       // loop over my block-columns, which have blocks in the upepr part of U
   {      
      if(rows_in_block > na_rows)   // this is number of rows in the upper part of current block-column
         rows_in_block = na_rows; 

      if ((na_cols - curr_col_loc) < nblk)
         cols_in_block = na_cols - curr_col_loc;     // how many columns do I have in the current block-column
      else
         cols_in_block = nblk; 
      
      if((rows_in_block > 0)&&(cols_in_block > 0))
      {
         float_ptr = &U[curr_col_loc*na_rows];   // pointer to start of the current block-column to be copied to buffer
         slacpy_("A", &rows_in_block, &cols_in_block, float_ptr, &na_rows, Buf_pos, &rows_in_block);     // copy upper part of block-column in the buffer with LDA = length of the upper part of block-column 
         Buf_pos = Buf_pos + rows_in_block*cols_in_block;                         // go to the position where the next block-column will be copied                                             
         Size_send_U = Size_send_U + rows_in_block*cols_in_block; 
      }
      curr_col_loc = curr_col_loc + nblk;      // go to the next local block-column of my local array U 
      rows_in_block = rows_in_block + ratio*nblk;  // update number of rows in the upper part of current block-column
   }
   rows_in_buffer = rows_in_block - ratio*nblk;    // remove redundant addition from the previous loop 
   *Buf_pos = (float)rows_in_buffer; // write number of the rows at the end of the buffer; we will need this for further multiplications on the other processors
   Size_send_U = Size_send_U + 1;
   
   //send and receive
   if(where_to_send_U != my_prow)
   {   
      // send and receive in the col_comm
      MPI_Sendrecv(Buf_to_send_U, Size_send_U, MPI_FLOAT, where_to_send_U, 0, Buf_to_receive_U, Buf_rows*na_cols, MPI_FLOAT, from_where_to_receive_U, 0, col_comm, &status); 
      MPI_Get_count(&status, MPI_FLOAT, &Size_receive_U); // find out how many elements I have received 
   }
   else // if I do not need to send 
      Size_receive_U = Size_send_U;         // how many elements I "have received"; the needed data I have already copied to the "receive" buffer

   //////////////////////////////////////////////////////////////////////// main loop /////////////////////////////////////////////////////
   where_to_send_A = (my_pcol - 1 + np_cols)%np_cols;
   from_where_to_receive_A = (my_pcol + 1)%np_cols;
   where_to_send_U = (my_prow - 1 + np_rows)%np_rows;
   from_where_to_receive_U = (my_prow + 1)%np_rows;
   
   for(j = 1; j < np_rows; j++)
   {
      // at this moment I need to send to neighbour what I have in the "received" arrays; that is why exchange pointers of the "received" and "send" arrays
      float_ptr = Buf_to_send_A; 
      Buf_to_send_A = Buf_to_receive_A; 
      Buf_to_receive_A = float_ptr; 
      
      float_ptr = Buf_to_send_U; 
      Buf_to_send_U = Buf_to_receive_U; 
      Buf_to_receive_U = float_ptr;
      
      ///// shift for A ////////////////////////////////////////////////////////////
      Size_send_A = Size_receive_A;  // number of block-columns of A and block-rows of U to send (that I have received on the previous step) 
      MPI_Isend(Buf_to_send_A, Size_send_A*na_rows, MPI_FLOAT, where_to_send_A, 0, row_comm, &request_A_Send); 
      MPI_Irecv(Buf_to_receive_A, Buf_cols*na_rows*ratio, MPI_FLOAT, from_where_to_receive_A, 0, row_comm, &request_A_Recv);
         
      ///// shift for U /////////////////////////////////////////////
      Size_send_U = Size_receive_U; 
      MPI_Isend(Buf_to_send_U, Size_send_U, MPI_FLOAT, where_to_send_U, 0, col_comm, &request_U_Send); 
      MPI_Irecv(Buf_to_receive_U, Buf_rows*na_cols, MPI_FLOAT, from_where_to_receive_U, 0, col_comm, &request_U_Recv); 
      
      ///// multiplication ////////////////////////////////////////////////////////////////////////////////////////////
      rows_in_buffer = (int)Buf_to_send_U[Size_receive_U-1];
      row_origin_U = (my_pcol + my_prow + np_cols + j - 1)%np_rows;
      
      if((my_pcol >= my_prow)&&(my_pcol >= row_origin_U))   // if I and sender are from the upper part of grid
      {
         cols_in_buffer = na_cols;                          // then we have the same number of columns in the upper triangular part
         curr_col_loc_res = 0;                              // all my block-columns have parts in the upper triangular part
         curr_col_loc_buf = 0;                              // I use all the block-columns of the received buffer
      }
      if((my_pcol < my_prow)&&(my_pcol < row_origin_U))     // if I and sender are from the lower part of grid
      {
         cols_in_buffer = na_cols - nblk;                   // then we have the same number of columns in the upper triangular part, but the first block-column was not included
         curr_col_loc_res = nblk;                           // I start update from the second block-column since the first on is in the lower triangular part
         curr_col_loc_buf = 0;                              // I use all the block-columns of the received buffer
      }
      if((my_pcol >= my_prow)&&(my_pcol < row_origin_U))    // if I am from the upper part of grid and sender is from the lower part
      {
         cols_in_buffer = na_cols - nblk;                   // then I have received one block-column less than I have
         curr_col_loc_res = nblk;                           // all my block-columns have parts in the upper triangular part, but the first block-column of the received buffers corresponds to my second one
         curr_col_loc_buf = 0;                              // I use all the block-columns of the received buffer
      }
      if((my_pcol < my_prow)&&(my_pcol >= row_origin_U))    // if I am from the lower part of grid and sender is from the upper part
      {
         cols_in_buffer = na_cols;                          // then I have received the full set of block-columns
         curr_col_loc_res = nblk;                           // I start update from the second block-column since the first on is in the lower triangular part
         curr_col_loc_buf = nblk;                           // I skip the first block-column of the buffer, since my first block-column is in the lower part
      }
    
      num_of_blocks_in_U_buffer = ceil(((float)cols_in_buffer - (float)curr_col_loc_buf)/(float)nblk); 
      
      startPos = (curr_col_loc_buf + nblk)*curr_col_loc_buf/2;
      U_local_start = &Buf_to_send_U[startPos];
      Res_ptr = &Res[curr_col_loc_res*na_rows];
  
      for (i = 0; i < num_of_blocks_in_U_buffer; i++)
      { 
         curr_col_glob = (curr_col_loc_res/nblk)*nblk*np_cols + my_pcol*nblk;
         proc_row_curr = (curr_col_glob/nblk)%np_rows; 
         rows_in_block_A = (curr_col_glob/(nblk*np_rows))*nblk;     // in A; not to go down beyond  the upper triangular part
         if(my_prow <= proc_row_curr)
            rows_in_block_A = rows_in_block_A + nblk; 
         
         if(rows_in_block_A > na_rows)
            rows_in_block_A = na_rows; 
      
         if((curr_col_loc_buf + nblk) <= cols_in_buffer)
            cols_in_block = nblk;      // number columns in block of U which will take part in this calculation
         else
            cols_in_block = cols_in_buffer - curr_col_loc_buf; 
      
         rows_in_block_U = (curr_col_glob/(nblk*np_rows))*nblk;    // corresponds to columns in A;
         if(proc_row_curr >= row_origin_U)
            rows_in_block_U = rows_in_block_U + nblk; 
         
         if(rows_in_block_U > rows_in_buffer)
            rows_in_block_U = rows_in_buffer;

         if ((rows_in_block_A > 0)&&(cols_in_block > 0))
            if (j == 1)
               sgemm_("N", "N", &rows_in_block_A, &cols_in_block, &rows_in_block_U, &done, Buf_to_send_A, &na_rows, U_local_start, &rows_in_block_U, &dzero, Res_ptr, &na_rows);
            else 
               sgemm_("N", "N", &rows_in_block_A, &cols_in_block, &rows_in_block_U, &done, Buf_to_send_A, &na_rows, U_local_start, &rows_in_block_U, &done, Res_ptr, &na_rows);
      
         U_local_start = U_local_start + rows_in_block_U*cols_in_block;
         curr_col_loc_res = curr_col_loc_res + nblk;
         Res_ptr = &Res[curr_col_loc_res*na_rows];
         curr_col_loc_buf = curr_col_loc_buf + nblk;  
      } 
     
      MPI_Wait(&request_A_Send, &status);
      MPI_Wait(&request_A_Recv, &status);
      MPI_Get_count(&status, MPI_FLOAT, &Size_receive_A); // find out how many elements I have received 
      Size_receive_A = Size_receive_A/na_rows;
      
      MPI_Wait(&request_U_Send, &status);
      MPI_Wait(&request_U_Recv, &status);
      MPI_Get_count(&status, MPI_FLOAT, &Size_receive_U); // find out how many elements I have received  
   }
   
   /////// do the last multiplication //////////////
   rows_in_buffer = (int)Buf_to_receive_U[Size_receive_U-1];
   row_origin_U = (my_pcol + my_prow + np_cols + np_rows -1)%np_rows;

   if((my_pcol >= my_prow)&&(my_pcol >= row_origin_U))   // if I and sender are from the upper part of grid
   {
      cols_in_buffer = na_cols;                          // then we have the same number of columns in the upper triangular part
      curr_col_loc_res = 0;                              // all my block-columns have parts in the upper triangular part
      curr_col_loc_buf = 0;                              // I use all the block-columns of the received buffer
   }
   if((my_pcol < my_prow)&&(my_pcol < row_origin_U))     // if I and sender are from the lower part of grid
   {
      cols_in_buffer = na_cols - nblk;                   // then we have the same number of columns in the upper triangular part, but the first block-column was not included
      curr_col_loc_res = nblk;                           // I start update from the second block-column since the first on is in the lower triangular part
      curr_col_loc_buf = 0;                              // I use all the block-columns of the received buffer
   }
   if((my_pcol >= my_prow)&&(my_pcol < row_origin_U))    // if I am from the upper part of grid and sender is from the lower part
   {
      cols_in_buffer = na_cols - nblk;                   // then I have received one block-column less than I have
      curr_col_loc_res = nblk;                           // all my block-columns have parts in the upper triangular part, but the first block-column of the received buffers corresponds to my second one
      curr_col_loc_buf = 0;                              // I use all the block-columns of the received buffer
   }
   if((my_pcol < my_prow)&&(my_pcol >= row_origin_U))    // if I am from the lower part of grid and sender is from the upper part
   {
      cols_in_buffer = na_cols;                          // then I have received the full set of block-columns
      curr_col_loc_res = nblk;                           // I start update from the second block-column since the first on is in the lower triangular part
      curr_col_loc_buf = nblk;                           // I skip the first block-column of the buffer, since my first block-column is in the lower part
   }
    
   num_of_blocks_in_U_buffer = ceil(((float)cols_in_buffer - (float)curr_col_loc_buf)/(float)nblk); 
      
   startPos = (curr_col_loc_buf + nblk)*curr_col_loc_buf/2;
   U_local_start = &Buf_to_receive_U[startPos];
   Res_ptr = &Res[curr_col_loc_res*na_rows];
  
   for (i = 0; i < num_of_blocks_in_U_buffer; i++)
   { 
      curr_col_glob = (curr_col_loc_res/nblk)*nblk*np_cols + my_pcol*nblk;
      proc_row_curr = (curr_col_glob/nblk)%np_rows; 
      rows_in_block_A = (curr_col_glob/(nblk*np_rows))*nblk;     // in A; not to go down beyond  the upper triangular part
      if(my_prow <= proc_row_curr)
         rows_in_block_A = rows_in_block_A + nblk; 
         
      if(rows_in_block_A > na_rows)
         rows_in_block_A = na_rows; 
      
      if((curr_col_loc_buf + nblk) <= cols_in_buffer)
         cols_in_block = nblk;      // number columns in block of U which will take part in this calculation
      else
         cols_in_block = cols_in_buffer - curr_col_loc_buf; 
      
      rows_in_block_U = (curr_col_glob/(nblk*np_rows))*nblk;    // corresponds to columns in A;
      if(proc_row_curr >= row_origin_U)
         rows_in_block_U = rows_in_block_U + nblk; 
        
      if(rows_in_block_U > rows_in_buffer)
         rows_in_block_U = rows_in_buffer; 

      if ((rows_in_block_A > 0)&&(cols_in_block > 0))
         if (j == 1)
            sgemm_("N", "N", &rows_in_block_A, &cols_in_block, &rows_in_block_U, &done, Buf_to_receive_A, &na_rows, U_local_start, &rows_in_block_U, &dzero, Res_ptr, &na_rows);
         else 
            sgemm_("N", "N", &rows_in_block_A, &cols_in_block, &rows_in_block_U, &done, Buf_to_receive_A, &na_rows, U_local_start, &rows_in_block_U, &done, Res_ptr, &na_rows);
      
      U_local_start = U_local_start + rows_in_block_U*cols_in_block;
      curr_col_loc_res = curr_col_loc_res + nblk;
      Res_ptr = &Res[curr_col_loc_res*na_rows];
      curr_col_loc_buf = curr_col_loc_buf + nblk;  
   }  
 
   free(Buf_to_send_A);
   free(Buf_to_receive_A);
   free(Buf_to_send_U);
   free(Buf_to_receive_U);
   if(ratio != 1)
      free(Buf_A);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////// Start of main program //////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
   
   float startVal, diff, diffSum;

   float *a, *b,  *c, *a_copy, *b_copy, *c1, *c2, *a_t, *work;
   int *a_desc, *b_desc, *c_desc; 
      
   float value; 
   
   float done = 1.0; 
   float dMinusOne = -1.0; 
   int one = 1; 
   float dzero = 0.0; 
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
   if (myid == 0) {
     printf("Matrix size: %d, blocksize: %d\n", na, nblk);
     printf("\n");
   }

   startVal = sqrt((float) nprocs);
   np_colsStart = (int) round(startVal);
   for (np_rows=np_colsStart;np_rows>1;np_rows--){
     if (nprocs %np_rows ==0)
     break;
     }
   np_cols = nprocs/np_rows;
   if (myid == 0) {
     printf("Number of processor rows %d, cols %d, total %d \n",np_rows,np_cols,nprocs);
     printf("\n");
   }
   
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
   
   /////////////////////////memory allocation area/////////////////////////////////////////////////////////////
   c = malloc(na_rows*na_cols*sizeof(float));
   b  = malloc(na_rows*na_cols*sizeof(float));
   a_copy  = malloc(na_rows*na_cols*sizeof(float));
   b_copy  = malloc(na_rows*na_cols*sizeof(float));
   c1 = malloc(na_rows*na_cols*sizeof(float));
   c2 = malloc(na_rows*na_cols*sizeof(float));
   a_t  = malloc(na_rows*na_cols*sizeof(float));
   work = malloc(na_cols*na_rows*sizeof(float));
   a  = malloc(na_rows*na_cols*sizeof(float));

   //////////////////////////generate matrices//////////////////////////////////////////////////////////////////////////////
   int i_global, j_global;
   for(i = 0; i < na_rows; i++)
      for(j = 0; j < na_cols; j++)
      {
         i_global = np_rows*nblk*(i/nblk) + i%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1; 
         j_global = np_cols*nblk*(j/nblk) + j%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
         a[i + j*na_rows] = (float)cos(i_global)*cos(j_global) + (float)sin(i_global)*sin(j_global);
         if(i_global == j_global)
            a[i + j*na_rows] = a[i + j*na_rows] + (float)(i_global + j_global)/na;
         b[i + j*na_rows] = (float)sin(i_global)*(float)sin(j_global);   
         if(i_global == j_global)
            b[i + j*na_rows] = b[i + j*na_rows] + 1;  
      }
   
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   for(i = 0; i < na_rows*na_cols; i++)
       c[i] = 0;
   for(i = 0; i < na_rows*na_cols; i++)
       c2[i] = 0;
   
   elpa_cholesky_real_single(na, b, na_rows, nblk, na_cols, mpi_comm_rows, mpi_comm_cols, 1);   // now b = U
   for(i = 0; i < na_rows; i++)
      for(j = 0; j < na_cols; j++)
      {
         i_global = np_rows*nblk*(i/nblk) + i%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1; 
         j_global = np_cols*nblk*(j/nblk) + j%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
         if(i_global > j_global)
            b[i + j*na_rows] = 0; ;
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
   s_Cannons_Mult1(a, b, np_rows, np_cols, my_prow, my_pcol, a_desc, c, mpi_comm_cols, mpi_comm_rows);   // c has an upper triangular part of a*b
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   
   if(myid == 0)
      printf("\n Cannon's algorithm. 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, avTime, maxTime);
   
   ///// test PSTRMM /////////////////////////////////////////////////////////////////////////////////////
   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
   pstrmm_("R", "U", "N", "N", &na, &na, &done, b_copy, &one, &one, b_desc, c1, &one, &one, c_desc);
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("\n PSTRMM from ScaLAPACK. 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, avTime, maxTime);
   
   ///// test ELPA ///////////////////////////////////////////////////////////////////////////////////////////
   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
   elpa_mult_at_b_real_single('U', 'L', na, na, b_copy, na_rows, na_cols, a, na_rows, na_cols, nblk, mpi_comm_rows, mpi_comm_cols, work, na_rows, na_cols);   // work has a lower triangular part of b(H)*a
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("\n elpa_mult_at_b_real_single(U,L). 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, avTime, maxTime);
   
   pstran_(&na, &na, &done, work, &one, &one, a_desc, &dzero, c2, &one, &one, a_desc);   // c2 has an upper triangular part of a*b
         
   for(i = 0; i < na_rows; i++)
      for(j = 0; j < na_cols; j++)
      {
         i_global = np_rows*nblk*(i/nblk) + i%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1; 
         j_global = np_cols*nblk*(j/nblk) + j%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
         if(i_global > j_global)
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
      diff = diff + fabsf(c1[i]-c[i]);
   MPI_Reduce(&diff, &diffSum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between Cannon's and PSTRMM = %.15e, average = %.15e\n", diffSum, diffSum/(na*na));
   
   diff = 0;
   diffSum = 0; 
   for(i = 0; i < na_rows*na_cols; i++)
      diff = diff + fabsf(c2[i]-c[i]);
   MPI_Reduce(&diff, &diffSum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between Cannon's and ELPA = %.15e, average = %.15e\n", diffSum, diffSum/(na*na));
   
   diff = 0;
   diffSum = 0; 
   for(i = 0; i < na_rows*na_cols; i++)
      diff = diff + fabsf(c2[i]-c1[i]);
   MPI_Reduce(&diff, &diffSum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between PSTRMM and ELPA = %.15e, average = %.15e\n", diffSum, diffSum/(na*na));

   if(myid == 0)
      printf("\n\nTest2 ___________________________________________________________________ \n");
   
   for(i = 0; i < na_rows*na_cols; i++)
      c[i] = 0;
   for(i = 0; i < na_rows*na_cols; i++)
      c2[i] = 0;
   for(i = 0; i < na_rows*na_cols; i++)
      c1[i] = a_copy[i];
   
   ///// test PSTRMM /////////////////////////////////////////////////////////////////////////////////////
   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
   pstrmm_("R", "U", "N", "N", &na, &na, &done, b_copy, &one, &one, b_desc, c1, &one, &one, c_desc);
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("\n PSTRMM from ScaLAPACK. 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, avTime, maxTime);
   
   ///// test Cannon's ///////////////////////////////////////////////////////////////////////////////
   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
   s_Cannons_Mult1(a, b, np_rows, np_cols, my_prow, my_pcol, a_desc, c, mpi_comm_cols, mpi_comm_rows);
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   
   if(myid == 0)
      printf("\n Cannon's algorithm. 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, avTime, maxTime);
   
   ///// test ELPA ///////////////////////////////////////////////////////////////////////////////////////////
   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
   elpa_mult_at_b_real_single('U', 'L', na, na, b, na_rows, na_cols, a, na_rows, na_cols, nblk, mpi_comm_rows, mpi_comm_cols, work, na_rows, na_cols);
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("\n elpa_mult_at_b_real_single(U,L). 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, avTime, maxTime);

   pstran_(&na, &na, &done, work, &one, &one, a_desc, &dzero, c2, &one, &one, a_desc);   // c2 has an upper triangular part of a*b
   
   for(i = 0; i < na_rows; i++)
      for(j = 0; j < na_cols; j++)
      {
         i_global = np_rows*nblk*(i/nblk) + i%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1; 
         j_global = np_cols*nblk*(j/nblk) + j%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
         if(i_global > j_global)
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
      diff = diff + fabsf(c1[i]-c[i]);
   MPI_Reduce(&diff, &diffSum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between Cannon's and PSTRMM = %.15e, average = %.15e\n", diffSum, diffSum/(na*na));
   
   diff = 0;
   diffSum = 0; 
   for(i = 0; i < na_rows*na_cols; i++)
      diff = diff + fabsf(c2[i]-c[i]);
   MPI_Reduce(&diff, &diffSum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between Cannon's and ELPA = %.15e, average = %.15e\n", diffSum, diffSum/(na*na));
   
   diff = 0;
   diffSum = 0; 
   for(i = 0; i < na_rows*na_cols; i++)
      diff = diff + fabsf(c2[i]-c1[i]);
   MPI_Reduce(&diff, &diffSum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between PSTRMM and ELPA = %.15e, average = %.15e\n", diffSum, diffSum/(na*na));
   
   if(myid == 0)
      printf("\n\nTest3 ___________________________________________________________________ \n");
   
   for(i = 0; i < na_rows*na_cols; i++)
      c[i] = 0;
   for(i = 0; i < na_rows*na_cols; i++)
      c2[i] = 0;
   for(i = 0; i < na_rows*na_cols; i++)
      c1[i] = a_copy[i];
   
   ///// test PSTRMM /////////////////////////////////////////////////////////////////////////////////////
   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
   pstrmm_("R", "U", "N", "N", &na, &na, &done, b_copy, &one, &one, b_desc, c1, &one, &one, c_desc);
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("\n PSTRMM from ScaLAPACK. 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, avTime, maxTime);
   
   ///// test ELPA ///////////////////////////////////////////////////////////////////////////////////////////
   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
   elpa_mult_at_b_real_single('U', 'L', na, na, b, na_rows, na_cols, a, na_rows, na_cols, nblk, mpi_comm_rows, mpi_comm_cols, work, na_rows, na_cols);
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("\n elpa_mult_at_b_real_single(U,L). 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, avTime, maxTime);
   
   pstran_(&na, &na, &done, work, &one, &one, a_desc, &dzero, c2, &one, &one, a_desc);   // c2 has an upper triangular part of a*b
   
   ///// test Cannon's ///////////////////////////////////////////////////////////////////////////////
   for(i = 0; i < na_rows*na_cols; i++)
     c[i] = 0;
   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
   s_Cannons_Mult1(a, b, np_rows, np_cols, my_prow, my_pcol, a_desc, c, mpi_comm_cols, mpi_comm_rows);
   MPI_Barrier(MPI_COMM_WORLD);
   endTime = MPI_Wtime();
   localTime = endTime - startTime; 
   MPI_Reduce(&localTime, &avTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   avTime = avTime/nprocs;
   MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   
   if(myid == 0)
      printf("\n Cannon's algorithm. 0 proc in: %lf, average over procs = %lf, max = %lf\n\n", localTime, avTime, maxTime);

   for(i = 0; i < na_rows; i++)
      for(j = 0; j < na_cols; j++)
      {
         i_global = np_rows*nblk*(i/nblk) + i%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1; 
         j_global = np_cols*nblk*(j/nblk) + j%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
         if(i_global > j_global)
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
      diff = diff + fabsf(c1[i]-c[i]);
   MPI_Reduce(&diff, &diffSum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between Cannon's and PSTRMM = %.15e, average = %.15e\n", diffSum, diffSum/(na*na));
   
   diff = 0;
   diffSum = 0; 
   for(i = 0; i < na_rows*na_cols; i++)
      diff = diff + fabsf(c2[i]-c[i]);
   MPI_Reduce(&diff, &diffSum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between Cannon's and ELPA = %.15e, average = %.15e\n", diffSum, diffSum/(na*na));
   
   diff = 0;
   diffSum = 0; 
   for(i = 0; i < na_rows*na_cols; i++)
      diff = diff + fabsf(c2[i]-c1[i]);
   MPI_Reduce(&diff, &diffSum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   if(myid == 0)
      printf("Summed difference between PSTRMM and ELPA = %.15e, average = %.15e\n", diffSum, diffSum/(na*na));
   
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