//    This file is part of ELPA.
//
//    The ELPA library was originally created by the ELPA consortium,
//    consisting of the following organizations:
//
//    - Max Planck Computing and Data Facility (MPCDF), formerly known as
//      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
//    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
//      Informatik,
//    - Technische Universität München, Lehrstuhl für Informatik mit
//      Schwerpunkt Wissenschaftliches Rechnen ,
//    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
//    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
//      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
//      and
//    - IBM Deutschland GmbH
//
//    This particular source code file has been developed within the ELPA-AEO //
//    project, which has been a joint effort of
//
//    - Max Planck Computing and Data Facility (MPCDF), formerly known as
//      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
//    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
//      Informatik,
//    - Technische Universität München, Lehrstuhl für Informatik mit
//      Schwerpunkt Wissenschaftliches Rechnen ,
//    - Technische Universität München, Lehrstuhl für Theoretische Chemie,
//    - Fritz-Haber-Institut, Berlin, Abt. Theorie,

//    More information can be found here:
//    http://elpa.mpcdf.mpg.de/ and
//    http://elpa-aeo.mpcdf.mpg.de
//
//    ELPA is free software: you can redistribute it and/or modify
//    it under the terms of the version 3 of the license of the
//    GNU Lesser General Public License as published by the Free
//    Software Foundation.
//
//    ELPA is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU Lesser General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
//
//    ELPA reflects a substantial effort on the part of the original
//    ELPA consortium, and we ask you to respect the spirit of the
//    license that we chose: i.e., please contribute any changes you
//    may have back to the original ELPA library distribution, and keep
//    any derivatives of ELPA under the same license that we chose for
//    the original distribution, the GNU Lesser General Public License.
//
// Author: Valeriy Manin (Bergische Universität Wuppertal)
// integreated into the ELPA library Pavel Kus, Andeas Marek (MPCDF)


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

void pdlacpy_(char*, int*, int*, double*, int*, int*, int*, double*, int*, int*, int*);
void dlacpy_(char*, int*, int*, double*, int*, double*, int*);
void dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*); 
void pdtran_(int*, int*, double*, double*, int*, int*, int*, double*, double*, int*, int*, int*);
void pdelset_(double*, int*, int*, int*, double*);
void pdsymm_(char*, char*, int*, int*, double*, double*, int*, int*, int*, double*, int*, int*, int*, double*, double*, int*, int*, int*);
void pdpotrf_(char*, int*, double*, int*, int*, int*, int*);
void pdsyngst_(int*, char*, int*, double*, int*, int*, int*, double*, int*, int*, int*, double*, double*, int*, int*);
void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
int numroc_(int*, int*, int*, int*, int*);
void set_up_blacsgrid_f1(int, int*, int*, int*, int*, int*, int*, int*);
void pdtrtrs_(char*, char*, char*, int*, int*, double*, int*, int*, int*, double*, int*, int*, int*, int*);
void pdsyevr_(char*, char*, char*, int*, double*, int*, int*, int*, int*, int*, int*, int*, int*, int*, double*, double*, int*, int*, int*, double*, int*, int*, int*, int*);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////// My function for reduction //////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void d_Cannons_Reduction(double* A, double* U, int np_rows, int np_cols, int my_prow, int my_pcol, int* a_desc, double *Res, int ToStore, MPI_Comm row_comm, MPI_Comm col_comm)
{
   // Input matrices: 
      // - A: full matrix
      // - U: upper triangular matrix U(-1)
   // Output matrix: 
      // - Res = U(-H)*A*U(-1)
   // row_comm: communicator along rows
   // col_comm: communicator along columns
  
   int na, nblk, i, j, Size_send_A, Size_receive_A, Size_send_U, Size_receive_U, Buf_rows, Buf_cols, where_to_send_A, from_where_to_receive_A, where_to_send_U, from_where_to_receive_U, last_proc_row, last_proc_col, cols_in_buffer_A, rows_in_buffer_A, intNumber;
   double *Buf_to_send_A, *Buf_to_receive_A, *Buf_to_send_U, *Buf_to_receive_U, *double_ptr, *Buf_A, *Buf_pos, *U_local_start, *Res_ptr, *M, *M_T, *A_local_start, *U_local_start_curr, *U_stored, *CopyTo, *CopyFrom, *U_to_calc;
   int ratio, num_of_iters, cols_in_buffer, rows_in_block, rows_in_buffer, curr_col_loc, cols_in_block, curr_col_glob, curr_row_loc, Size_receive_A_now, Nb, owner, cols_in_buffer_A_now;
   int  row_of_origin_U, rows_in_block_U, num_of_blocks_in_U_buffer, k, startPos, cols_in_buffer_U, rows_in_buffer_U, col_of_origin_A, curr_row_loc_res, curr_row_loc_A, curr_col_glob_res; 
   int curr_col_loc_res, curr_col_loc_buf, proc_row_curr, curr_col_loc_U, A_local_index, LDA_A, LDA_A_new, index_row_A_for_LDA, ii, rows_in_block_U_curr, width, row_origin_U, rows_in_block_A, cols_in_buffer_A_my_initial, rows_in_buffer_A_my_initial, proc_col_min;
   int *SizesU;
   int Size_U_skewed, Size_U_stored, Curr_pos_in_U_stored, rows_in_buffer_A_now;
   double done = 1.0;
   double dzero = 0.0;
   int one = 1; 
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
   
   if(ToStore > (np_rows -1))
      if((my_prow == 0)&&(my_pcol == 0))
         printf("Buffering level is larger than (np_rows-1) !!!\n");
   if((my_prow == 0)&&(my_pcol == 0))
         printf("Buffering level = %d\n", ToStore); 
   
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
      
   intNumber = ceil((double)na/(double)(np_cols*nblk));   // max. possible number of the local block columns of U
   Size_U_stored = ratio*nblk*nblk*intNumber*(intNumber+1)/2 + 2;   // number of local elements from the upper triangular part that every proc. has (max. possible value among all the procs.)
   
   U_stored = malloc((Size_U_stored*(ToStore+1))*sizeof(double));
   SizesU = malloc(ToStore*sizeof(int));  // here will be stored the sizes of the buffers of U that I have stored     
   Buf_to_send_A = malloc(ratio*Buf_cols*Buf_rows*sizeof(double));
   Buf_to_receive_A = malloc(ratio*Buf_cols*Buf_rows*sizeof(double));
   Buf_to_send_U = malloc(Size_U_stored*sizeof(double));
   Buf_to_receive_U = malloc(Size_U_stored*sizeof(double));
   if(ratio != 1)
      Buf_A = malloc(Buf_cols*Buf_rows*sizeof(double));   // in this case we will receive data into initial buffer and after place block-columns to the needed positions of buffer for calculation
   M = malloc(na_rows*na_cols*sizeof(double));
   M_T = malloc(na_rows*na_cols*sizeof(double));
   for(i = 0; i < na_rows*na_cols; i++)
      M[i] = 0; 
        
   ////////////////////////////////////////////////////////////// initial reordering of A ///////////////////////////////////////////////////////////////////////////////////////// 
   
   // here we assume, that np_rows < np_cols; then I will send to the number of processors equal to <ratio> with the "leap" equal to np_rows; the same holds for receive  
   if(ratio != 1)
      dlacpy_("A", &na_rows, &na_cols, A, &na_rows, Buf_to_send_A, &na_rows);   // copy my buffer to send
   Size_receive_A = 0; 
   
   // receive from different processors and place in my buffer for calculation;
   for(i = 0; i < ratio; i++)
   {
      where_to_send_A = (my_pcol - my_prow - i*np_rows + np_cols)%np_cols;                
      from_where_to_receive_A = (my_pcol + my_prow + i*np_rows)%np_cols;
      
      // send and receive in the row_comm
      if(ratio != 1)   // if grid is not square
      {
         if(where_to_send_A != my_pcol)
         {
            MPI_Sendrecv(Buf_to_send_A, na_cols*na_rows, MPI_DOUBLE, where_to_send_A, 0, Buf_A, na_rows*Buf_cols, MPI_DOUBLE, from_where_to_receive_A, 0, row_comm, &status);
            MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_A_now);
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
         
         intNumber = ceil((double)Size_receive_A_now/(double)nblk);   // how many block-columns I have received
         for(j = 0; j < intNumber; j++)
         {
            width = nblk; // width of the current block column
            if(nblk*(j+1) > Size_receive_A_now)
               width = Size_receive_A_now - nblk*j; 
            dlacpy_("A", &na_rows, &width, CopyFrom, &na_rows, CopyTo, &na_rows);
            CopyTo = CopyTo + na_rows*nblk*ratio; 
            CopyFrom = CopyFrom + na_rows*nblk; 
         }
      }
      else  // if grid is square then simply receive from one processor to a calculation buffer
         if(my_prow > 0)
         {
            dlacpy_("A", &na_rows, &na_cols, A, &na_rows, Buf_to_send_A, &na_rows);   // copy my buffer to send
            MPI_Sendrecv(Buf_to_send_A, na_cols*na_rows, MPI_DOUBLE, where_to_send_A, 0, Buf_to_receive_A, na_rows*Buf_cols, MPI_DOUBLE, from_where_to_receive_A, 0, row_comm, &status);
            MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_A);
            Size_receive_A = Size_receive_A/na_rows;       // how many columns of A I have received
         }
         else
         {
            dlacpy_("A", &na_rows, &na_cols, A, &na_rows, Buf_to_receive_A, &na_rows);   // copy A to the received buffer if I do not need to send
            Size_receive_A = na_cols; 
         }
   }
   
   ////////////////////////////////////////////////////////////// initial reordering of U //////////////////////////////////////////////////////
     
   // form array to send by block-columns
   num_of_iters = ceil((double)na_cols/(double)nblk);             // number my of block-columns
   
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
      curr_col_loc = 1;   //ceil((double)(((double)my_prow - (double)my_pcol)/(double)np_cols)) always will give 1 since np_cols > np_rows 
      
   num_of_iters = num_of_iters - curr_col_loc;   // I will exclude the first <curr_col_loc> block-columns since they do not have blocks from the upper part of matrix U
   curr_col_loc = curr_col_loc*nblk;             // local index of the found block-column

   if(my_pcol >= my_prow )
      rows_in_block = ceil(((double)(my_pcol + 1) - (double)my_prow)/(double)np_rows)*nblk;
   else
      rows_in_block = ratio*nblk;
   
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
      }
      curr_col_loc = curr_col_loc + nblk;      // go to the next local block-column of my local array U 
      rows_in_block = rows_in_block + ratio*nblk;
   }
   rows_in_buffer = rows_in_block - ratio*nblk;    // remove redundant addition from the previous loop 
   *Buf_pos = (double)rows_in_buffer; // write number of the rows at the end of the buffer; we will need this for further multiplications on the other processors
   Size_send_U = Size_send_U + 1;
   
   //send and receive
   if(where_to_send_U != my_prow)
   {   
      // send and receive in the col_comm
      MPI_Sendrecv(Buf_to_send_U, Size_send_U, MPI_DOUBLE, where_to_send_U, 0, Buf_to_receive_U, Buf_rows*na_cols, MPI_DOUBLE, from_where_to_receive_U, 0, col_comm, &status); 
      MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_U); // find out how many elements I have received 
   }
   else // if I do not need to send 
      Size_receive_U = Size_send_U;         // how many elements I "have received"; the needed data I have already copied to the "receive" buffer
      
   for(i = 0; i < Size_receive_U; i++)
      U_stored[i] = Buf_to_receive_U[i];
   Size_U_skewed = Size_receive_U; 
   Curr_pos_in_U_stored = Size_U_skewed;

   //////////////////////////////////////////////////////////////////////// main loop /////////////////////////////////////////////////////
   where_to_send_A = (my_pcol - 1 + np_cols)%np_cols;
   from_where_to_receive_A = (my_pcol + 1)%np_cols;
   where_to_send_U = (my_prow - 1 + np_rows)%np_rows;
   from_where_to_receive_U = (my_prow + 1)%np_rows;
   
   for(j = 1; j < np_rows; j++)
   {
      // at this moment I need to send to neighbour what I have in the "received" arrays; that is why exchange pointers of the "received" and "send" arrays
      double_ptr = Buf_to_send_A; 
      Buf_to_send_A = Buf_to_receive_A; 
      Buf_to_receive_A = double_ptr; 
      
      double_ptr = Buf_to_send_U; 
      Buf_to_send_U = Buf_to_receive_U; 
      Buf_to_receive_U = double_ptr;
      
      ///// shift for A ////////////////////////////////////////////////////////////
      Size_send_A = Size_receive_A;  // number of block-columns of A and block-rows of U to send (that I have received on the previous step) 
      MPI_Isend(Buf_to_send_A, Size_send_A*na_rows, MPI_DOUBLE, where_to_send_A, 0, row_comm, &request_A_Send); 
      MPI_Irecv(Buf_to_receive_A, Buf_cols*na_rows*ratio, MPI_DOUBLE, from_where_to_receive_A, 0, row_comm, &request_A_Recv);
         
      ///// shift for U /////////////////////////////////////////////
      Size_send_U = Size_receive_U; 
      MPI_Isend(Buf_to_send_U, Size_send_U, MPI_DOUBLE, where_to_send_U, 0, col_comm, &request_U_Send); 
      MPI_Irecv(Buf_to_receive_U, Buf_rows*na_cols, MPI_DOUBLE, from_where_to_receive_U, 0, col_comm, &request_U_Recv); 
      
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
    
      num_of_blocks_in_U_buffer = ceil(((double)cols_in_buffer - (double)curr_col_loc_buf)/(double)nblk); 
      
      startPos = (curr_col_loc_buf + nblk)*curr_col_loc_buf/2;
      U_local_start = &Buf_to_send_U[startPos];
      Res_ptr = &M[curr_col_loc_res*na_rows];
  
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
               dgemm_("N", "N", &rows_in_block_A, &cols_in_block, &rows_in_block_U, &done, Buf_to_send_A, &na_rows, U_local_start, &rows_in_block_U, &dzero, Res_ptr, &na_rows);
            else 
               dgemm_("N", "N", &rows_in_block_A, &cols_in_block, &rows_in_block_U, &done, Buf_to_send_A, &na_rows, U_local_start, &rows_in_block_U, &done, Res_ptr, &na_rows);
      
         U_local_start = U_local_start + rows_in_block_U*cols_in_block;
         curr_col_loc_res = curr_col_loc_res + nblk;
         Res_ptr = &M[curr_col_loc_res*na_rows];
         curr_col_loc_buf = curr_col_loc_buf + nblk;  
      } 
     
      MPI_Wait(&request_A_Send, &status);
      MPI_Wait(&request_A_Recv, &status);
      MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_A); // find out how many elements I have received 
      Size_receive_A = Size_receive_A/na_rows;
      
      MPI_Wait(&request_U_Send, &status);
      MPI_Wait(&request_U_Recv, &status);
      MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_U); // find out how many elements I have received  
      
       //// write in the buffer for later use //////////////////////////////7
      if(j <= ToStore)
      {
         for(k = 0; k < Size_receive_U; k++)
            U_stored[Curr_pos_in_U_stored + k] = Buf_to_receive_U[k]; 
         Curr_pos_in_U_stored = Curr_pos_in_U_stored + Size_receive_U; 
         SizesU[j-1] = Size_receive_U; 
      }
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
    
   num_of_blocks_in_U_buffer = ceil(((double)cols_in_buffer - (double)curr_col_loc_buf)/(double)nblk); 
      
   startPos = (curr_col_loc_buf + nblk)*curr_col_loc_buf/2;
   U_local_start = &Buf_to_receive_U[startPos];
   Res_ptr = &M[curr_col_loc_res*na_rows];
  
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
            dgemm_("N", "N", &rows_in_block_A, &cols_in_block, &rows_in_block_U, &done, Buf_to_receive_A, &na_rows, U_local_start, &rows_in_block_U, &dzero, Res_ptr, &na_rows);
         else 
            dgemm_("N", "N", &rows_in_block_A, &cols_in_block, &rows_in_block_U, &done, Buf_to_receive_A, &na_rows, U_local_start, &rows_in_block_U, &done, Res_ptr, &na_rows);
      
      U_local_start = U_local_start + rows_in_block_U*cols_in_block;
      curr_col_loc_res = curr_col_loc_res + nblk;
      Res_ptr = &M[curr_col_loc_res*na_rows];
      curr_col_loc_buf = curr_col_loc_buf + nblk;  
   }  
   
   ///////////////////// Now M has an upper part of A*U(-1) ///////////////////////////////////////////////
   
   pdtran_(&na, &na, &done, M, &one, &one, a_desc, &dzero, M_T, &one, &one, a_desc);     // now M_T has lower part of U(-H)*A 
 
   ////////////////////////////////////////////////// start algorithm to find lower part of U(-H)*A*U(-1) //////////////////////////
           
   /////////////////////////////////////////////////////////////// initial reordering of A ////////////////////////////////////////////////
   
   // here we assume, that np_rows < np_cols; then I will send to the number of processors equal to <ratio> with the "leap" equal to np_rows; the same holds for receive  
   if((ratio != 1)||(my_prow != 0))   // if grid is rectangular or my_prow is not 0
      Buf_pos = Buf_to_send_A;     // I will copy to the send buffer
   else
      Buf_pos = Buf_to_receive_A;  // if grid is square and my_prow is 0, then I will copy to the received buffer
   
   // form array to send by block-columns; we need only lower triangular part
   num_of_iters = ceil((double)na_cols/(double)nblk);             // number my of block-columns
   
   cols_in_buffer_A_my_initial = 0;
   Size_send_A = 0; 
   
   if(my_pcol <= my_prow)  // if I am from the lower part of grid
   {
      curr_row_loc = 0;     // I will copy all my block-rows
      rows_in_buffer_A_my_initial = na_rows;
   }
   else
   {
      curr_row_loc = ceil((double)(((double)my_pcol - (double)my_prow)/(double)np_rows))*nblk; // I will skip some of my block-rows
      rows_in_buffer_A_my_initial = na_rows - curr_row_loc;   
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
         A_local_start = &M_T[curr_col_loc*na_rows + curr_row_loc];
         dlacpy_("A", &rows_in_block, &cols_in_block, A_local_start, &na_rows, Buf_pos, &rows_in_block);     // copy lower part of block-column in the buffer with LDA = length of the lower part of block-column 
         Buf_pos = Buf_pos + rows_in_block*cols_in_block;
         Size_send_A = Size_send_A + rows_in_block*cols_in_block; 
         cols_in_buffer_A_my_initial = cols_in_buffer_A_my_initial + cols_in_block; 
      }
      curr_row_loc = curr_row_loc + ratio*nblk;
   }
   *Buf_pos = (double)cols_in_buffer_A_my_initial; // write number of the columns at the end of the buffer; we will need this for furhter multiplications on the other processors
   Size_send_A = Size_send_A + 1;
   
   // now we have the local buffer to send
   // find the lowest processor column among those who will send me
   proc_col_min = np_cols; 
   for(i = 0; i < ratio; i++)
   {
      from_where_to_receive_A = (my_pcol + my_prow + i*np_rows)%np_cols;
      if(from_where_to_receive_A < proc_col_min)
         proc_col_min = from_where_to_receive_A;
   }
   // do communications and form local buffers for calculations
   Size_receive_A = 0;       // size of the accumulated buffer
   cols_in_buffer_A = 0;     // number of columns in the accumulated buffer
   rows_in_buffer_A = 0;     // number of rows in the accumulated buffer
   for(i = 0; i < ratio; i++)
   {
      where_to_send_A = (my_pcol - my_prow - i*np_rows + np_cols)%np_cols;                
      from_where_to_receive_A = (my_pcol + my_prow + i*np_rows)%np_cols;
      
      // send and receive in the row_comm
      if(ratio != 1)   // if grid is not square
      {
         if(where_to_send_A != my_pcol)   // if I need to send and receive on this step
         {
            MPI_Sendrecv(Buf_to_send_A, Size_send_A, MPI_DOUBLE, where_to_send_A, 0, Buf_A, Size_U_stored, MPI_DOUBLE, from_where_to_receive_A, 0, row_comm, &status);
            MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_A_now);
            Size_receive_A = Size_receive_A + Size_receive_A_now - 1; // we need only number of elements, so exclude information about cols_in_buffer_A
            
            cols_in_buffer_A_now = Buf_A[Size_receive_A_now-1];
            cols_in_buffer_A = cols_in_buffer_A + cols_in_buffer_A_now; 
            
            // determine number of rows in the received buffer
            if(from_where_to_receive_A <= my_prow)  // if source is from the lower part of grid
            {
               rows_in_buffer_A_now = na_rows;
            }
            else
            {
               rows_in_buffer_A_now = na_rows - ceil((double)(((double)from_where_to_receive_A - (double)my_prow)/(double)np_rows))*nblk; // some of the block-rows have been skipped
            }
            if(rows_in_buffer_A < rows_in_buffer_A_now)
               rows_in_buffer_A = rows_in_buffer_A_now; 

            intNumber = from_where_to_receive_A/np_rows; // how many processors will send me blocks, such that they will be placed before the current blocks  
            if(proc_col_min <= my_prow)   // if among procs who will send me there is one with the full sets of block-rows in the lower part
               CopyTo = &Buf_to_receive_A[nblk*(na_rows*intNumber - nblk*(intNumber-1)*intNumber/2)];  // here I will copy to; formula based on arithm. progression
            else
               CopyTo = &Buf_to_receive_A[nblk*(na_rows*intNumber - nblk*intNumber*(intNumber+1)/2)];  // otherwise, the first block-column will be shorter by one block
            CopyFrom = Buf_A; 
         }
         else  // if I need to send to myself on this step, then I will copy from Buf_to_send_L to Buf_to_receive_A
         {
            cols_in_buffer_A_now = cols_in_buffer_A_my_initial;
            cols_in_buffer_A = cols_in_buffer_A + cols_in_buffer_A_now; 
            
            rows_in_buffer_A_now = rows_in_buffer_A_my_initial;
            if(rows_in_buffer_A < rows_in_buffer_A_now)
               rows_in_buffer_A = rows_in_buffer_A_now; 

            intNumber = my_pcol/np_rows; // how many processors will send me blocks, such that they will be placed before the current blocks  
            if(proc_col_min <= my_prow)   // if among procs who will send me there is one with the full sets of block-rows in the lower part
               CopyTo = &Buf_to_receive_A[nblk*(na_rows*intNumber - nblk*(intNumber-1)*intNumber/2)];  // here I will copy to; formula based on arithm. progression
            else
               CopyTo = &Buf_to_receive_A[nblk*(na_rows*intNumber - nblk*intNumber*(intNumber+1)/2)];  // otherwise, the first block-column will be shorter by one block
            CopyFrom = Buf_to_send_A;  

            Size_receive_A = Size_receive_A + Size_send_A - 1;
         }
            
         // copy by block-columns
         intNumber = ceil((double)cols_in_buffer_A_now/(double)nblk);  // how many block-columns I have received on this iteration
         rows_in_block = rows_in_buffer_A_now; 
         for(j = 0; j < intNumber; j++)
         {
            if((j+1)*nblk < cols_in_buffer_A_now)
               cols_in_block = nblk; 
            else
               cols_in_block = cols_in_buffer_A_now - j*nblk;
               
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
            MPI_Sendrecv(Buf_to_send_A, Size_send_A, MPI_DOUBLE, where_to_send_A, 0, Buf_to_receive_A, Size_U_stored, MPI_DOUBLE, from_where_to_receive_A, 0, row_comm, &status);
            MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_A);
            cols_in_buffer_A = (int)Buf_to_receive_A[Size_receive_A-1];
            if(from_where_to_receive_A <= my_prow)  // if source is from the lower part of grid
            {
               rows_in_buffer_A = na_rows;
            }
            else
            {
               rows_in_buffer_A = na_rows - ceil((double)(((double)from_where_to_receive_A - (double)my_prow)/(double)np_rows))*nblk; // some of the block-rows have been skipped
            }
         }
         else    // if my_prow == 0, then I have already everything in my Buf_to_receive_A buffer
         {
            Size_receive_A = Size_send_A;
            rows_in_buffer_A = rows_in_buffer_A_my_initial;
            cols_in_buffer_A = cols_in_buffer_A_my_initial;
         }
      }
   }
   if(ratio != 1)
   {
      Buf_to_receive_A[Size_receive_A] = cols_in_buffer_A;
      Buf_to_receive_A[Size_receive_A + 1] = rows_in_buffer_A;
      Size_receive_A = Size_receive_A + 2;
   }
   else
   {
      Buf_to_receive_A[Size_receive_A] = rows_in_buffer_A;
      Size_receive_A = Size_receive_A + 1;
   }

   ////////////////////////////////////////////////////////////// initial reordering of U: restore skewed U from the first multiplication ///////////////////////////
   
   Size_receive_U = Size_U_skewed;
   U_to_calc = U_stored;
   
   //////////////////////////////////////////////////////////////////////// main loop ////////////////////////////////////////////////////////////////////////////////
   
   where_to_send_A = (my_pcol - 1 + np_cols)%np_cols;
   from_where_to_receive_A = (my_pcol + 1)%np_cols;
   where_to_send_U = (my_prow - 1 + np_rows)%np_rows;
   from_where_to_receive_U = (my_prow + 1)%np_rows;
   Curr_pos_in_U_stored = Size_U_skewed;
  
   for(j = 1; j < np_rows; j++)
   {
      // at this moment I need to send to neighbour what I have in the "received" arrays; that is why exchange pointers of the "received" and "send" arrays
      double_ptr = Buf_to_send_A; 
      Buf_to_send_A = Buf_to_receive_A; 
      Buf_to_receive_A = double_ptr; 
      
      if (j > ToStore)
      {
         double_ptr = Buf_to_send_U; 
         Buf_to_send_U = Buf_to_receive_U; 
         Buf_to_receive_U = double_ptr;
      }
        
      ///// shift for A ////////////////////////////////////////////////////////////
      Size_send_A = Size_receive_A; 
      MPI_Isend(Buf_to_send_A, Size_send_A, MPI_DOUBLE, where_to_send_A, 0, row_comm, &request_A_Send); 
      MPI_Irecv(Buf_to_receive_A, ratio*Size_U_stored, MPI_DOUBLE, from_where_to_receive_A, 0, row_comm, &request_A_Recv);
         
      ///// shift for U /////////////////////////////////////////////
      Size_send_U = Size_receive_U; 
      if (j > ToStore)
      {
         if(j > ToStore + 1)
         {
            MPI_Isend(Buf_to_send_U, Size_send_U, MPI_DOUBLE, where_to_send_U, 0, col_comm, &request_U_Send); 
            U_to_calc = Buf_to_send_U;
         }
         else
            MPI_Isend(U_to_calc, Size_send_U, MPI_DOUBLE, where_to_send_U, 0, col_comm, &request_U_Send);
         MPI_Irecv(Buf_to_receive_U, Size_U_stored, MPI_DOUBLE, from_where_to_receive_U, 0, col_comm, &request_U_Recv);
      }
      
      ///// multiplication ////////////////////////////////////////////////////////////////////////////////////////////
      rows_in_buffer_U = (int)U_to_calc[Size_receive_U-1];
      row_of_origin_U = (my_pcol + my_prow + np_cols + j - 1)%np_rows;
      if(my_pcol >= row_of_origin_U)
         cols_in_buffer_U = na_cols;
      else
         cols_in_buffer_U = na_cols - nblk;
      
      cols_in_buffer_A = (int)Buf_to_send_A[Size_receive_A-2];
      rows_in_buffer_A = (int)Buf_to_send_A[Size_receive_A-1];
      // find the minimal pcol among those who have sent A for this iteration
      col_of_origin_A = np_cols; 
      for(i = 0; i < ratio; i++)
      {
         intNumber = (my_pcol + my_prow + i*np_rows + np_cols + j - 1)%np_cols;
         if(intNumber < col_of_origin_A)
            col_of_origin_A = intNumber;
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
      
      U_local_start = U_to_calc;
      
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
      
         curr_row_loc_A = curr_row_loc_res;     // it is impossible, that both col_of_origin_L and row_of_origin_U are from upper part
         if(col_of_origin_A > my_prow)
            curr_row_loc_A = curr_row_loc_A - nblk;  
        
         rows_in_block = rows_in_buffer_A - curr_row_loc_A;    // rows in current block of A
              
         curr_col_loc_U = i*nblk;   // local index in the buffer U of the current column
      
         if((curr_col_loc_U + nblk) <= cols_in_buffer_U)
            cols_in_block = nblk;      // number columns in block of U which will take part in this calculation
         else
            cols_in_block = cols_in_buffer_U - curr_col_loc_U; 
      
         if(rows_in_block_U > rows_in_buffer_U)
            rows_in_block_U = rows_in_buffer_U;     // rows in current column of U; also a leading dimension for U
 
         A_local_index = curr_row_loc_A;
         A_local_start = &Buf_to_send_A[A_local_index];
         Res_ptr = &Res[curr_col_loc_res*na_rows + curr_row_loc_res];

         LDA_A = rows_in_buffer_A;
         LDA_A_new = LDA_A;
         if ((rows_in_block > 0)&&(cols_in_block > 0))
         {
            U_local_start_curr = U_local_start; 
 
            // loop over block-columns of the "active" part of L buffer
            for (ii = 0; ii < ceil((double)rows_in_block_U/(double)nblk); ii++)
            {
               if((ii+1)*nblk <= cols_in_buffer_A)
                  rows_in_block_U_curr = nblk; 
               else
                  rows_in_block_U_curr = cols_in_buffer_A - ii*nblk;  

               if((j == 1)&&(ii == 0))
                  dgemm_("N", "N", &rows_in_block, &cols_in_block, &rows_in_block_U_curr, &done, A_local_start, &LDA_A, U_local_start_curr, &rows_in_block_U, &dzero, Res_ptr, &na_rows); 
               else 
                  dgemm_("N", "N", &rows_in_block, &cols_in_block, &rows_in_block_U_curr, &done, A_local_start, &LDA_A, U_local_start_curr, &rows_in_block_U, &done, Res_ptr, &na_rows);

               LDA_A_new = LDA_A_new - nblk;
      
               U_local_start_curr = U_local_start_curr + rows_in_block_U_curr; 
               A_local_index = A_local_index - LDA_A + LDA_A*nblk + LDA_A_new; 
               A_local_start = &Buf_to_send_A[A_local_index];
               LDA_A = LDA_A_new; 
            }
         }
      
         U_local_start = U_local_start + rows_in_block_U*cols_in_block;
         curr_col_loc_res = curr_col_loc_res + nblk; 
         rows_in_block_U = rows_in_block_U + ratio*nblk;
      }    
      
      MPI_Wait(&request_A_Send, &status);
      MPI_Wait(&request_A_Recv, &status);
      MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_A); // find out how many elements I have received 
      
      if (j <= ToStore)
      {
         U_to_calc = &U_stored[Curr_pos_in_U_stored];
         Curr_pos_in_U_stored = Curr_pos_in_U_stored + SizesU[j-1]; 
         Size_receive_U =  SizesU[j-1];
      }
      else
      {
         MPI_Wait(&request_U_Send, &status);
         MPI_Wait(&request_U_Recv, &status);
         MPI_Get_count(&status, MPI_DOUBLE, &Size_receive_U); // find out how many elements I have received  
      }
   }
   
   /////// do the last multiplication //////////////
   if(ToStore < np_rows - 1)
      U_to_calc = Buf_to_receive_U;
   rows_in_buffer_U = (int)U_to_calc[Size_receive_U-1];
   row_of_origin_U = (my_pcol + my_prow + np_cols + j - 1)%np_rows;     
   if(my_pcol >= row_of_origin_U)
      cols_in_buffer_U = na_cols;
   else
      cols_in_buffer_U = na_cols - nblk;
      
   cols_in_buffer_A = (int)Buf_to_receive_A[Size_receive_A-2];
   rows_in_buffer_A = (int)Buf_to_receive_A[Size_receive_A-1];
   // find the minimal pcol among those who have sent A for this iteration
   col_of_origin_A = np_cols; 
   for(i = 0; i < ratio; i++)
   {
      intNumber = (my_pcol + my_prow + i*np_rows + np_cols + np_rows - 1)%np_cols;
      if(intNumber < col_of_origin_A)
         col_of_origin_A = intNumber;
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
      
   U_local_start = U_to_calc;
      
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
      
      curr_row_loc_A = curr_row_loc_res;     // it is impossible, that both col_of_origin_L and row_of_origin_U are from upper part
      if(col_of_origin_A > my_prow)
         curr_row_loc_A = curr_row_loc_A - nblk;
      
      rows_in_block = rows_in_buffer_A - curr_row_loc_A;    //rows in current block of  
              
      curr_col_loc_U = i*nblk;   // local index in the buffer U of the current column
      
      if((curr_col_loc_U + nblk) <= cols_in_buffer_U)
         cols_in_block = nblk;      // number columns in block of U which will take part in this calculation
      else
         cols_in_block = cols_in_buffer_U - curr_col_loc_U; 
      
      if(rows_in_block_U > rows_in_buffer_U)
         rows_in_block_U = rows_in_buffer_U; 
 
      A_local_index = curr_row_loc_A;
      A_local_start = &Buf_to_receive_A[A_local_index];
      Res_ptr = &Res[curr_col_loc_res*na_rows + curr_row_loc_res];
      LDA_A = rows_in_buffer_A; 
      LDA_A_new = LDA_A; 
      if ((rows_in_block > 0) &&(cols_in_block > 0))
      {
         U_local_start_curr = U_local_start; 

         // loop over block-columns of the "active" part of L buffer
         for (ii = 0; ii < ceil((double)rows_in_block_U/(double)nblk); ii++)
         {
            if((ii+1)*nblk <= cols_in_buffer_A)
               rows_in_block_U_curr = nblk; 
            else
               rows_in_block_U_curr = cols_in_buffer_A - ii*nblk;  

            if((j == 1)&&(ii == 0))
               dgemm_("N", "N", &rows_in_block, &cols_in_block, &rows_in_block_U_curr, &done, A_local_start, &LDA_A, U_local_start_curr, &rows_in_block_U, &dzero, Res_ptr, &na_rows); 
            else 
               dgemm_("N", "N", &rows_in_block, &cols_in_block, &rows_in_block_U_curr, &done, A_local_start, &LDA_A, U_local_start_curr, &rows_in_block_U, &done, Res_ptr, &na_rows);

            LDA_A_new = LDA_A_new - nblk;
              
            U_local_start_curr = U_local_start_curr + rows_in_block_U_curr; 
            A_local_index = A_local_index - (LDA_A - rows_in_block) + LDA_A*nblk + LDA_A_new - rows_in_block; 
            A_local_start = &Buf_to_receive_A[A_local_index];
            LDA_A = LDA_A_new;
         }
      }
      
      U_local_start = U_local_start + rows_in_block_U*cols_in_block;
      curr_col_loc_res = curr_col_loc_res + nblk; 
      rows_in_block_U = rows_in_block_U + ratio*nblk;
   }
   
   pdtran_(&na, &na, &done, Res, &one, &one, a_desc, &dzero, M, &one, &one, a_desc);
   pdlacpy_("U", &na, &na, M, &one, &one, a_desc, Res, &one, &one, a_desc);
      
   free(Buf_to_send_A);
   free(Buf_to_receive_A);
   free(Buf_to_send_U);
   free(Buf_to_receive_U);
   free(M); 
   free(M_T);
   if(ratio != 1)
      free(Buf_A);
   free(U_stored);
}

void d_Cannons_triang_rectangular(double* U, double* B, int np_rows, int np_cols, int my_prow, int my_pcol, int* U_desc, int*b_desc, double *Res, MPI_Comm row_comm, MPI_Comm col_comm)
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
   nev = (int)na*0.33;
   if (myid == 0)
      printf("Number of eigenvalues: %d\n", nev);
   nblk = atoi(argv[2]);
   Liwork = 20*na;
   double BuffLevel = atof(argv[3]);

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
      d_Cannons_Reduction(a, b, np_rows, np_cols, my_prow, my_pcol, a_desc, AUinv, BuffLevelInt, mpi_comm_cols, mpi_comm_rows);
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
      d_Cannons_triang_rectangular(b, EigenVectors, np_rows, np_cols, my_prow, my_pcol, b_desc, EigenVectors_desc1, EigVectors_gen, mpi_comm_cols, mpi_comm_rows);
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
      if(myid == 0)
         printf("max accumulated diff of the Ax-lamBx = %.15e \n", diff_max);
      if(myid == 0)
         printf("_______________________________________________________________________________________________________\n");
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
      if(myid == 0)
         printf("max accumulated diff of the Ax-lamBx = %.15e \n", diff_max);
      if(myid == 0)
         printf("_______________________________________________________________________________________________________\n");
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
      if(myid == 0)
         printf("max accumulated diff of the Ax-lamBx = %.15e \n", diff_max);
      if(myid == 0)
         printf("_______________________________________________________________________________________________________\n");
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
   return 0;
}
