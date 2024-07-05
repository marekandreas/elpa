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
// integrated into the ELPA library Pavel Kus, Andeas Marek (MPCDF)
// ported to GPU by Peter Karpov (MPCDF)

#ifdef HAVE_64BIT_INTEGER_MATH_SUPPORT
#define C_INT_TYPE_PTR long int*
#define C_INT_TYPE long int
#define BLAS_KIND c_int64_t
#else
#define C_INT_TYPE_PTR int*
#define C_INT_TYPE int
#define BLAS_KIND c_int
#endif
#ifdef HAVE_64BIT_INTEGER_MPI_SUPPORT
#define C_INT_MPI_TYPE_PTR long int*
#define C_INT_MPI_TYPE long int
#define MPI_KIND c_int64_t
#else
#define C_INT_MPI_TYPE_PTR int*
#define C_INT_MPI_TYPE int
#define MPI_KIND c_int
#endif

// it seems, that we need those two levels of indirection to correctly expand macros
#define cannons_reduction_impl_expand2(SUFFIX) cannons_reduction_##SUFFIX
#define cannons_reduction_impl_expand1(SUFFIX) cannons_reduction_impl_expand2(SUFFIX)
#define cannons_reduction_impl cannons_reduction_impl_expand1(ELPA_IMPL_SUFFIX)

#define cannons_reduction_c_impl_expand2(SUFFIX) cannons_reduction_c_##SUFFIX
#define cannons_reduction_c_impl_expand1(SUFFIX) cannons_reduction_c_impl_expand2(SUFFIX)
#define cannons_reduction_c_impl cannons_reduction_c_impl_expand1(ELPA_IMPL_SUFFIX)

#include "../general/precision_typedefs.h"

#include "../helpers/lapack_interfaces.h"
#include "../helpers/scalapack_interfaces.h"

void cannons_reduction_impl(math_type* A, math_type* U, 
                            C_INT_TYPE np_rows, C_INT_TYPE np_cols, C_INT_TYPE my_prow, C_INT_TYPE my_pcol,
                            C_INT_TYPE_PTR a_desc, math_type *Res, C_INT_MPI_TYPE ToStore, 
                            MPI_Comm row_comm, MPI_Comm col_comm,
                            int wantDebug, int useGPU, intptr_t *gpublasHandle)
{
   // Input matrices: 
      // - A: full matrix
      // - U: upper triangular matrix U^(-1)
   // Output matrix: 
      // - Res = U^(-H)*A*U^(-1), where U^(-H) := (U^(-1))^H
   // ToStore = cannon_buffer_size. Increasing the buffer size might make it faster, but costs memory. By default cannon_buffer_size=0
            // GPU port supports only ToStore=0
   // row_comm: communicator along rows
   // col_comm: communicator along columns

   C_INT_TYPE na, nblk, i, j, Size_send_A, Size_receive_A, Size_send_U, Size_receive_U, Buf_rows, Buf_cols, 
              pcol_where_to_send_A, pcol_from_where_to_receive_A, where_to_send_U, from_where_to_receive_U,
              last_proc_row, last_proc_col, cols_in_buffer_A, rows_in_buffer_A, intNumber;
   C_INT_TYPE ratio, num_of_iters, cols_in_buffer, rows_in_block, rows_in_buffer, curr_col_loc, cols_in_block, 
              curr_col_glob, curr_row_loc, Size_receive_A_now, Nb, owner, cols_in_buffer_A_now;
   C_INT_MPI_TYPE Size_receive_A_nowMPI, Size_receive_AMPI, Size_receive_UMPI;

   math_type *Buf_to_send_A, *Buf_to_receive_A, *Buf_to_send_U, *Buf_to_receive_U, *data_ptr, *Buf_A, 
             *Buf_pos, *U_local_start, *Res_ptr, *M, *M_T, *A_local_start, *U_local_start_curr, *U_stored,
             *CopyTo, *CopyFrom, *U_to_calc;

   C_INT_TYPE row_of_origin_U, rows_in_block_U, num_of_blocks_in_U_buffer, k, startPos, cols_in_buffer_U,
              rows_in_buffer_U, col_of_origin_A, curr_row_loc_res, curr_row_loc_A, curr_col_glob_res; 
   C_INT_TYPE curr_col_loc_res, curr_col_loc_buf, proc_row_curr, curr_col_loc_U, A_local_index, 
              LDA_A, LDA_A_new, index_row_A_for_LDA, ii, rows_in_block_U_curr, width, row_origin_U, 
              rows_in_block_A, cols_in_buffer_A_my_initial, rows_in_buffer_A_my_initial, proc_col_min;
   C_INT_TYPE *SizesU;
   C_INT_TYPE Size_U_skewed, Size_U_stored, Curr_pos_in_U_stored, rows_in_buffer_A_now;
   math_type dOne = 1.0;
   math_type dZero = 0.0;
   C_INT_TYPE one = 1;
   C_INT_TYPE zero = 0;
   C_INT_TYPE na_rows, na_cols;
         
   MPI_Status status;
   MPI_Request request_A_Recv;
   MPI_Request request_A_Send;
   MPI_Request request_U_Recv;
   MPI_Request request_U_Send;

   na = a_desc[2];
   nblk = a_desc[4];
   na_rows = numroc_(&na, &nblk, &my_prow, &zero, &np_rows);
   na_cols = numroc_(&na, &nblk, &my_pcol, &zero, &np_cols); 
   
   //if(ToStore > (np_rows -1))
   //   if((my_prow == 0)&&(my_pcol == 0))
   //      printf("Buffering level is larger than (np_rows-1) !!!\n");
   //if((my_prow == 0)&&(my_pcol == 0))
   //      printf("Buffering level = %d\n", ToStore); 
   
//////////////////////////////////////////// Start of algorithm //////////////////////////////////////////////////////////////////////////////
   if (np_cols%np_rows != 0)
   {
      //if((my_prow == 0)&& (my_pcol ==0))
      //   printf("!!!!! np_cols must be a multiple of np_rows!!!!! I do nothing! \n");
      return;
   }
  
   if (np_cols < np_rows != 0)
   {
       //if((my_prow == 0)&& (my_pcol ==0))
       //   printf("np_cols < np_rows \n");
       return;
   }
   
   ratio = np_cols/np_rows; 
   last_proc_row = ((na-1)/nblk) % np_rows;          // processor row having the last block-row of matrix
   last_proc_col = ((na-1)/nblk) % np_cols;          // processor column having the last block-column of matrix
   
   /////////////////////////memory allocation area//////////////////////////////////////////////////////////////
   if (na%nblk == 0) {
      if (my_pcol <= last_proc_col) {
         Buf_cols = na_cols;
      }
      else {
         Buf_cols = na_cols + nblk;   
      }
   }
   else {
      if (my_pcol < last_proc_col) {
         Buf_cols = na_cols;
      }
      else if (my_pcol > last_proc_col) {
         Buf_cols = na_cols + nblk;
      }
      else {  // if my_pcol == last_proc_col
         Buf_cols = na_cols + nblk - na_cols%nblk;  
      }
   }
  
   if (na%nblk == 0) {
      if (my_prow <= last_proc_row) {
         Buf_rows = na_rows + 1;   // Soheil: added + 1 to be able to accommodate for MPI message sizes in the case of pure blocked configuration e.g. na=100; with 9xMPI ranks and nblk=30 or with 4xMPI ranks and nblk=45 
      }
      else {
         Buf_rows = na_rows + nblk;
      }
   }
   else {
      if (my_prow < last_proc_row) {
         Buf_rows = na_rows;
      }
      else if (my_prow > last_proc_row) {
         Buf_rows = na_rows + nblk; 
      }
      else { // if my_prow == last_proc_row
         Buf_rows = na_rows + nblk - na_rows%nblk; 
      }
   }

   intNumber = ceil((math_type)na/(math_type)(np_cols*nblk));   // max. possible number of the local block columns of U
   Size_U_stored = ratio*nblk*nblk*intNumber*(intNumber+1)/2 + 2;   // number of local elements from the upper triangular part that every proc. has (max. possible value among all the procs.)
   
   U_stored = malloc((Size_U_stored*(ToStore+1))*sizeof(math_type));
   SizesU = malloc(ToStore*sizeof(C_INT_TYPE));  // here will be stored the sizes of the buffers of U that I have stored     
   Buf_to_send_A    = malloc(ratio*Buf_cols*Buf_rows*sizeof(math_type));
   Buf_to_receive_A = malloc(ratio*Buf_cols*Buf_rows*sizeof(math_type));
   Buf_to_send_U    = malloc(Size_U_stored*sizeof(math_type));
   Buf_to_receive_U = malloc(Size_U_stored*sizeof(math_type));
   if(ratio != 1)
      Buf_A = malloc(Buf_cols*Buf_rows*sizeof(math_type));   // in this case we will receive data into initial buffer and after place block-columns to the needed positions of buffer for calculation
   M   = calloc(na_rows*na_cols, sizeof(math_type));
   M_T = malloc(na_rows*na_cols*sizeof(math_type));

   math_type *Buf_to_send_receive_A_dev;
   math_type *Buf_to_send_receive_U_dev;
   math_type *M_dev;

   math_type *U_local_start_dev, *Res_ptr_dev; // pointers for first PxGEMM
   math_type *A_local_start_dev, *U_local_start_curr_dev; // pointers for second PxGEMM

   if (useGPU){
      set_gpu_parameters(&gpuMemcpyHostToDevice, &gpuMemcpyDeviceToHost);

      gpuErrCheck( gpuMalloc((intptr_t *)&Buf_to_send_receive_A_dev   , ratio*Buf_cols*Buf_rows*sizeof(math_type)) );
      gpuErrCheck( gpuMalloc((intptr_t *)&Buf_to_send_receive_U_dev   , Size_U_stored*sizeof(math_type)) );
      gpuErrCheck( gpuMalloc((intptr_t *)&M_dev, na_rows*na_cols*sizeof(math_type)) );
      gpuErrCheck( gpuMemset((intptr_t *)M_dev, 0, na_rows*na_cols*sizeof(math_type)) );
   }

   ////////////////////////////////////////////////////////////// initial reordering of A ///////////////////////////////////////////////////////////////////////////////////////// 

   NVTX_RANGE_PUSH("initial reordering of A");

   // here we assume, that np_rows < np_cols; then I will send to the number of processors equal to <ratio> with the "leap" equal to np_rows; the same holds for receive  
   if(ratio != 1) {
#ifdef WITH_NVTX
      nvtxRangePushA("LACPY");
#endif    
      C_LACPY("A", &na_rows, &na_cols, A, &na_rows, Buf_to_send_A, &na_rows);   // copy A to Buf_to_send_A
#ifdef WITH_NVTX
      nvtxRangePop();
#endif
   }
   Size_receive_A = 0; 
   
   // receive from different processors and place in my buffer for calculation;
   for(i = 0; i < ratio; i++)
   {
      pcol_where_to_send_A = (my_pcol - my_prow - i*np_rows + np_cols)%np_cols;                
      pcol_from_where_to_receive_A = (my_pcol + my_prow + i*np_rows)%np_cols;
      
      // send and receive in the row_comm
      if(ratio != 1)   // if grid is not square
      {
         if(pcol_where_to_send_A != my_pcol)
         {  
            MPI_Sendrecv(Buf_to_send_A, (C_INT_MPI_TYPE) (na_cols*na_rows) , MPI_MATH_DATATYPE_PRECISION_C, 
                        (C_INT_MPI_TYPE) pcol_where_to_send_A        , (C_INT_MPI_TYPE) zero, 
                         Buf_A        , (C_INT_MPI_TYPE) (na_rows*Buf_cols), MPI_MATH_DATATYPE_PRECISION_C, 
                        (C_INT_MPI_TYPE) pcol_from_where_to_receive_A, (C_INT_MPI_TYPE) zero, 
                         row_comm, &status);
            MPI_Get_count(&status, MPI_MATH_DATATYPE_PRECISION_C, &Size_receive_A_nowMPI);
            Size_receive_A_now = (C_INT_TYPE) Size_receive_A_nowMPI;
            Size_receive_A_now = Size_receive_A_now/na_rows;       // how many columns of A I have received
         }
         else {
            Size_receive_A_now = na_cols;
	      }
      
         Size_receive_A = Size_receive_A + Size_receive_A_now;  // here accumulate number of columns of A that I will receive

         // now I need to copy the received block to my buffer for A
         intNumber = pcol_from_where_to_receive_A/np_rows; // how many blocks I will receive, such that I will need to put them before the just received block
         
         CopyTo = &Buf_to_receive_A[intNumber*na_rows*nblk];  // here I will start copying the received buffer
         if (pcol_where_to_send_A != my_pcol) {
            CopyFrom = Buf_A; 
	      }
         else {
            CopyFrom = A;
	      }
      
         intNumber = ceil((math_type)Size_receive_A_now/(math_type)nblk);   // how many block-columns I have received
         for(j = 0; j < intNumber; j++)
         {
            width = nblk; // width of the current block column
            if(nblk*(j+1) > Size_receive_A_now)
               width = Size_receive_A_now - nblk*j; 
            C_LACPY("A", &na_rows, &width, CopyFrom, &na_rows, CopyTo, &na_rows);
            CopyTo = CopyTo + na_rows*nblk*ratio; 
            CopyFrom = CopyFrom + na_rows*nblk; 
         }
      }

      else { // if grid is square then simply receive from one processor to a calculation buffer
         if(my_prow > 0)
         {
            C_LACPY("A", &na_rows, &na_cols, A, &na_rows, Buf_to_send_A, &na_rows);   // copy A to Buf_to_send_A
            MPI_Sendrecv(Buf_to_send_A   , (C_INT_MPI_TYPE) (na_cols*na_rows) , MPI_MATH_DATATYPE_PRECISION_C, 
                        (C_INT_MPI_TYPE) pcol_where_to_send_A        , (C_INT_MPI_TYPE) zero, 
                         Buf_to_receive_A, (C_INT_MPI_TYPE) (na_rows*Buf_cols), MPI_MATH_DATATYPE_PRECISION_C, 
                        (C_INT_MPI_TYPE) pcol_from_where_to_receive_A, (C_INT_MPI_TYPE) zero, 
                         row_comm, &status);           
            MPI_Get_count(&status, MPI_MATH_DATATYPE_PRECISION_C, &Size_receive_AMPI);
            Size_receive_A = (C_INT_TYPE) Size_receive_AMPI;
            Size_receive_A = Size_receive_A/na_rows;       // how many columns of A I have received
         }
         else
         {
            C_LACPY("A", &na_rows, &na_cols, A, &na_rows, Buf_to_receive_A, &na_rows);   // copy A to the received buffer if I do not need to send
            Size_receive_A = na_cols; 
         }
      }
   }

   NVTX_RANGE_POP(); // initial reordering of A

   ////////////////////////////////////////////////////////////// initial reordering of U //////////////////////////////////////////////////////

   NVTX_RANGE_PUSH("initial reordering of U");

   // form array to send by block-columns
   num_of_iters = ceil((math_type)na_cols/(math_type)nblk);             // number my of block-columns
   
   where_to_send_U = (my_prow - my_pcol + np_cols)%np_rows;                 // shift = my_pcol; we assume that np_cols%np_rows = 0
   from_where_to_receive_U = (my_pcol + my_prow)%np_rows;
   
   if (where_to_send_U == my_prow) {   // if I will not need to send my local part of U, then copy my local data to the "received" buffer
      Buf_pos = Buf_to_receive_U;
   }
   else {
      Buf_pos = Buf_to_send_U;         // else form the array to send
   }
   // find the first local block belonging to the upper part of matrix U
   if (my_pcol >= my_prow) {  // if I am in the upper part of proc. grid
      curr_col_loc = 0;    // my first local block-column has block from the upper part of matrix
   }
   else  {
      curr_col_loc = 1;   //ceil((math_type)(((math_type)my_prow - (math_type)my_pcol)/(math_type)np_cols)) always will give 1 since np_cols > np_rows 
   }   
   num_of_iters = num_of_iters - curr_col_loc;   // I will exclude the first <curr_col_loc> block-columns since they do not have blocks from the upper part of matrix U
   curr_col_loc = curr_col_loc*nblk;             // local index of the found block-column

   if (my_pcol >= my_prow ) {
      rows_in_block = ceil(((math_type)(my_pcol + 1) - (math_type)my_prow)/(math_type)np_rows)*nblk;
   }
   else {
      rows_in_block = ratio*nblk;
   }
   Size_send_U = 0; 
   for(i = 0; i < num_of_iters; i++)       // loop over my block-columns, which have blocks in the upper part of U
   {      
      if (rows_in_block > na_rows) {
         rows_in_block = na_rows; 
      }
      if ((na_cols - curr_col_loc) < nblk) {
         cols_in_block = na_cols - curr_col_loc;     // how many columns do I have in the current block-column
      }
      else {
         cols_in_block = nblk; 
      }
      if ((rows_in_block > 0)&&(cols_in_block > 0))
      {
         data_ptr = &U[curr_col_loc*na_rows];   // pointer to start of the current block-column to be copied to buffer
         C_LACPY("A", &rows_in_block, &cols_in_block, data_ptr, &na_rows, Buf_pos, &rows_in_block);     // copy upper part of block-column in the buffer with LDA = length of the upper part of block-column 
         Buf_pos = Buf_pos + rows_in_block*cols_in_block;                         // go to the position where the next block-column will be copied                                             
         Size_send_U = Size_send_U + rows_in_block*cols_in_block; 
      }
      curr_col_loc = curr_col_loc + nblk;      // go to the next local block-column of my local array U 
      rows_in_block = rows_in_block + ratio*nblk;
   }
   rows_in_buffer = rows_in_block - ratio*nblk;    // remove redundant addition from the previous loop 
   *Buf_pos = (math_type)rows_in_buffer; // write number of the rows at the end of the buffer; we will need this for further multiplications on the other processors
   Size_send_U = Size_send_U + 1;
   
   //send and receive
   if (where_to_send_U != my_prow)
   {   
      // send and receive in the col_comm
      MPI_Sendrecv(Buf_to_send_U   , (C_INT_MPI_TYPE) Size_send_U       , MPI_MATH_DATATYPE_PRECISION_C, 
                  (C_INT_MPI_TYPE) where_to_send_U        , (C_INT_MPI_TYPE) zero,
                   Buf_to_receive_U, (C_INT_MPI_TYPE) (Buf_rows*na_cols), MPI_MATH_DATATYPE_PRECISION_C,
                  (C_INT_MPI_TYPE) from_where_to_receive_U, (C_INT_MPI_TYPE) zero,
                  col_comm, &status);
      MPI_Get_count(&status, MPI_MATH_DATATYPE_PRECISION_C, &Size_receive_UMPI); // find out how many elements I have received
      Size_receive_U = (C_INT_TYPE) Size_receive_UMPI;
   }
   else {// if I do not need to send 
      Size_receive_U = Size_send_U;         // how many elements I "have received"; the needed data I have already copied to the "receive" buffer
   }
   for(i = 0; i < Size_receive_U; i++)
      U_stored[i] = Buf_to_receive_U[i];
   Size_U_skewed = Size_receive_U; 
   Curr_pos_in_U_stored = Size_U_skewed;

   NVTX_RANGE_POP(); // initial reordering of U

  ///////////////////////////////////////////////////// main loop  for first PxGEMM M=A*U^(-1) /////////////////////////////////////////////////////

   pcol_where_to_send_A = (my_pcol - 1 + np_cols)%np_cols;
   pcol_from_where_to_receive_A = (my_pcol + 1)%np_cols;
   where_to_send_U = (my_prow - 1 + np_rows)%np_rows;
   from_where_to_receive_U = (my_prow + 1)%np_rows;
   
   NVTX_RANGE_PUSH("loop j<np_rows");
   for(j = 1; j < np_rows; j++)
   {
      // at this moment I need to send to neighbour what I have in the "received" arrays; that is why exchange pointers of the "received" and "send" arrays
      data_ptr = Buf_to_send_A; 
      Buf_to_send_A = Buf_to_receive_A; 
      Buf_to_receive_A = data_ptr; 
      
      data_ptr = Buf_to_send_U; 
      Buf_to_send_U = Buf_to_receive_U; 
      Buf_to_receive_U = data_ptr;
      
      ///// shift for A ////////////////////////////////////////////////////////////
      Size_send_A = Size_receive_A;  // number of block-columns of A and block-rows of U to send (that I have received on the previous step) 
      MPI_Isend(Buf_to_send_A, (C_INT_MPI_TYPE) (Size_send_A*na_rows), MPI_MATH_DATATYPE_PRECISION_C, 
               (C_INT_MPI_TYPE) pcol_where_to_send_A, (C_INT_MPI_TYPE) zero, row_comm, &request_A_Send); 
      MPI_Irecv(Buf_to_receive_A, (C_INT_MPI_TYPE) (Buf_cols*na_rows*ratio), MPI_MATH_DATATYPE_PRECISION_C, 
               (C_INT_MPI_TYPE) pcol_from_where_to_receive_A, (C_INT_MPI_TYPE) zero, row_comm, &request_A_Recv);
         
      ///// shift for U /////////////////////////////////////////////
      Size_send_U = Size_receive_U; 
      MPI_Isend(Buf_to_send_U, (C_INT_MPI_TYPE) Size_send_U, MPI_MATH_DATATYPE_PRECISION_C, 
               (C_INT_MPI_TYPE) where_to_send_U, (C_INT_MPI_TYPE) zero, col_comm, &request_U_Send); 
      MPI_Irecv(Buf_to_receive_U, (C_INT_MPI_TYPE) (Buf_rows*na_cols), MPI_MATH_DATATYPE_PRECISION_C, 
               (C_INT_MPI_TYPE) from_where_to_receive_U, (C_INT_MPI_TYPE) zero, col_comm, &request_U_Recv); 
      
      ///// multiplication ////////////////////////////////////////////////////////////////////////////////////////////
      rows_in_buffer = (int)Buf_to_send_U[Size_receive_U-1]; // copied above: *Buf_pos = (math_type)rows_in_buffer;
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
    
      num_of_blocks_in_U_buffer = ceil(((math_type)cols_in_buffer - (math_type)curr_col_loc_buf)/(math_type)nblk); 
      
      startPos = (curr_col_loc_buf + nblk)*curr_col_loc_buf/2;

      if (useGPU){
         gpuErrCheck( gpuMemcpy((intptr_t *)Buf_to_send_receive_A_dev, (intptr_t *)Buf_to_send_A, ratio*Buf_cols*Buf_rows*sizeof(math_type), gpuMemcpyHostToDevice) );
         gpuErrCheck( gpuMemcpy((intptr_t *)Buf_to_send_receive_U_dev, (intptr_t *)Buf_to_send_U, Size_U_stored*sizeof(math_type), gpuMemcpyHostToDevice) );
      }

      if (useGPU){
         U_local_start_dev = Buf_to_send_receive_U_dev + startPos;
         Res_ptr_dev = M_dev + curr_col_loc_res*na_rows;
      }
      else {
         U_local_start = &Buf_to_send_U[startPos];
         Res_ptr = &M[curr_col_loc_res*na_rows];
      }

      NVTX_RANGE_PUSH("loop i<num_of_blocks_in_U_buffer");
      for (i = 0; i < num_of_blocks_in_U_buffer; i++)
      { 
         curr_col_glob = (curr_col_loc_res/nblk)*nblk*np_cols + my_pcol*nblk;
         proc_row_curr = (curr_col_glob/nblk)%np_rows; 
         rows_in_block_A = (curr_col_glob/(nblk*np_rows))*nblk;     // in A; not to go down beyond  the upper triangular part
         if (my_prow <= proc_row_curr) {
            rows_in_block_A = rows_in_block_A + nblk; 
	      }
         if (rows_in_block_A > na_rows) {
            rows_in_block_A = na_rows; 
         }
         if ((curr_col_loc_buf + nblk) <= cols_in_buffer) {
            cols_in_block = nblk;      // number columns in block of U which will take part in this calculation
	      }
         else {
            cols_in_block = cols_in_buffer - curr_col_loc_buf;
	      }
      
         rows_in_block_U = (curr_col_glob/(nblk*np_rows))*nblk;    // corresponds to columns in A;
         if (proc_row_curr >= row_origin_U) {
            rows_in_block_U = rows_in_block_U + nblk; 
	      }
         if (rows_in_block_U > rows_in_buffer) {
            rows_in_block_U = rows_in_buffer;
         }

         if ((rows_in_block_A > 0)&&(cols_in_block > 0)) {

            NVTX_RANGE_PUSH("GEMM_1");
            // Res_ptr = Buf_to_send_A*U_local_start + Res_ptr
            // M = Buf_to_send_A*Buf_to_send_U + M
            if (useGPU){
               gpublasXgemm(gpublasHandle, 'N', 'N', 
                            rows_in_block_A, cols_in_block, rows_in_block_U, dOne, 
                            Buf_to_send_receive_A_dev, na_rows, 
                            U_local_start_dev, rows_in_block_U, dOne, 
                            Res_ptr_dev, na_rows);
               if (wantDebug) gpuDeviceSynchronize();
            }
            else {
               C_GEMM("N", "N", &rows_in_block_A, &cols_in_block, &rows_in_block_U, &dOne, 
               Buf_to_send_A, &na_rows, U_local_start, &rows_in_block_U, &dOne, Res_ptr, &na_rows);
            }
            NVTX_RANGE_POP(); // GEMM_1
         }

         curr_col_loc_res = curr_col_loc_res + nblk;
         curr_col_loc_buf = curr_col_loc_buf + nblk;
         
         if (useGPU) {
            U_local_start_dev = U_local_start_dev + rows_in_block_U*cols_in_block;
            Res_ptr_dev = M_dev + curr_col_loc_res*na_rows;
         }
         else {
            U_local_start = U_local_start + rows_in_block_U*cols_in_block;
            Res_ptr = &M[curr_col_loc_res*na_rows];
         }
         
      }
      NVTX_RANGE_POP(); // loop i<num_of_blocks_in_U_buffer

      MPI_Wait(&request_A_Send, &status);
      MPI_Wait(&request_A_Recv, &status);

      MPI_Get_count(&status, MPI_MATH_DATATYPE_PRECISION_C, &Size_receive_AMPI); // find out how many elements I have received
      Size_receive_A = (C_INT_TYPE) Size_receive_AMPI;
      Size_receive_A = Size_receive_A / na_rows;
      
      
      MPI_Wait(&request_U_Send, &status);
      MPI_Wait(&request_U_Recv, &status);
      MPI_Get_count(&status, MPI_MATH_DATATYPE_PRECISION_C, &Size_receive_UMPI); // find out how many elements I have received  
      Size_receive_U = (C_INT_TYPE) Size_receive_UMPI; 
       //// write in the buffer for later use //////////////////////////////7
      if(j <= ToStore)
      {
         for(k = 0; k < Size_receive_U; k++)
            U_stored[Curr_pos_in_U_stored + k] = Buf_to_receive_U[k]; 
         Curr_pos_in_U_stored = Curr_pos_in_U_stored + Size_receive_U; 
         SizesU[j-1] = Size_receive_U; 
      }
   }
   NVTX_RANGE_POP(); // loop j<np_rows


   /////// do the last multiplication //////////////
   rows_in_buffer = (C_INT_TYPE)Buf_to_receive_U[Size_receive_U-1];
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
    
   num_of_blocks_in_U_buffer = ceil(((math_type)cols_in_buffer - (math_type)curr_col_loc_buf)/(math_type)nblk); 
      
   startPos = (curr_col_loc_buf + nblk)*curr_col_loc_buf/2;

   if (useGPU){
      gpuErrCheck( gpuMemcpy((intptr_t *)Buf_to_send_receive_A_dev, (intptr_t *)Buf_to_receive_A, ratio*Buf_cols*Buf_rows*sizeof(math_type), gpuMemcpyHostToDevice) );
      gpuErrCheck( gpuMemcpy((intptr_t *)Buf_to_send_receive_U_dev, (intptr_t *)Buf_to_receive_U, Size_U_stored*sizeof(math_type), gpuMemcpyHostToDevice) );
   }

   if (useGPU){
      U_local_start_dev = Buf_to_send_receive_U_dev + startPos;
      Res_ptr_dev = M_dev + curr_col_loc_res*na_rows;
   }
   else {
      U_local_start = &Buf_to_receive_U[startPos];
      Res_ptr = &M[curr_col_loc_res*na_rows];
   }

#ifdef WITH_NVTX
   nvtxRangePushA("loop-last i<num_of_blocks_in_U_buffer");
#endif
   for (i = 0; i < num_of_blocks_in_U_buffer; i++)
   { 
      curr_col_glob = (curr_col_loc_res/nblk)*nblk*np_cols + my_pcol*nblk;
      proc_row_curr = (curr_col_glob/nblk)%np_rows; 
      rows_in_block_A = (curr_col_glob/(nblk*np_rows))*nblk;     // in A; not to go down beyond  the upper triangular part
      if (my_prow <= proc_row_curr) {
         rows_in_block_A = rows_in_block_A + nblk; 
      }
      if (rows_in_block_A > na_rows) {
         rows_in_block_A = na_rows; 
      }
      if ((curr_col_loc_buf + nblk) <= cols_in_buffer) {
         cols_in_block = nblk;      // number columns in block of U which will take part in this calculation
      }
      else {
         cols_in_block = cols_in_buffer - curr_col_loc_buf; 
      }
      rows_in_block_U = (curr_col_glob/(nblk*np_rows))*nblk;    // corresponds to columns in A;
      if (proc_row_curr >= row_origin_U) {
         rows_in_block_U = rows_in_block_U + nblk; 
      }
      if (rows_in_block_U > rows_in_buffer) {
         rows_in_block_U = rows_in_buffer; 
      }
      if ((rows_in_block_A > 0)&&(cols_in_block > 0)) {
#ifdef WITH_NVTX
         nvtxRangePushA("GEMM_1_last");
#endif
         // Res_ptr = Buf_to_receive_A*U_local_start + Res_ptr
         // M = Buf_to_receive_A*Buf_to_recieve_U + M
         if (useGPU){
            gpublasXgemm(gpublasHandle, 'N', 'N', 
                          rows_in_block_A, cols_in_block, rows_in_block_U, dOne, 
                          Buf_to_send_receive_A_dev, na_rows, 
                          U_local_start_dev, rows_in_block_U, dOne, 
                          Res_ptr_dev, na_rows);
            if (wantDebug) gpuDeviceSynchronize();
         }
         else { 
            C_GEMM("N", "N", &rows_in_block_A, &cols_in_block, &rows_in_block_U, &dOne, 
            Buf_to_receive_A, &na_rows, U_local_start, &rows_in_block_U, &dOne, Res_ptr, &na_rows);
         }
#ifdef WITH_NVTX
         nvtxRangePop();
#endif
      }
      
      curr_col_loc_res = curr_col_loc_res + nblk;
      curr_col_loc_buf = curr_col_loc_buf + nblk;

      if (useGPU) {
         U_local_start_dev = U_local_start_dev + rows_in_block_U*cols_in_block;
         Res_ptr_dev = M_dev + curr_col_loc_res*na_rows;
      }
      else {
         U_local_start = U_local_start + rows_in_block_U*cols_in_block;
         Res_ptr = &M[curr_col_loc_res*na_rows];
      }

   }
#ifdef WITH_NVTX
   nvtxRangePop(); // loop-last i<num_of_blocks_in_U_buffer
#endif

   if (useGPU) gpuErrCheck( gpuMemcpy((intptr_t *)M, (intptr_t *)M_dev, na_rows*na_cols*sizeof(math_type), gpuMemcpyDeviceToHost) );
   
   ///////////////////// Now M has an upper part of A*U^(-1) ///////////////////////////////////////////////
#ifdef WITH_NVTX
   nvtxRangePushA("PTRAN");
#endif 
   C_PTRAN(&na, &na, &dOne, M, &one, &one, a_desc, &dZero, M_T, &one, &one, a_desc); // M_T <- M, now M_T has lower part of U^(-H)*A 
#ifdef WITH_NVTX
   nvtxRangePop();
#endif 

   ////////////////////////////////////////////////// start algorithm to find lower part of U^(-H)*A*U^(-1) //////////////////////////
           
   /////////////////////////////////////////////////////////////// initial reordering of A ////////////////////////////////////////////////
#ifdef WITH_NVTX
      nvtxRangePushA("initial reordering of A");
#endif 

   // here we assume, that np_rows < np_cols; then I will send to the number of processors equal to <ratio> with the "leap" equal to np_rows; the same holds for receive  
   if ((ratio != 1)||(my_prow != 0)) {   // if grid is rectangular or my_prow is not 0
      Buf_pos = Buf_to_send_A;     // I will copy to the send buffer
   }
   else {
      Buf_pos = Buf_to_receive_A;  // if grid is square and my_prow is 0, then I will copy to the received buffer
   }
   // form array to send by block-columns; we need only lower triangular part
   num_of_iters = ceil((math_type)na_cols/(math_type)nblk);             // number my of block-columns
   
   cols_in_buffer_A_my_initial = 0;
   Size_send_A = 0; 
   
   if (my_pcol <= my_prow)  // if I am from the lower part of grid
   {
      curr_row_loc = 0;     // I will copy all my block-rows
      rows_in_buffer_A_my_initial = na_rows;
   }
   else
   {
      curr_row_loc = ceil((math_type)(((math_type)my_pcol - (math_type)my_prow)/(math_type)np_rows))*nblk; // I will skip some of my block-rows
      rows_in_buffer_A_my_initial = na_rows - curr_row_loc;   
   }
       
   for(i = 0; i < num_of_iters; i++)       // loop over my block-columns
   {
      curr_col_loc = i*nblk;      // local index of start of the current block-column 
      rows_in_block = na_rows - curr_row_loc;    // how many rows do I have in the lower part of the current block-column
      
      if ((na_cols - curr_col_loc) < nblk) {
         cols_in_block = na_cols - curr_col_loc;     // how many columns do I have in the block-column
      }
      else {
         cols_in_block = nblk; 
      }
      if ((rows_in_block > 0)&&(cols_in_block > 0))
      {
         A_local_start = &M_T[curr_col_loc*na_rows + curr_row_loc];
         C_LACPY("A", &rows_in_block, &cols_in_block, A_local_start, &na_rows, Buf_pos, &rows_in_block);     // copy lower part of block-column in the buffer with LDA = length of the lower part of block-column 
         Buf_pos = Buf_pos + rows_in_block*cols_in_block;
         Size_send_A = Size_send_A + rows_in_block*cols_in_block; 
         cols_in_buffer_A_my_initial = cols_in_buffer_A_my_initial + cols_in_block; 
      }
      curr_row_loc = curr_row_loc + ratio*nblk;
   }
   *Buf_pos = (math_type)cols_in_buffer_A_my_initial; // write number of the columns at the end of the buffer; we will need this for furhter multiplications on the other processors
   Size_send_A = Size_send_A + 1;
   
   // now we have the local buffer to send
   // find the lowest processor column among those who will send me
   proc_col_min = np_cols; 
   for(i = 0; i < ratio; i++)
   {
      pcol_from_where_to_receive_A = (my_pcol + my_prow + i*np_rows)%np_cols;
      if(pcol_from_where_to_receive_A < proc_col_min)
         proc_col_min = pcol_from_where_to_receive_A;
   }
   // do communications and form local buffers for calculations
   Size_receive_A = 0;       // size of the accumulated buffer
   cols_in_buffer_A = 0;     // number of columns in the accumulated buffer
   rows_in_buffer_A = 0;     // number of rows in the accumulated buffer
   for(i = 0; i < ratio; i++)
   {
      pcol_where_to_send_A = (my_pcol - my_prow - i*np_rows + np_cols)%np_cols;                
      pcol_from_where_to_receive_A = (my_pcol + my_prow + i*np_rows)%np_cols;
      
      // send and receive in the row_comm
      if(ratio != 1)   // if grid is not square
      {
         if(pcol_where_to_send_A != my_pcol)   // if I need to send and receive on this step
         {
            MPI_Sendrecv(Buf_to_send_A, (C_INT_MPI_TYPE) Size_send_A  , MPI_MATH_DATATYPE_PRECISION_C, 
                        (C_INT_MPI_TYPE) pcol_where_to_send_A         , (C_INT_MPI_TYPE) zero, 
                         Buf_A        , (C_INT_MPI_TYPE) Size_U_stored, MPI_MATH_DATATYPE_PRECISION_C, 
                        (C_INT_MPI_TYPE) pcol_from_where_to_receive_A , (C_INT_MPI_TYPE) zero, 
                         row_comm, &status);

            MPI_Get_count(&status, MPI_MATH_DATATYPE_PRECISION_C, &Size_receive_A_nowMPI);
            Size_receive_A_now = (C_INT_TYPE) Size_receive_A_nowMPI;

            Size_receive_A = Size_receive_A + Size_receive_A_now - 1; // we need only number of elements, so exclude information about cols_in_buffer_A

            cols_in_buffer_A_now = Buf_A[Size_receive_A_now-1];
            cols_in_buffer_A = cols_in_buffer_A + cols_in_buffer_A_now; 
            
            // determine number of rows in the received buffer
            if(pcol_from_where_to_receive_A <= my_prow)  // if source is from the lower part of grid
            {
               rows_in_buffer_A_now = na_rows;
            }
            else
            {
               rows_in_buffer_A_now = na_rows - ceil((math_type)(((math_type)pcol_from_where_to_receive_A - (math_type)my_prow)/(math_type)np_rows))*nblk; // some of the block-rows have been skipped
            }
            if(rows_in_buffer_A < rows_in_buffer_A_now)
               rows_in_buffer_A = rows_in_buffer_A_now; 

            intNumber = pcol_from_where_to_receive_A/np_rows; // how many processors will send me blocks, such that they will be placed before the current blocks  
            if (proc_col_min <= my_prow) {   // if among procs who will send me there is one with the full sets of block-rows in the lower part
               CopyTo = &Buf_to_receive_A[nblk*(na_rows*intNumber - nblk*(intNumber-1)*intNumber/2)];  // here I will copy to; formula based on arithm. progression
	    }
            else {
               CopyTo = &Buf_to_receive_A[nblk*(na_rows*intNumber - nblk*intNumber*(intNumber+1)/2)];  // otherwise, the first block-column will be shorter by one block
	    }
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
            if (proc_col_min <= my_prow) {   // if among procs who will send me there is one with the full sets of block-rows in the lower part
               CopyTo = &Buf_to_receive_A[nblk*(na_rows*intNumber - nblk*(intNumber-1)*intNumber/2)];  // here I will copy to; formula based on arithm. progression
	    }
            else {
               CopyTo = &Buf_to_receive_A[nblk*(na_rows*intNumber - nblk*intNumber*(intNumber+1)/2)];  // otherwise, the first block-column will be shorter by one block
	    }
            CopyFrom = Buf_to_send_A;  

            Size_receive_A = Size_receive_A + Size_send_A - 1;
         }
            
         // copy by block-columns
         intNumber = ceil((math_type)cols_in_buffer_A_now/(math_type)nblk);  // how many block-columns I have received on this iteration
         rows_in_block = rows_in_buffer_A_now; 
         for(j = 0; j < intNumber; j++)
         {
            if ((j+1)*nblk < cols_in_buffer_A_now) {
               cols_in_block = nblk; 
	    }
            else {
               cols_in_block = cols_in_buffer_A_now - j*nblk;
	    }   
            C_LACPY("A", &rows_in_block, &cols_in_block, CopyFrom, &rows_in_block, CopyTo, &rows_in_block);

            CopyFrom = CopyFrom + rows_in_block*cols_in_block; 
            CopyTo = CopyTo + nblk*(ratio*rows_in_block - nblk*(ratio-1)*ratio/2);  // I need to leave place for ratio block-columns of the other procs. of the lengths rows_in_block, (rows_in_block-nblk), (rows_in_block-2*nblk) and so on
            rows_in_block = rows_in_block - ratio*nblk;     // number of rows in the next block-columns
         }
      }
      else    // if grid is square
      {
         if(my_prow > 0)
         {  
            MPI_Sendrecv(Buf_to_send_A   , (C_INT_MPI_TYPE) Size_send_A  , MPI_MATH_DATATYPE_PRECISION_C, 
                        (C_INT_MPI_TYPE) pcol_where_to_send_A        , (C_INT_MPI_TYPE) zero, 
                         Buf_to_receive_A, (C_INT_MPI_TYPE) Size_U_stored, MPI_MATH_DATATYPE_PRECISION_C, 
                        (C_INT_MPI_TYPE) pcol_from_where_to_receive_A, (C_INT_MPI_TYPE) zero, 
                         row_comm, &status);

            MPI_Get_count(&status, MPI_MATH_DATATYPE_PRECISION_C, &Size_receive_AMPI);

            Size_receive_A = (C_INT_TYPE) Size_receive_AMPI;

            cols_in_buffer_A = (C_INT_TYPE)Buf_to_receive_A[Size_receive_A-1];
            if(pcol_from_where_to_receive_A <= my_prow)  // if source is from the lower part of grid
            {
               rows_in_buffer_A = na_rows;
            }
            else
            {
               rows_in_buffer_A = na_rows - ceil((math_type)(((math_type)pcol_from_where_to_receive_A - (math_type)my_prow)/(math_type)np_rows))*nblk; // some of the block-rows have been skipped
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

#ifdef WITH_NVTX
      nvtxRangePop(); // initial reordering of A
#endif

   ////////////////////////////////////////////////////////////// initial reordering of U: restore skewed U from the first multiplication ///////////////////////////
   
   Size_receive_U = Size_U_skewed;
   U_to_calc = U_stored;
   
   ///////////////////////////////////////////////////// main loop for second PxGEMM Res = U^(-H)*M = U^(-H)*A*U^(-1) /////////////////////////////////////////////////////
   
   pcol_where_to_send_A = (my_pcol - 1 + np_cols)%np_cols;
   pcol_from_where_to_receive_A = (my_pcol + 1)%np_cols;
   where_to_send_U = (my_prow - 1 + np_rows)%np_rows;
   from_where_to_receive_U = (my_prow + 1)%np_rows;
   Curr_pos_in_U_stored = Size_U_skewed;
   
   if (useGPU) {
      gpuErrCheck( gpuMemset((intptr_t *)M_dev, 0, na_rows*na_cols*sizeof(math_type)) ); // Reset the buffer
   }

#ifdef WITH_NVTX
   nvtxRangePushA("loop j<np_rows");
#endif
   for(j = 1; j < np_rows; j++)
   {
      // at this moment I need to send to neighbour what I have in the "received" arrays; that is why exchange pointers of the "received" and "send" arrays
      data_ptr = Buf_to_send_A; 
      Buf_to_send_A = Buf_to_receive_A; 
      Buf_to_receive_A = data_ptr; 
      
      if (j > ToStore)
      {
         data_ptr = Buf_to_send_U; 
         Buf_to_send_U = Buf_to_receive_U; 
         Buf_to_receive_U = data_ptr;
      }
        
      ///// shift for A ////////////////////////////////////////////////////////////
      Size_send_A = Size_receive_A; 
      MPI_Isend(Buf_to_send_A, (C_INT_MPI_TYPE) Size_send_A, MPI_MATH_DATATYPE_PRECISION_C, (C_INT_MPI_TYPE) pcol_where_to_send_A, (C_INT_MPI_TYPE) zero, row_comm, &request_A_Send); 
      MPI_Irecv(Buf_to_receive_A, (C_INT_MPI_TYPE) (ratio*Size_U_stored), MPI_MATH_DATATYPE_PRECISION_C, (C_INT_MPI_TYPE) pcol_from_where_to_receive_A, (C_INT_MPI_TYPE) zero, row_comm, &request_A_Recv);
         
      ///// shift for U /////////////////////////////////////////////
      Size_send_U = Size_receive_U; 
      if (j > ToStore)
      {
         if(j > ToStore + 1)
         {
            MPI_Isend(Buf_to_send_U, (C_INT_MPI_TYPE) Size_send_U, MPI_MATH_DATATYPE_PRECISION_C, (C_INT_MPI_TYPE) where_to_send_U, (C_INT_MPI_TYPE) zero, col_comm, &request_U_Send);
            U_to_calc = Buf_to_send_U;
         }
         else {
	         MPI_Isend(U_to_calc, (C_INT_MPI_TYPE) Size_send_U, MPI_MATH_DATATYPE_PRECISION_C, (C_INT_MPI_TYPE) where_to_send_U, (C_INT_MPI_TYPE) zero, col_comm, &request_U_Send);
	      }
         MPI_Irecv(Buf_to_receive_U, (C_INT_MPI_TYPE) Size_U_stored, MPI_MATH_DATATYPE_PRECISION_C, (C_INT_MPI_TYPE) from_where_to_receive_U, (C_INT_MPI_TYPE) zero, col_comm, &request_U_Recv);	 
      }
      
      ///// multiplication ////////////////////////////////////////////////////////////////////////////////////////////
      rows_in_buffer_U = (C_INT_TYPE)U_to_calc[Size_receive_U-1];
      row_of_origin_U = (my_pcol + my_prow + np_cols + j - 1)%np_rows;
      if (my_pcol >= row_of_origin_U) {
         cols_in_buffer_U = na_cols;
      }
      else {
         cols_in_buffer_U = na_cols - nblk;
      }
      cols_in_buffer_A = (C_INT_TYPE)Buf_to_send_A[Size_receive_A-2];
      rows_in_buffer_A = (C_INT_TYPE)Buf_to_send_A[Size_receive_A-1];
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
      if (my_pcol >= row_of_origin_U) {   // if origin of U is from the upper part 
         curr_col_loc_res = 0;          // then I update all columns of Result    
      }
      else {
         curr_col_loc_res = nblk;       // the first block column of U corresponds to my second one and I do not need to update the first block-column
      }
      num_of_blocks_in_U_buffer = ceil((math_type)((math_type)cols_in_buffer_U/(math_type)nblk)); 
      if (my_pcol >= row_of_origin_U) {    // if origin of U is from the upper part
         rows_in_block_U = ceil(((math_type)(my_pcol + 1) - (math_type)row_of_origin_U)/(math_type)np_rows)*nblk;  // blocks in the first block-column of U buffer
      }
      else {
         rows_in_block_U = ratio*nblk;
      }

      if (useGPU){
         gpuErrCheck( gpuMemcpy((intptr_t *)Buf_to_send_receive_A_dev, (intptr_t *)Buf_to_send_A, ratio*Buf_cols*Buf_rows*sizeof(math_type), gpuMemcpyHostToDevice) );
         gpuErrCheck( gpuMemcpy((intptr_t *)Buf_to_send_receive_U_dev, (intptr_t *)U_to_calc, Size_U_stored*sizeof(math_type), gpuMemcpyHostToDevice) );
      }

      if (useGPU){
         U_local_start_dev = Buf_to_send_receive_U_dev;
      }
      else {
         U_local_start = U_to_calc;
      }

      NVTX_RANGE_PUSH("loop i<num_of_blocks_in_U_buffer"); 
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
      
         if ((curr_col_loc_U + nblk) <= cols_in_buffer_U) {
            cols_in_block = nblk;      // number columns in block of U which will take part in this calculation
	      }
         else {
            cols_in_block = cols_in_buffer_U - curr_col_loc_U; 
         }
         if (rows_in_block_U > rows_in_buffer_U) {
            rows_in_block_U = rows_in_buffer_U;     // rows in current column of U; also a leading dimension for U
         }
         A_local_index = curr_row_loc_A;

         if (useGPU){
            A_local_start_dev = Buf_to_send_receive_A_dev + A_local_index;
            Res_ptr_dev = M_dev + curr_col_loc_res*na_rows + curr_row_loc_res; // we reuse M_dev buffer instead of introducing Res_dev
         }
         else {
            A_local_start = &Buf_to_send_A[A_local_index];
            Res_ptr = &Res[curr_col_loc_res*na_rows + curr_row_loc_res];
         }

         LDA_A = rows_in_buffer_A;
         LDA_A_new = LDA_A;
         if ((rows_in_block > 0)&&(cols_in_block > 0))
         {
            if (useGPU){
               U_local_start_curr_dev = U_local_start_dev;
            }
            else {
               U_local_start_curr = U_local_start; 
            }

            // loop over block-columns of the "active" part of L buffer
            for (ii = 0; ii < ceil((math_type)rows_in_block_U/(math_type)nblk); ii++)
            {
               if ((ii+1)*nblk <= cols_in_buffer_A) {
                  rows_in_block_U_curr = nblk; 
	            }
               else {
                  rows_in_block_U_curr = cols_in_buffer_A - ii*nblk;  
               }

               NVTX_RANGE_PUSH("GEMM_2");
               // Res_ptr = A_local_start*U_local_start_curr + Res_ptr
               // Res = Buf_to_send_A*Buf_to_send_U + Res
               if (useGPU){
               gpublasXgemm(gpublasHandle, 'N', 'N', 
                            rows_in_block, cols_in_block, rows_in_block_U_curr, dOne, 
                            A_local_start_dev, LDA_A, 
                            U_local_start_curr_dev, rows_in_block_U, dOne, 
                            Res_ptr_dev, na_rows);
               if (wantDebug) gpuDeviceSynchronize();
               }
               else {
                  C_GEMM("N", "N", &rows_in_block, &cols_in_block, &rows_in_block_U_curr, &dOne, 
                          A_local_start, &LDA_A, U_local_start_curr, &rows_in_block_U, &dOne, Res_ptr, &na_rows);
               }
               NVTX_RANGE_POP();

               LDA_A_new = LDA_A_new - nblk;
               A_local_index = A_local_index - LDA_A + LDA_A*nblk + LDA_A_new;

               if (useGPU){
                  A_local_start_dev = Buf_to_send_receive_A_dev + A_local_index;
                  U_local_start_curr_dev = U_local_start_curr_dev + rows_in_block_U_curr;
               }
               else {
                  A_local_start = &Buf_to_send_A[A_local_index];
                  U_local_start_curr = U_local_start_curr + rows_in_block_U_curr; 
               }

               LDA_A = LDA_A_new; 
            }
         }
      
         if (useGPU){
            U_local_start_dev = U_local_start_dev + rows_in_block_U*cols_in_block;
         }
         else {
            U_local_start = U_local_start + rows_in_block_U*cols_in_block;
         }

         curr_col_loc_res = curr_col_loc_res + nblk; 
         rows_in_block_U = rows_in_block_U + ratio*nblk;
      }
      NVTX_RANGE_POP(); // loop i<num_of_blocks_in_U_buffer

      MPI_Wait(&request_A_Send, &status);
      MPI_Wait(&request_A_Recv, &status);
      MPI_Get_count(&status, MPI_MATH_DATATYPE_PRECISION_C, &Size_receive_AMPI); // find out how many elements I have received 
      Size_receive_A = (C_INT_TYPE) Size_receive_AMPI;
      
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
	      MPI_Get_count(&status, MPI_MATH_DATATYPE_PRECISION_C, &Size_receive_UMPI); // find out how many elements I have received 
         Size_receive_U = (C_INT_TYPE) Size_receive_UMPI;
      }
   }
#ifdef WITH_NVTX
   nvtxRangePop(); // loop j<np_rows"
#endif


   /////// do the last multiplication //////////////
   if(ToStore < np_rows - 1)
      U_to_calc = Buf_to_receive_U;
   rows_in_buffer_U = (C_INT_TYPE)U_to_calc[Size_receive_U-1];
   row_of_origin_U = (my_pcol + my_prow + np_cols + j - 1)%np_rows;     
   if (my_pcol >= row_of_origin_U) {
      cols_in_buffer_U = na_cols;
   }
   else {
      cols_in_buffer_U = na_cols - nblk;
   }
   cols_in_buffer_A = (C_INT_TYPE)Buf_to_receive_A[Size_receive_A-2];
   rows_in_buffer_A = (C_INT_TYPE)Buf_to_receive_A[Size_receive_A-1];
   // find the minimal pcol among those who have sent A for this iteration
   col_of_origin_A = np_cols; 
   for(i = 0; i < ratio; i++)
   {
      intNumber = (my_pcol + my_prow + i*np_rows + np_cols + np_rows - 1)%np_cols;
      if(intNumber < col_of_origin_A)
         col_of_origin_A = intNumber;
   }
   
   // find block-column of the result to start update with
   if (my_pcol >= row_of_origin_U) {  // if origin of U is from the upper part 
      curr_col_loc_res = 0;          // then I update all columns of Result    
   }
   else {
      curr_col_loc_res = nblk;       // the first block column of U corresponds to my second one and I do not need to update the first block-column
   }
   num_of_blocks_in_U_buffer = ceil((math_type)((math_type)cols_in_buffer_U/(math_type)nblk));
   if (my_pcol >= row_of_origin_U) {    // if origin of U is from the upper part
      rows_in_block_U = ceil(((math_type)(my_pcol + 1) - (math_type)row_of_origin_U)/(math_type)np_rows)*nblk;  // blocks in the first block-column of U buffer
   }
   else {
      rows_in_block_U = ratio*nblk;
   }

   if (useGPU){
      gpuErrCheck( gpuMemcpy((intptr_t *)Buf_to_send_receive_A_dev, (intptr_t *)Buf_to_receive_A, ratio*Buf_cols*Buf_rows*sizeof(math_type), gpuMemcpyHostToDevice) );
      gpuErrCheck( gpuMemcpy((intptr_t *)Buf_to_send_receive_U_dev, (intptr_t *)U_to_calc, Size_U_stored*sizeof(math_type), gpuMemcpyHostToDevice) );
   }

   if (useGPU){
      U_local_start_dev = Buf_to_send_receive_U_dev;
   }
   else {
      U_local_start = U_to_calc;
   }

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
      
      if ((curr_col_loc_U + nblk) <= cols_in_buffer_U) {
         cols_in_block = nblk;      // number columns in block of U which will take part in this calculation
      }
      else {
         cols_in_block = cols_in_buffer_U - curr_col_loc_U; 
      }
      if (rows_in_block_U > rows_in_buffer_U) {
         rows_in_block_U = rows_in_buffer_U;
      }
 
      A_local_index = curr_row_loc_A;

      if (useGPU){
         A_local_start_dev = Buf_to_send_receive_A_dev + A_local_index;
         Res_ptr_dev = M_dev + curr_col_loc_res*na_rows + curr_row_loc_res; // we reuse M_dev buffer instead of introducing Res_dev
      }
      else { 
         A_local_start = &Buf_to_receive_A[A_local_index];
         Res_ptr = &Res[curr_col_loc_res*na_rows + curr_row_loc_res];
      }

      LDA_A = rows_in_buffer_A; 
      LDA_A_new = LDA_A; 
      
      if ((rows_in_block > 0) &&(cols_in_block > 0))
      {
         if (useGPU){
            U_local_start_curr_dev = U_local_start_dev;
         }
         else {
            U_local_start_curr = U_local_start;
         }

         // loop over block-columns of the "active" part of L buffer
         for (ii = 0; ii < ceil((math_type)rows_in_block_U/(math_type)nblk); ii++)
         {
            if ((ii+1)*nblk <= cols_in_buffer_A) {
               rows_in_block_U_curr = nblk; 
	         }
            else {
               rows_in_block_U_curr = cols_in_buffer_A - ii*nblk;  
            }

            NVTX_RANGE_PUSH("GEMM_2_last");
            // Res_ptr = A_local_start*U_local_start_curr + Res_ptr
            // Res = Buf_to_send_A*Buf_to_send_U + Res
            if (useGPU){
            gpublasXgemm(gpublasHandle, 'N', 'N', 
                          rows_in_block, cols_in_block, rows_in_block_U_curr, dOne, 
                          A_local_start_dev, LDA_A, 
                          U_local_start_curr_dev, rows_in_block_U, dOne, 
                          Res_ptr_dev, na_rows);
            if (wantDebug) gpuDeviceSynchronize();
            }
            else {
               C_GEMM("N", "N", &rows_in_block, &cols_in_block, &rows_in_block_U_curr, &dOne, 
                       A_local_start, &LDA_A, U_local_start_curr, &rows_in_block_U, &dOne, Res_ptr, &na_rows);
            }
            NVTX_RANGE_POP();

            LDA_A_new = LDA_A_new - nblk;
            A_local_index = A_local_index - (LDA_A - rows_in_block) + LDA_A*nblk + LDA_A_new - rows_in_block; 
            
            if (useGPU){
               A_local_start_dev = Buf_to_send_receive_A_dev + A_local_index;
               U_local_start_curr_dev = U_local_start_curr_dev + rows_in_block_U_curr;
            }
            else {
               A_local_start = &Buf_to_receive_A[A_local_index];
               U_local_start_curr = U_local_start_curr + rows_in_block_U_curr;
            }

            LDA_A = LDA_A_new;
         }
      }

      if (useGPU){
         U_local_start_dev = U_local_start_dev + rows_in_block_U*cols_in_block;
      }     
      else { 
         U_local_start = U_local_start + rows_in_block_U*cols_in_block;
      }

      curr_col_loc_res = curr_col_loc_res + nblk; 
      rows_in_block_U = rows_in_block_U + ratio*nblk;
   }

   if (useGPU) gpuErrCheck( gpuMemcpy((intptr_t *)Res, (intptr_t *)M_dev, na_rows*na_cols*sizeof(math_type), gpuMemcpyDeviceToHost) );

#ifdef WITH_NVTX
   nvtxRangePushA("PTRAN");
#endif 
   C_PTRAN(&na, &na, &dOne, Res, &one, &one, a_desc, &dZero, M, &one, &one, a_desc); // M <- Res^T  (or M <- Res^H)
#ifdef WITH_NVTX
   nvtxRangePop();
#endif

#ifdef WITH_NVTX
   nvtxRangePushA("PLACPY");
#endif 
   C_PLACPY("U", &na, &na, M, &one, &one, a_desc, Res, &one, &one, a_desc); // Res <- M
#ifdef WITH_NVTX
   nvtxRangePop();
#endif

   if (useGPU){
      gpuErrCheck( gpuFree((intptr_t *)Buf_to_send_receive_A_dev) );
      gpuErrCheck( gpuFree((intptr_t *)Buf_to_send_receive_U_dev) );
      gpuErrCheck( gpuFree((intptr_t *)M_dev) );
   }

   free(Buf_to_send_A);
   free(Buf_to_receive_A);
   free(Buf_to_send_U);
   free(Buf_to_receive_U);
   free(M); 
   free(M_T);
   if(ratio != 1)
      free(Buf_A);
   free(U_stored);
   free(SizesU);

}

void cannons_reduction_c_impl(math_type* A, math_type* U, int local_rowsCast, int local_colsCast,
                              C_INT_TYPE_PTR a_desc, math_type *Res, C_INT_MPI_TYPE ToStore,
                              C_INT_MPI_TYPE row_comm, C_INT_MPI_TYPE col_comm,
                              int wantDebug, int useGPU, intptr_t *gpublasHandle)
{
   C_INT_TYPE local_rows, local_cols;
   local_rows = (C_INT_TYPE) local_rowsCast;
   local_cols = (C_INT_TYPE) local_colsCast;

   MPI_Comm c_row_comm = MPI_Comm_f2c(row_comm);
   MPI_Comm c_col_comm = MPI_Comm_f2c(col_comm);


   C_INT_TYPE my_prow, my_pcol, np_rows, np_cols;
   C_INT_MPI_TYPE my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI;

   MPI_Comm_rank(c_row_comm, &my_prowMPI);
   MPI_Comm_size(c_row_comm, &np_rowsMPI);
   MPI_Comm_rank(c_col_comm, &my_pcolMPI);
   MPI_Comm_size(c_col_comm, &np_colsMPI);

   my_prow = (C_INT_TYPE) my_prowMPI;
   my_pcol = (C_INT_TYPE) my_pcolMPI;
   np_rows = (C_INT_TYPE) np_rowsMPI;
   np_cols = (C_INT_TYPE) np_colsMPI;

   // BEWARE
   // in the cannons algorithm, column and row communicators are exchanged
   // What we usually call row_comm in elpa, is thus passed to col_comm parameter of the function and vice versa
   // (order is swapped in the following call)
   // It is a bit unfortunate, maybe it should be changed in the Cannon algorithm to comply with ELPA standard notation?
   
   // ELPA convention  : col_comm means that the data is sent in the direction, where  my_pcol changes (my_prow=const). Size of col_comm is np_cols
   // cannon convention: col_comm means that the data is sent in the direction, wheree my_pcol=const (my_prow changes). Size of col_comm is np_rows
   // Example of 2D process grid: 
   // A1 A2 A3 A4
   // B1 B2 B3 B4
   // C1 C2 C3 C4
   // In ELPA, {B1, B2, B3, B4} belong to the same col_comm, in cannon they belong to the same row_comm
   cannons_reduction_impl(A, U, np_rows, np_cols, my_prow, my_pcol, 
                           a_desc, Res, ToStore, c_col_comm, c_row_comm, wantDebug, useGPU, gpublasHandle);
}

