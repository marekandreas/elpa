#ifdef REDISTRIBUTE_MATRIX
   doRedistributeMatrix = .false.

   if (obj%is_set("internal_nblk") == 1) then
     if (obj%is_set("matrix_order") == 1) then
       reDistributeMatrix = .true.
     else
       reDistributeMatrix = .false.
       if (my_pe == 0) then
         write(error_unit,*) 'Warning: Matrix re-distribution is not used, since you did not set the matrix_order'
         write(error_unit,*) '         Only task 0 prints this warning'
       endif
     endif
   endif
   if (reDistributeMatrix) then
     ! get the block-size to be used
     call obj%get("internal_nblk", nblkInternal, error)
     if (error .ne. ELPA_OK) then
       print *,"Problem getting option. Aborting..."
       stop
     endif

     call obj%get("matrix_order", matrixOrder, error)
     if (error .ne. ELPA_OK) then
       print *,"Problem getting option. Aborting..."
       stop
     endif

     layoutExternal=matrixLayouts(matrixOrder)

     if (nblkInternal == nblk) then
       doRedistributeMatrix = .false.
     else
       doRedistributeMatrix = .true.
     endif
   endif

   if (doRedistributeMatrix) then
     call obj%timer%start("redistribute")

     ! collect some necessary values
     ! and np_rows and np_cols
     call obj%get("mpi_comm_rows",mpi_comm_rows,error)
     call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)
     call obj%get("mpi_comm_cols",mpi_comm_cols,error)
     call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND), np_colsMPI, mpierr)

     np_rows = int(np_rowsMPI,kind=c_int)
     np_cols = int(np_colsMPI,kind=c_int)



     ! create a new internal blacs discriptor
     ! matrix will still be distributed over the same process grid
     ! as input matrices A and Q where
     !external_blacs_ctxt = int(mpi_comm_all,kind=BLAS_KIND)
     ! construct current descriptor
     if (obj%is_set("blacs_context") == 1) then
       call obj%get("blacs_context",external_blacs_ctxt, error)
       external_blacs_ctxt_ = int(external_blacs_ctxt,kind=BLAS_KIND)
     else

       ! we have to re-create the blacs context
       external_blacs_ctxt_ = int(mpi_comm_all,kind=BLAS_KIND)
       call BLACS_Gridinit(external_blacs_ctxt_, layoutExternal, int(np_rows,kind=BLAS_KIND), int(np_cols,kind=BLAS_KIND))
     endif

      sc_desc(1) = 1
      sc_desc(2) = external_blacs_ctxt_
      sc_desc(3) = obj%na
      sc_desc(4) = obj%na
      sc_desc(5) = obj%nblk
      sc_desc(6) = obj%nblk
      sc_desc(7) = 0
      sc_desc(8) = 0
      sc_desc(9) = obj%local_nrows

     layoutInternal = 'C'

     blacs_ctxt_ = int(mpi_comm_all,kind=BLAS_KIND)

     !if (layoutInternal /= layoutExternal) then
       ! we get new blacs context ; we want to keep the np_rows, and np_cols
       np_rows_ = int(np_rows,kind=BLAS_KIND)
       np_cols_ = int(np_cols,kind=BLAS_KIND)

       ! blacs_ctxt_ inout
       ! np_rows,cols in
       call BLACS_Gridinit(blacs_ctxt_, layoutInternal, int(np_rows,kind=BLAS_KIND), int(np_cols,kind=BLAS_KIND))


       call BLACS_Gridinfo(blacs_ctxt_, np_rows_, np_cols_, my_prow_, my_pcol_)
       if (np_rows /= np_rows_) then
         print *, "BLACS_Gridinfo returned different values for np_rows as set by BLACS_Gridinit"
         print *,np_rows,np_rows_
         stop 1
       endif
       if (np_cols /= np_cols_) then
         print *, "BLACS_Gridinfo returned different values for np_cols as set by BLACS_Gridinit"
         print *,np_cols,np_cols_
         stop 1
       endif

       call mpi_comm_split(int(mpi_comm_all,kind=MPI_KIND), int(my_pcol_,kind=MPI_KIND), &
                           int(my_prow_,kind=MPI_KIND), mpi_comm_rowsMPI_, mpierr)
       mpi_comm_rows_ = int(mpi_comm_rowsMPI_,kind=c_int)

       call mpi_comm_split(int(mpi_comm_all,kind=MPI_KIND), int(my_prow_,kind=MPI_KIND), &
                           int(my_pcol_,kind=MPI_KIND), mpi_comm_colsMPI_, mpierr)
       mpi_comm_cols_ = int(mpi_comm_colsMPI_,kind=c_int)

       if (int(np_rows_,kind=c_int) /= np_rows) then
         print *, "BLACS_Gridinfo returned different values for np_rows as set by BLACS_Gridinit"
         stop
       endif
       if (int(np_cols_,kind=c_int) /= np_cols) then
         print *, "BLACS_Gridinfo returned different values for np_cols as set by BLACS_Gridinit"
         stop
       endif
     !else
     !  na_rows_ = int(na_rows,kind=BLAS_KIND)
     !  na_cols_ = int(na_cols,kind=BLAS_KIND)
     !  sc_desc_ 
     !endif
     
     ! now we can set up the the blacs descriptor

     !sc_desc_(:) = 0
     na_rows_ = numroc(int(na,kind=BLAS_KIND), int(nblkInternal,kind=BLAS_KIND), my_prow_, 0_BLAS_KIND, np_rows_)
     na_cols_ = numroc(int(na,kind=BLAS_KIND), int(nblkInternal,kind=BLAS_KIND), my_pcol_, 0_BLAS_KIND, np_cols_)

     info_ = 0
     call descinit(sc_desc_, int(na,kind=BLAS_KIND), int(na,kind=BLAS_KIND), int(nblkInternal,kind=BLAS_KIND), &
                   int(nblkInternal,kind=BLAS_KIND), 0_BLAS_KIND, 0_BLAS_KIND, &
                   blacs_ctxt_, na_rows_, info_)

     if (info_ .ne. 0) then
       write(error_unit,*) 'Error in BLACS descinit! info=',info_
       write(error_unit,*) 'the internal re-distribution of the matrices failed!'
       call MPI_ABORT(int(mpi_comm_all,kind=MPI_KIND), 1_MPI_KIND, mpierr)
     endif


     ! allocate the memory for the redistributed matrices
     allocate(aIntern(na_rows_,na_cols_))
#ifdef HAVE_SKEWSYMMETRIC
     allocate(qIntern(na_rows_,2*na_cols_))
#else
     allocate(qIntern(na_rows_,na_cols_))
#endif
     call obj%timer%start("GEMR2D")
     call scal_PRECISION_GEMR2D &
     (int(na,kind=BLAS_KIND), int(na,kind=BLAS_KIND), aExtern, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, aIntern, &
     1_BLAS_KIND, 1_BLAS_KIND, sc_desc_, blacs_ctxt_)
     call obj%timer%stop("GEMR2D")
     !call scal_PRECISION_GEMR2D &
     !(int(na,kind=BLAS_KIND), int(na,kind=BLAS_KIND), qExtern, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, qIntern, &
     !1_BLAS_KIND, 1_BLAS_KIND, sc_desc_, blacs_ctxt_)

     !map all important new variables to be used from here
     nblk                    = nblkInternal
     matrixCols              = na_cols_
     matrixRows              = na_rows_
     mpi_comm_cols           = mpi_comm_cols_
     mpi_comm_rows           = mpi_comm_rows_

     a => aIntern(1:matrixRows,1:matrixCols)
     q => qIntern(1:matrixRows,1:matrixCols)

     
     call obj%timer%stop("redistribute")
   else
     a => aExtern(1:matrixRows,1:matrixCols)
     q => qExtern(1:matrixRows,1:matrixCols)
   endif

#endif /* REDISTRIBUTE_MATRIX */

