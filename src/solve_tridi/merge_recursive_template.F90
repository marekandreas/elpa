
recursive subroutine merge_recursive_&
          &PRECISION &
   (obj, np_off, nprocs, ldq, matrixCols, nblk, &
   l_col, p_col, l_col_bc, p_col_bc, limits, &
   np_cols, na, q, d, e, &
   mpi_comm_all, mpi_comm_rows, mpi_comm_cols, &
   useGPU, wantDebug, success, max_threads)

   use precision
   use elpa_abstract_impl
!#ifdef WITH_OPENMP_TRADITIONAL
!   use elpa_omp
!#endif
   use elpa_mpi
   use merge_systems
   use ELPA_utilities
   implicit none

   ! noff is always a multiple of nblk_ev
   ! nlen-noff is always > nblk_ev


   class(elpa_abstract_impl_t), intent(inout) :: obj
   integer(kind=ik), intent(in)               :: max_threads
   integer(kind=ik), intent(in)               :: mpi_comm_all, mpi_comm_rows, mpi_comm_cols
   integer(kind=ik), intent(in)               :: ldq, matrixCols, nblk, na, np_cols
   integer(kind=ik), intent(in)               :: l_col_bc(na), p_col_bc(na), l_col(na), p_col(na), &
                                                 limits(0:np_cols)
#ifdef USE_ASSUMED_SIZE
   real(kind=REAL_DATATYPE), intent(inout)    :: q(ldq,*)
#else
   real(kind=REAL_DATATYPE), intent(inout)    :: q(ldq,matrixCols)
#endif
#ifdef WITH_MPI
   integer(kind=MPI_KIND)                     :: mpierr, my_pcolMPI
#endif
   integer(kind=ik)                           :: my_pcol
   real(kind=REAL_DATATYPE), intent(inout)    :: d(na), e(na)           
   integer(kind=ik)                           :: np_off, nprocs
   integer(kind=ik)                           :: np1, np2, noff, nlen, nmid, n
   logical, intent(in)                        :: useGPU, wantDebug
   logical, intent(out)                       :: success

   success = .true.

#ifdef WITH_MPI
   call obj%timer%start("mpi_communication")
   call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND) ,my_pcolMPI, mpierr)

   my_pcol = int(my_pcolMPI,kind=c_int)
   call obj%timer%stop("mpi_communication")

#endif

   if (nprocs<=1) then
     ! Safety check only
     if (wantDebug) write(error_unit,*) "ELPA1_merge_recursive: INTERNAL error merge_recursive: nprocs=",nprocs
     success = .false.
     return
   endif
   ! Split problem into 2 subproblems of size np1 / np2

   np1 = nprocs/2
   np2 = nprocs-np1

   if (np1 > 1) call merge_recursive_&
                &PRECISION &
   (obj, np_off, np1, ldq, matrixCols, nblk, &
   l_col, p_col, l_col_bc, p_col_bc, limits, &
   np_cols, na, q, d, e, &
   mpi_comm_all, mpi_comm_rows, mpi_comm_cols, &
   useGPU, wantDebug, success, max_threads)
   if (.not.(success)) then
     write(error_unit,*) "Error in merge_recursice. Aborting..."
     return
   endif

   if (np2 > 1) call merge_recursive_&
                &PRECISION &
   (obj, np_off+np1, np2, ldq, matrixCols, nblk, &
   l_col, p_col, l_col_bc, p_col_bc, limits, &
   np_cols, na, q, d, e, &
   mpi_comm_all, mpi_comm_rows, mpi_comm_cols, &
   useGPU, wantDebug, success, max_threads)
   if (.not.(success)) then
     write(error_unit,*) "Error in merge_recursice. Aborting..."
     return
   endif             

   noff = limits(np_off)
   nmid = limits(np_off+np1) - noff
   nlen = limits(np_off+nprocs) - noff

#ifdef WITH_MPI
   call obj%timer%start("mpi_communication")
   if (my_pcol==np_off) then
     do n=np_off+np1,np_off+nprocs-1
       call mpi_send(d(noff+1), int(nmid,kind=MPI_KIND), MPI_REAL_PRECISION, int(n,kind=MPI_KIND), 12_MPI_KIND, &
                     int(mpi_comm_cols,kind=MPI_KIND), mpierr)
     enddo
   endif
   call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
#endif /* WITH_MPI */

   if (my_pcol>=np_off+np1 .and. my_pcol<np_off+nprocs) then
#ifdef WITH_MPI
     call obj%timer%start("mpi_communication")
     call mpi_recv(d(noff+1), int(nmid,kind=MPI_KIND), MPI_REAL_PRECISION, int(np_off,kind=MPI_KIND), 12_MPI_KIND, &
                   int(mpi_comm_cols,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
     call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
!             d(noff+1:noff+1+nmid-1) = d(noff+1:noff+1+nmid-1)
#endif /* WITH_MPI */
   endif

   if (my_pcol==np_off+np1) then
     do n=np_off,np_off+np1-1
#ifdef WITH_MPI
       call obj%timer%start("mpi_communication")
       call mpi_send(d(noff+nmid+1), int(nlen-nmid,kind=MPI_KIND), MPI_REAL_PRECISION, int(n,kind=MPI_KIND), &
                     15_MPI_KIND, int(mpi_comm_cols,kind=MPI_KIND), mpierr)
       call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
#endif /* WITH_MPI */

     enddo
   endif

   if (my_pcol>=np_off .and. my_pcol<np_off+np1) then
#ifdef WITH_MPI
     call obj%timer%start("mpi_communication")
     call mpi_recv(d(noff+nmid+1), int(nlen-nmid,kind=MPI_KIND), MPI_REAL_PRECISION, int(np_off+np1,kind=MPI_KIND), &
                   15_MPI_KIND, int(mpi_comm_cols,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
     call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
!             d(noff+nmid+1:noff+nmid+1+nlen-nmid-1) = d(noff+nmid+1:noff+nmid+1+nlen-nmid-1)
#endif /* WITH_MPI */
   endif
   if (nprocs == np_cols) then

     ! Last merge, result distribution must be block cyclic, noff==0,
     ! p_col_bc is set so that only nev eigenvalues are calculated
     call merge_systems_&
          &PRECISION &
                         (obj, nlen, nmid, d(noff+1), e(noff+nmid), q, ldq, noff, &
                         nblk, matrixCols, int(mpi_comm_rows,kind=ik), int(mpi_comm_cols,kind=ik), &
                         l_col, p_col, &
                         l_col_bc, p_col_bc, np_off, nprocs, useGPU, wantDebug, success, max_threads)
     if (.not.(success)) then
       write(error_unit,*) "Error in merge_systems: Aborting..."
       return
     endif

   else
     ! Not last merge, leave dense column distribution
     call merge_systems_&
          &PRECISION &
                        (obj, nlen, nmid, d(noff+1), e(noff+nmid), q, ldq, noff, &
                         nblk, matrixCols, int(mpi_comm_rows,kind=ik), int(mpi_comm_cols,kind=ik), &
                         l_col(noff+1), p_col(noff+1), &
                         l_col(noff+1), p_col(noff+1), np_off, nprocs, useGPU, wantDebug, success, &
                         max_threads)
     if (.not.(success)) then
       write(error_unit,*) "Error in merge_systems: Aborting..."
       return
     endif             
   endif
end subroutine merge_recursive_&
           &PRECISION

