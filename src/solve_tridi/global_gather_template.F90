subroutine global_gather_&
&PRECISION&
&(obj, z, n, mpi_comm_rows, mpi_comm_cols, npc_n, np_prev, np_next, &
  success)
  ! This routine sums up z over all processors.
  ! It should only be used for gathering distributed results,
  ! i.e. z(i) should be nonzero on exactly 1 processor column,
  ! otherways the results may be numerically different on different columns
  use precision
  use elpa_abstract_impl
  use elpa_mpi
  use ELPA_utilities
#ifdef WITH_OPENMP_TRADITIONAL
  use elpa_omp
#endif
  implicit none
  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik), intent(in)               :: mpi_comm_cols, mpi_comm_rows
  integer(kind=ik), intent(in)               :: npc_n, np_prev, np_next
#ifdef WITH_MPI
  integer(kind=MPI_KIND)                     :: mpierr, np_rowsMPI, np_colsMPI
#endif
  integer(kind=ik)                           :: n, np_rows, np_cols
  real(kind=REAL_DATATYPE)                   :: z(n)
  real(kind=REAL_DATATYPE)                   :: tmp(n)
  integer(kind=ik)                           :: np
  integer(kind=MPI_KIND)                     :: allreduce_request1, allreduce_request2
  logical                                    :: useNonBlockingCollectivesCols
  logical                                    :: useNonBlockingCollectivesRows
  integer(kind=c_int)                        :: non_blocking_collectives_rows, error, &
                                                non_blocking_collectives_cols
  logical                                    :: success

  success = .true.

  call obj%get("nbc_row_global_gather", non_blocking_collectives_rows, error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "Problem getting option for non blocking collectives for rows in global_gather. Aborting..."
    success = .false.
    return
  endif

  call obj%get("nbc_col_global_gather", non_blocking_collectives_cols, error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "Problem getting option for non blocking collectives for cols in global_gather. Aborting..."
    success = .false.
    return
  endif

  if (non_blocking_collectives_rows .eq. 1) then
    useNonBlockingCollectivesRows = .true.
  else
    useNonBlockingCollectivesRows = .false.
  endif

  if (non_blocking_collectives_cols .eq. 1) then
    useNonBlockingCollectivesCols = .true.
  else
    useNonBlockingCollectivesCols = .false.
  endif

#ifdef WITH_MPI
  call obj%timer%start("mpi_communication")
  call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND) ,np_rowsMPI, mpierr)
  np_rows = int(np_rowsMPI,kind=c_int)

  !call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND) ,my_pcolMPI, mpierr)
  call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND) ,np_colsMPI, mpierr)
  np_cols = int(np_colsMPI,kind=c_int)

  call obj%timer%stop("mpi_communication")

#else
#endif
  if (npc_n==1 .and. np_rows==1) return ! nothing to do

  ! Do an mpi_allreduce over processor rows
#ifdef WITH_MPI
  if (useNonBlockingCollectivesRows) then
    call obj%timer%start("mpi_nbc_communication")
    call mpi_iallreduce(z, tmp, int(n,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), &
                      allreduce_request1, mpierr)
    call mpi_wait(allreduce_request1, MPI_STATUS_IGNORE, mpierr)
    call obj%timer%stop("mpi_nbc_communication")
  else
    call obj%timer%start("mpi_communication")
    call mpi_allreduce(z, tmp, int(n,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), &
                      mpierr)
    call obj%timer%stop("mpi_communication")
  endif
#else /* WITH_MPI */
  tmp = z
#endif /* WITH_MPI */
  ! If only 1 processor column, we are done
  if (npc_n==1) then
    z(:) = tmp(:)
    return
  endif

  ! If all processor columns are involved, we can use mpi_allreduce
  if (npc_n==np_cols) then
#ifdef WITH_MPI
    if (useNonBlockingCollectivesCols) then
      call obj%timer%start("mpi_nbc_communication")
      call mpi_iallreduce(tmp, z, int(n,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_SUM, int(mpi_comm_cols,kind=MPI_KIND), &
              allreduce_request2, mpierr)
      call mpi_wait(allreduce_request2, MPI_STATUS_IGNORE, mpierr)
      call obj%timer%stop("mpi_nbc_communication")
    else
      call obj%timer%start("mpi_communication")
      call mpi_allreduce(tmp, z, int(n,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_SUM, int(mpi_comm_cols,kind=MPI_KIND), &
              mpierr)
      call obj%timer%stop("mpi_communication")
    endif
#else /* WITH_MPI */
    tmp = z
#endif /* WITH_MPI */

    return
  endif

  ! Do a ring send over processor columns
  z(:) = 0
  do np = 1, npc_n
    z(:) = z(:) + tmp(:)
#ifdef WITH_MPI
    call obj%timer%start("mpi_communication")
    call mpi_sendrecv_replace(z, int(n,kind=MPI_KIND), MPI_REAL_PRECISION, int(np_next,kind=MPI_KIND), &
                              1112_MPI_KIND+int(np,kind=MPI_KIND), &
                              int(np_prev,kind=MPI_KIND), 1112_MPI_KIND+int(np,kind=MPI_KIND), &
                              int(mpi_comm_cols,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
    call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
#endif /* WITH_MPI */
  enddo
end subroutine global_gather_&
        &PRECISION
