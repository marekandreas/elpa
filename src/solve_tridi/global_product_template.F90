subroutine global_product_&
&PRECISION&
&(obj, z, n, mpi_comm_rows, mpi_comm_cols, npc_0, npc_n)
  ! This routine calculates the global product of z.
  use precision
  use elpa_abstract_impl
  use elpa_mpi
#ifdef WITH_OPENMP_TRADITIONAL
  !use elpa_omp
#endif          
  implicit none
  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik), intent(in)               :: mpi_comm_cols, mpi_comm_rows
  integer(kind=ik), intent(in)               :: npc_0, npc_n
#ifdef WITH_MPI
  integer(kind=MPI_KIND)                     :: mpierr,my_pcolMPI, np_colsMPI, np_rowsMPI
#endif
  integer(kind=ik)                           :: n, my_pcol, np_cols, np_rows
  real(kind=REAL_DATATYPE)                   :: z(n)

  real(kind=REAL_DATATYPE)                   :: tmp(n)
  integer(kind=ik)                           :: np

#ifdef WITH_MPI
  call obj%timer%start("mpi_communication")
  call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND) ,np_rowsMPI, mpierr)
  np_rows = int(np_rowsMPI,kind=c_int)

  call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND) ,my_pcolMPI, mpierr)
  my_pcol = int(my_pcolMPI,kind=c_int)
  call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND) ,np_colsMPI, mpierr)
  np_cols = int(np_colsMPI,kind=c_int)

  !!my_pcol = int(my_pcolMPI,kind=c_int)
  call obj%timer%stop("mpi_communication")
#endif

  if (npc_n==1 .and. np_rows==1) return ! nothing to do

  ! Do an mpi_allreduce over processor rows
#ifdef WITH_MPI
  call obj%timer%start("mpi_communication")


  call mpi_allreduce(z, tmp, int(n,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_PROD, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
  call obj%timer%stop("mpi_communication")
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
    call obj%timer%start("mpi_communication")
    call mpi_allreduce(tmp, z, int(n,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_PROD, int(mpi_comm_cols,kind=MPI_KIND), mpierr)
    call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
    z = tmp
#endif /* WITH_MPI */
    return
  endif

  ! We send all vectors to the first proc, do the product there
  ! and redistribute the result.

  if (my_pcol == npc_0) then
    z(1:n) = tmp(1:n)
    do np = npc_0+1, npc_0+npc_n-1
#ifdef WITH_MPI
      call obj%timer%start("mpi_communication")
      call mpi_recv(tmp, int(n,kind=MPI_KIND), MPI_REAL_PRECISION, int(np,kind=MPI_KIND), 1117_MPI_KIND, &
                    int(mpi_comm_cols,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
      call obj%timer%stop("mpi_communication")
#else  /* WITH_MPI */
      tmp(1:n) = z(1:n)
#endif  /* WITH_MPI */
      z(1:n) = z(1:n)*tmp(1:n)
    enddo
    do np = npc_0+1, npc_0+npc_n-1
#ifdef WITH_MPI
      call obj%timer%start("mpi_communication")
      call mpi_send(z, int(n,kind=MPI_KIND), MPI_REAL_PRECISION, int(np,kind=MPI_KIND), 1118_MPI_KIND, &
                    int(mpi_comm_cols,kind=MPI_KIND), mpierr)
      call obj%timer%stop("mpi_communication")
#else
#endif  /* WITH_MPI */
    enddo
  else
#ifdef WITH_MPI
    call obj%timer%start("mpi_communication")
    call mpi_send(tmp, int(n,kind=MPI_KIND), MPI_REAL_PRECISION, int(npc_0,kind=MPI_KIND), 1117_MPI_KIND, &
                  int(mpi_comm_cols,kind=MPI_KIND), mpierr)
    call mpi_recv(z, int(n,kind=MPI_KIND), MPI_REAL_PRECISION, int(npc_0,kind=MPI_KIND), 1118_MPI_KIND, &
                  int(mpi_comm_cols,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
    call obj%timer%stop("mpi_communication")
#else  /* WITH_MPI */
    z(1:n) = tmp(1:n)
#endif  /* WITH_MPI */

  endif
end subroutine global_product_&
        &PRECISION
