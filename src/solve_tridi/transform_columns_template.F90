subroutine transform_columns_&
&PRECISION&
&(obj, col1, col2, na, tmp, l_rqs, l_rqe, q, ldq, matrixCols, &
  l_rows, mpi_comm_cols, p_col, l_col, qtrans)
  use precision
  use elpa_abstract_impl
#ifdef WITH_OPENMP_TRADITIONAL
  use elpa_omp
#endif
  use elpa_mpi
  implicit none
  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik), intent(in)               :: na, l_rqs, l_rqe, ldq, matrixCols
  integer(kind=ik), intent(in)               :: l_rows, mpi_comm_cols
  integer(kind=ik), intent(in)               :: p_col(na), l_col(na)
#ifdef USE_ASSUMED_SIZE
  real(kind=REAL_DATATYPE), intent(inout)    :: q(ldq,*)
#else
  real(kind=REAL_DATATYPE), intent(inout)    :: q(ldq,matrixCols)
#endif
  real(kind=REAL_DATATYPE), intent(in)       :: qtrans(2,2)
#ifdef WITH_MPI
  integer(kind=MPI_KIND)                     :: mpierrMPI, my_pcolMPI
  integer(kind=ik)                           :: mpierr
#endif
  integer(kind=ik)                           :: my_pcol
  integer(kind=ik)                           :: col1, col2
  real(kind=REAL_DATATYPE)                   :: tmp(na)
  integer(kind=ik)                           :: pc1, pc2, lc1, lc2

  if (l_rows==0) return ! My processor column has no work to do

#ifdef WITH_MPI
  call obj%timer%start("mpi_communication")
  call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND) ,my_pcolMPI, mpierr)
  !call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND) ,np_colsMPI, mpierr)

  my_pcol = int(my_pcolMPI,kind=c_int)
  !np_cols = int(np_colsMPI,kind=c_int)

  call obj%timer%stop("mpi_communication")
#else
#endif
  pc1 = p_col(col1)
  lc1 = l_col(col1)
  pc2 = p_col(col2)
  lc2 = l_col(col2)

  if (pc1==my_pcol) then
    if (pc2==my_pcol) then
      ! both columns are local
      tmp(1:l_rows)      = q(l_rqs:l_rqe,lc1)*qtrans(1,1) + q(l_rqs:l_rqe,lc2)*qtrans(2,1)
      q(l_rqs:l_rqe,lc2) = q(l_rqs:l_rqe,lc1)*qtrans(1,2) + q(l_rqs:l_rqe,lc2)*qtrans(2,2)
      q(l_rqs:l_rqe,lc1) = tmp(1:l_rows)
    else
#ifdef WITH_MPI
      call obj%timer%start("mpi_communication")
      call mpi_sendrecv(q(l_rqs,lc1), int(l_rows,kind=MPI_KIND), MPI_REAL_PRECISION, pc2, 1_MPI_KIND, &
                        tmp, int(l_rows,kind=MPI_KIND), MPI_REAL_PRECISION, pc2, 1_MPI_KIND,          &
                        int(mpi_comm_cols,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
      call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
#endif /* WITH_MPI */
      q(l_rqs:l_rqe,lc1) = q(l_rqs:l_rqe,lc1)*qtrans(1,1) + tmp(1:l_rows)*qtrans(2,1)
    endif
  else if (pc2==my_pcol) then
#ifdef WITH_MPI
    call obj%timer%start("mpi_communication")
    call mpi_sendrecv(q(l_rqs,lc2), int(l_rows,kind=MPI_KIND), MPI_REAL_PRECISION, pc1, 1_MPI_KIND, &
                      tmp, int(l_rows,kind=MPI_KIND), MPI_REAL_PRECISION, pc1, 1_MPI_KIND,          &
                      int(mpi_comm_cols,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
    call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
    tmp(1:l_rows) = q(l_rqs:l_rqe,lc2)
#endif /* WITH_MPI */

    q(l_rqs:l_rqe,lc2) = tmp(1:l_rows)*qtrans(1,2) + q(l_rqs:l_rqe,lc2)*qtrans(2,2)
  endif
end subroutine transform_columns_&
        &PRECISION

