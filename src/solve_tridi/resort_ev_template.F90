#include "../general/error_checking.inc"

subroutine resort_ev_&
&PRECISION&
  &(obj, idx_ev, nLength, na, p_col_out, q, ldq, matrixCols, l_rows, l_rqe, l_rqs, &
    mpi_comm_cols, p_col, l_col, l_col_out)
    use precision
#ifdef WITH_OPENMP_TRADITIONAL
    use elpa_omp
#endif
    use elpa_mpi
    use ELPA_utilities
    use elpa_abstract_impl
    implicit none
    class(elpa_abstract_impl_t), intent(inout) :: obj
    integer(kind=ik), intent(in)               :: nLength, na
    integer(kind=ik), intent(in)               :: ldq, matrixCols, l_rows, l_rqe, l_rqs
    integer(kind=ik), intent(in)               :: mpi_comm_cols
    integer(kind=ik), intent(in)               :: p_col(na), l_col(na), l_col_out(na)
#ifdef WITH_MPI
    integer(kind=MPI_KIND)                     :: mpierrMPI, my_pcolMPI
    integer(kind=ik)                           :: mpierr
#endif
    integer(kind=ik)                           :: my_pcol
#ifdef USE_ASSUMED_SIZE
    real(kind=REAL_DATATYPE), intent(inout)    :: q(ldq,*)
#else
    real(kind=REAL_DATATYPE), intent(inout)    :: q(ldq,matrixCols)
#endif
    integer(kind=ik), intent(in)               :: p_col_out(na)
    integer(kind=ik)                           :: idx_ev(nLength)
    integer(kind=ik)                           :: i, nc, pc1, pc2, lc1, lc2, l_cols_out

    real(kind=REAL_DATATYPE), allocatable      :: qtmp(:,:)
    integer(kind=ik)                           :: istat
    character(200)                             :: errorMessage

    if (l_rows==0) return ! My processor column has no work to do

#ifdef WITH_MPI
    call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND) ,my_pcolMPI, mpierr)
    my_pcol = int(my_pcolMPI,kind=c_int)
#endif

    ! Resorts eigenvectors so that q_new(:,i) = q_old(:,idx_ev(i))

    l_cols_out = COUNT(p_col_out(1:na)==my_pcol)
    allocate(qtmp(l_rows,l_cols_out), stat=istat, errmsg=errorMessage)
    check_allocate("resort_ev: qtmp",istat, errorMessage)

    nc = 0

    do i=1,na

      pc1 = p_col(idx_ev(i))
      lc1 = l_col(idx_ev(i))
      pc2 = p_col_out(i)

      if (pc2<0) cycle ! This column is not needed in output

      if (pc2==my_pcol) nc = nc+1 ! Counter for output columns

      if (pc1==my_pcol) then
        if (pc2==my_pcol) then
          ! send and recieve column are local
          qtmp(1:l_rows,nc) = q(l_rqs:l_rqe,lc1)
        else
#ifdef WITH_MPI
          call obj%timer%start("mpi_communication")
          call mpi_send(q(l_rqs,lc1), int(l_rows,kind=MPI_KIND), MPI_REAL_PRECISION, pc2, int(mod(i,4096),kind=MPI_KIND), &
                              int(mpi_comm_cols,kind=MPI_KIND), mpierr)
          call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */

#endif /* WITH_MPI */
        endif
      else if (pc2==my_pcol) then
#ifdef WITH_MPI
        call obj%timer%start("mpi_communication")
        call mpi_recv(qtmp(1,nc), int(l_rows,kind=MPI_KIND), MPI_REAL_PRECISION, pc1, int(mod(i,4096),kind=MPI_KIND), &
                      int(mpi_comm_cols,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
        call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
        qtmp(1:l_rows,nc) = q(l_rqs:l_rqe,lc1)
#endif /* WITH_MPI */
       endif
     enddo

     ! Insert qtmp into (output) q

     nc = 0

     do i=1,na

       pc2 = p_col_out(i)
       lc2 = l_col_out(i)

       if (pc2==my_pcol) then
         nc = nc+1
         q(l_rqs:l_rqe,lc2) = qtmp(1:l_rows,nc)
       endif
     enddo

     deallocate(qtmp, stat=istat, errmsg=errorMessage)
     check_deallocate("resort_ev: qtmp",istat, errorMessage)
   end subroutine resort_ev_&
        &PRECISION

