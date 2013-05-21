!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium, 
!    consisting of the following organizations:
!
!    - Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG), 
!    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!      Informatik,
!    - Technische Universität München, Lehrstuhl für Informatik mit
!      Schwerpunkt Wissenschaftliches Rechnen , 
!    - Fritz-Haber-Institut, Berlin, Abt. Theorie, 
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaftrn, 
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition, 
!      and  
!    - IBM Deutschland GmbH
!
!
!    More information can be found here:
!    http://elpa.rzg.mpg.de/
!
!    ELPA is free software: you can redistribute it and/or modify
!    it under the terms of the version 3 of the license of the 
!    GNU Lesser General Public License as published by the Free 
!    Software Foundation.
!
!    ELPA is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public License
!    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
!
!    ELPA reflects a substantial effort on the part of the original
!    ELPA consortium, and we ask you to respect the spirit of the
!    license that we chose: i.e., please contribute any changes you
!    may have back to the original ELPA library distribution, and keep
!    any derivatives of ELPA under the same license that we chose for
!    the original distribution, the GNU Lesser General Public License.
!
!
subroutine read_test_parameters (na, nev, nblk, myid, mpi_comm)

!-------------------------------------------------------------------------------
! Subroutine read_test_parameters allows to set the marix parameters for the 
! ELPA test programs at runtime. It is used by all the test programs which
! set up their test matrices as random matrices:
!
! - test_complex2.f90
! - test_complex.f90
! - test_complex_gen.f90
! - test_real2.f90
! - test_real.f90
! - test_real_gen.f90
!
! If an input file "test_parameters.in" is found, we scan it for any of the parameters:
!
! na   : System size (matrix size)
! nev  : Number of eigenvectors to be calculated
! nblk : Blocking factor in block cyclic distribution
!
! The following parameters are fixed and only pass on the MPI infrastructure and task information:
!
! myid : MPI task number (we only read on myid=0 and broadcast to all other)
! mpi_comm : MPI communicator ID
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
!-------------------------------------------------------------------------------

! this should be a simple read-in routine ... no use statements

  implicit none
  include 'mpif.h'

  integer :: na, nev, nblk
  integer :: myid, mpi_comm

!-------------------------------------------------------------------------------
!  Local Variables

  ! For reading operations:
  integer :: info, i_code
  character*132 :: inputline
  character*40 :: desc_str

  ! For MPI allreduce
  integer :: temp_mpi
  integer :: mpierr

!-------------------------------------------------------------------------------
! begin work

if (myid==0) then
  ! we only read on task 0

  info=0
  open (7,file='test_parameters.in',status='OLD',iostat=info)

  if (info.ne.0) then
      ! File not found or not readable. In this case, we keep the defaults.
      write(6,*) 
      write (6,'(2X,A)') "Input file 'test_parameters.in' not found. We keep the defaults for na, nev, nblk."
  else
      ! File exists, attempt to read.

      write(6,*) 
      write (6,'(2X,A)') "Attempting to read input file 'test_parameters.in' for values of na, nev, nblk."

      lineloop: do
          read(7,'(A)',iostat=i_code) inputline

          if (i_code.lt.0) then
              write (6,'(2X,A)') "| End of input file 'test_parameters.in' reached."
              exit lineloop
          else if (i_code.gt.0) then
              write (6,'(1X,A)') "* Unknown error reading next line of input file 'test_parameters.in'."
              exit lineloop
          end if 

          ! if we are here, inputline was read correctly. Next, we dissect it for its content.
          read(inputline,*,iostat=i_code) desc_str
          if (i_code /= 0) then
              cycle ! empty line
          elseif (desc_str(1:1).eq.'#') then
              cycle ! comment
          elseif (desc_str.eq."na") then
              read(inputline,*,end=88,err=99) desc_str, na
              write(6,'(2X,A,I15)') "| Found value for na    : ", na
          elseif (desc_str.eq."nev") then
              read(inputline,*,end=88,err=99) desc_str, nev
              write(6,'(2X,A,I15)') "| Found value for nev   : ", nev
          elseif (desc_str.eq."nblk") then
              read(inputline,*,end=88,err=99) desc_str, nblk
              write(6,'(2X,A,I15)') "| Found value for nblk  : ", nblk
          end if

      enddo lineloop
      close(7)

  end if ! info == 0 (input file exists)

  ! Next, check current values for consistency.

  if (na.le.0) then
       write(6,*) "* Error - Found illegal value for na: ", na
       write(6,*) "* na must be greater than zero - stopping the test run."
       ! harsh exit - but we can only get here from process number one
       call MPI_Abort(mpi_comm, 0, mpierr)
  end if

  if (nev.le.0) then
       write(6,*) "* Error - Found illegal value for nev: ", nev
       write(6,*) "* nev must be greater than zero - stopping the test run."
       ! harsh exit - but we can only get here from process number one
       call MPI_Abort(mpi_comm, 0, mpierr)
  else if (nev.gt.na) then
       write(6,*) "* Error - Found nev value that is greater than na. nev: ", nev
       nev = na
       write(6,*) "* Reducing nev to nev = na = ", nev, "."
  end if

  if (nblk.le.0) then
       write(6,*) "* Error - Found illegal value for nblk: ", nblk
       write(6,*) "* nblk must be greater than zero - stopping the test run."
       ! harsh exit - but we can only get here from process number one
       call MPI_Abort(mpi_comm, 0, mpierr)
  else if (nblk.gt.na) then
       write(6,*) "* Error - Found illegal value for nblk: ", nblk
       write(6,*) "* nblk must be (much!) less than na - stopping the test run."
       ! harsh exit - but we can only get here from process number one
       call MPI_Abort(mpi_comm, 0, mpierr)
  end if

else ! if we are not on myid=0 :
     ! for later allreduce, zero values on other processes
     na   = 0
     nev  = 0
     nblk = 0
end if
 
! Synchronize all values. Note that, if a value was not read in the input file,
! a default value was already compiled into the code.

! synchronize na
temp_mpi = 0
call MPI_ALLREDUCE(na, temp_mpi, 1, &
MPI_INTEGER, MPI_SUM, mpi_comm, mpierr)
na  = temp_mpi

! synchronize nev
temp_mpi = 0
call MPI_ALLREDUCE(nev, temp_mpi, 1, &
MPI_INTEGER, MPI_SUM, mpi_comm, mpierr)
nev  = temp_mpi

! synchronize nblk
temp_mpi = 0
call MPI_ALLREDUCE(nblk, temp_mpi, 1, &
MPI_INTEGER, MPI_SUM, mpi_comm, mpierr)
nblk  = temp_mpi

! normally the subroutine is done at this point
return

! Error traps for read statements

88 continue
    if (myid == 0) then
        write (*,*) "Syntax error reading 'test_parameters.in' (missing arguments?)."
        write (*,*) "line: '"//trim(inputline)//"'"
    end if
    ! harsh exit - but we can only get here from process number one
    call MPI_Abort(mpi_comm, 0, mpierr)

99 continue
     if (myid == 0) then
         write (*,*) "Syntax error reading 'test_parameters.in'."
         write (*,*) "line: '"//trim(inputline)//"'"
     end if
     ! harsh exit - but we can only get here from process number one
     call MPI_Abort(mpi_comm, 0, mpierr)

end subroutine read_test_parameters
