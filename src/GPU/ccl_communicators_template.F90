#if 0
!    Copyright 2014-2023, A. Marek
!
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
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!      and
!    - IBM Deutschland GmbH
!
!
!    More information can be found here:
!    http://elpa.mpcdf.mpg.de/
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
! This file was written by A. Marek, MPCDF
#endif

#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
            ! mpi_comm_all
            if (myid .eq. 0) then
              success = ccl_get_unique_id(ncclId)
              if (.not.success) then
                write(error_unit,*) "Error in setting up unique nccl id!"
                stop 1
              endif
            endif
          
            !broadcast id currently not possible
            call mpi_comm_size(mpi_comm_all, nprocs, mpierr)
            call MPI_Bcast(ncclId, 128, MPI_BYTE, 0, mpi_comm_all, mpierr)
            if (mpierr .ne. MPI_SUCCESS) then
              write(error_unit,*) "Error when sending unique id"
              stop 1
            endif

            success = ccl_group_start()
            if (.not.success) then
              write(error_unit,*) "Error in setting up ccl_group_start!"
              stop 1
            endif

#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
            success = ccl_comm_init_rank(ccl_comm_all, nprocs, ncclId, myid)
            if (.not.success) then
              write(error_unit,*) "Error in setting up communicator ccl_comm_all id!"
              stop 1
            endif
#endif
            success = ccl_group_end()
            if (.not.success) then
              write(error_unit,*) "Error in setting up ccl_group_end!"
              stop 1
            endif

            OBJECT%gpu_setup%ccl_comm_all = ccl_comm_all


            ! mpi_comm_rows
            call OBJECT%get("mpi_comm_rows",mpi_comm_rows, error)
            if (error .ne. ELPA_OK) then
              write(error_unit,*) "Problem getting option for mpi_comm_rows. Aborting..."
              stop 1
            endif

            call mpi_comm_rank(mpi_comm_rows, myid_rows, mpierr)
            if (myid_rows .eq. 0) then
              success = ccl_get_unique_id(ncclId)
              if (.not.success) then
                write(error_unit,*) "Error in setting up unique nccl id for rows!"
                stop 1
              endif
            endif
            call mpi_comm_size(mpi_comm_rows, nprows, mpierr)
            call MPI_Bcast(ncclId, 128, MPI_BYTE, 0, mpi_comm_rows, mpierr)
            if (mpierr .ne. MPI_SUCCESS) then
              write(error_unit,*) "Error when sending unique id for rows"
              stop 1
            endif
            success = ccl_group_start()
            if (.not.success) then
              write(error_unit,*) "Error in setting up ccl_group_start!"
              stop 1
            endif

#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
            success = ccl_comm_init_rank(ccl_comm_rows, nprows, ncclId, myid_rows)
            if (.not.success) then
              write(error_unit,*) "Error in setting up communicator nccl_comm_rows id!"
              stop 1
            endif
#endif

            success = ccl_group_end()
            if (.not.success) then
              write(error_unit,*) "Error in setting up ccl_group_end!"
              stop 1
            endif

            OBJECT%gpu_setup%ccl_comm_rows = ccl_comm_rows


            ! mpi_comm_cols
            call OBJECT%get("mpi_comm_cols",mpi_comm_cols, error)
            if (error .ne. ELPA_OK) then
              write(error_unit,*) "Problem getting option for mpi_comm_cols. Aborting..."
              stop 1
            endif
            call mpi_comm_rank(mpi_comm_cols, myid_cols, mpierr)
            if (myid_cols .eq. 0) then
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
              success = ccl_get_unique_id(ncclId)
              if (.not.success) then
                write(error_unit,*) "Error in setting up unique nccl id for cols!"
                stop 1
              endif
#endif
            endif
            call mpi_comm_size(mpi_comm_cols, npcols, mpierr)
            call MPI_Bcast(ncclId, 128, MPI_BYTE, 0, mpi_comm_cols, mpierr)
            if (mpierr .ne. MPI_SUCCESS) then
              write(error_unit,*) "Error when sending unique id for cols"
              stop 1
            endif

            success = ccl_group_start()
            if (.not.success) then
              write(error_unit,*) "Error in setting up ccl_group_start!"
              stop 1
            endif

#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
            success = ccl_comm_init_rank(ccl_comm_cols, npcols, ncclId, myid_cols)
            if (.not.success) then
              write(error_unit,*) "Error in setting up communicator nccl_comm_cols id!"
              stop 1
            endif
#endif

            success = ccl_group_end()
            if (.not.success) then
              write(error_unit,*) "Error in setting up ccl_group_end!"
              stop 1
            endif
            OBJECT%gpu_setup%ccl_comm_cols = ccl_comm_cols

            !success = ccl_comm_destroy(ccl_comm_all)
            !if (.not.success) then
            !  write(error_unit,*) "Error in destroying ccl_comm_all!"
            !  stop 1
            !endif
#endif
