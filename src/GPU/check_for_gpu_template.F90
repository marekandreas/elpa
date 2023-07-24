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

      gpuAvailable = .false.

      if (OBJECT%gpu_setup%gpuIsAssigned) then
        gpuAvailable = .true.
        return
      endif

#ifdef ADDITIONAL_OBJECT_CODE
      if (.not.(present(wantDebug))) then
        wantDebugMessage = .false.
      else
        if (wantDebug) then
          wantDebugMessage=.true.
        else
          wantDebugMessage=.false.
        endif
      endif
#else
      wantDebugMessage = .false.
#endif


      call OBJECT%get("mpi_comm_parent",mpi_comm_all, error)
      if (error .ne. ELPA_OK) then
        write(error_unit,*) "Problem getting option for mpi_comm_parent. Aborting..."
        stop 1
      endif

      ! needed later for handle creation
#ifdef WITH_OPENMP_TRADITIONAL
      maxThreads=omp_get_max_threads()
#else /* WITH_OPENMP_TRADITIONAL */
      maxThreads=1
#endif /* WITH_OPENMP_TRADITIONAL */

#ifdef WITH_NVIDIA_GPU_VERSION
      if (.not.(allocated(OBJECT%gpu_setup%cublasHandleArray))) then
        allocate(OBJECT%gpu_setup%cublasHandleArray(0:maxThreads-1))
        allocate(OBJECT%gpu_setup%gpublasHandleArray(0:maxThreads-1))
        do thread=0, maxThreads-1
          OBJECT%gpu_setup%cublasHandleArray(thread) = -1
          OBJECT%gpu_setup%gpublasHandleArray(thread) = -1
        enddo
      endif
#endif
#ifdef WITH_NVIDIA_GPU_VERSION
#ifdef WITH_NVIDIA_CUSOLVER
      if (.not.(allocated(OBJECT%gpu_setup%cusolverHandleArray))) then
        allocate(OBJECT%gpu_setup%cusolverHandleArray(0:maxThreads-1))
        allocate(OBJECT%gpu_setup%gpusolverHandleArray(0:maxThreads-1))
        do thread=0, maxThreads-1
          OBJECT%gpu_setup%cusolverHandleArray(thread) = -1
          OBJECT%gpu_setup%gpusolverHandleArray(thread) = -1
        enddo
      endif
#endif
#endif
#ifdef WITH_AMD_GPU_VERSION
      if (.not.(allocated(OBJECT%gpu_setup%rocblasHandleArray))) then
        allocate(OBJECT%gpu_setup%rocblasHandleArray(0:maxThreads-1))
        allocate(OBJECT%gpu_setup%gpublasHandleArray(0:maxThreads-1))
        do thread=0, maxThreads-1
          OBJECT%gpu_setup%rocblasHandleArray(thread) = -1
          OBJECT%gpu_setup%gpublasHandleArray(thread) = -1
        enddo
      endif
#endif
#ifdef WITH_AMD_GPU_VERSION
#ifdef WITH_AMD_ROCSOLVER
      if (.not.(allocated(OBJECT%gpu_setup%rocsolverHandleArray))) then
        allocate(OBJECT%gpu_setup%rocsolverHandleArray(0:maxThreads-1))
        allocate(OBJECT%gpu_setup%gpusolverHandleArray(0:maxThreads-1))
        do thread=0, maxThreads-1
          OBJECT%gpu_setup%rocsolverHandleArray(thread) = -1
          OBJECT%gpu_setup%gpusolverHandleArray(thread) = -1
        enddo
      endif
#endif
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (.not.(allocated(OBJECT%gpu_setup%openmpOffloadHandleArray))) then
        allocate(OBJECT%gpu_setup%openmpOffloadHandleArray(0:maxThreads-1))
        allocate(OBJECT%gpu_setup%gpublasHandleArray(0:maxThreads-1))
        do thread=0, maxThreads-1
          OBJECT%gpu_setup%openmpOffloadHandleArray(thread) = -1
          OBJECT%gpu_setup%gpublasHandleArray(thread) = -1
        enddo
      endif
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      if (.not.(allocated(OBJECT%gpu_setup%openmpOffloadsolverHandleArray))) then
        allocate(OBJECT%gpu_setup%openmpOffloadsolverHandleArray(0:maxThreads-1))
        allocate(OBJECT%gpu_setup%gpusolverHandleArray(0:maxThreads-1))
        do thread=0, maxThreads-1
          OBJECT%gpu_setup%openmpOffloadsolverHandleArray(thread) = -1
          OBJECT%gpu_setup%gpusolverHandleArray(thread) = -1
        enddo
      endif
#endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (.not.(allocated(OBJECT%gpu_setup%syclHandleArray))) then
        allocate(OBJECT%gpu_setup%syclHandleArray(0:maxThreads-1))
        allocate(OBJECT%gpu_setup%gpublasHandleArray(0:maxThreads-1))
        do thread=0, maxThreads-1
          OBJECT%gpu_setup%syclHandleArray(thread) = -1
          OBJECT%gpu_setup%gpublasHandleArray(thread) = -1
        enddo
      endif
#endif
#ifdef WITH_SYCL_GPU_VERSION
#ifdef WITH_SYCL_SOLVER
      if (.not.(allocated(OBJECT%gpu_setup%syclsolverHandleArray))) then
        allocate(OBJECT%gpu_setup%syclsolverHandleArray(0:maxThreads-1))
        do thread=0, maxThreads-1
          OBJECT%gpu_setup%syclsolverHandleArray(thread) = -1
        enddo
      endif
#endif
#endif


      if (OBJECT%is_set("use_gpu_id") == 1) then ! useGPUid
        call OBJECT%get("use_gpu_id", use_gpu_id, error)
        if (error .ne. ELPA_OK) then
          write(error_unit,*) "check_for_gpu: cannot querry use_gpu_id. Aborting..."
          stop 1
        endif

        if (use_gpu_id == -99) then
          write(error_unit,*) "Problem you did not set which gpu id this task should use"
        endif
 
        ! check whether gpu ud has been set for each proces
#ifdef WITH_MPI
        call mpi_allreduce(use_gpu_id, min_use_gpu_id, 1, MPI_INTEGER, MPI_MAX, mpi_comm_all, mpierr)

        if (min_use_gpu_id .lt. 0) then
          write(error_unit,*) "Not all tasks have set which GPU id should be used! GPU usage switched off!"
          gpuAvailable = .false.
          return
        endif
#endif
        gpuAvailable = .true.

        !if (myid==0) then
        !  if (wantDebugMessage) then
        !    print *
        !    print '(3(a,i0))','Found ', numberOfDevices, ' GPUs'
        !  endif
        !endif

        success = .true.
        if (.not.(OBJECT%gpu_setup%gpuAlreadySet)) then
          deviceNumber = use_gpu_id
#include "./device_arrays_template.F90"

#include "./handle_creation_template.F90"

        endif ! alreadySET
        OBJECT%gpu_setup%gpuAlreadySET = .true.
        OBJECT%gpu_setup%gpuIsAssigned =.true.
        !gpuIsInitialized = .true.

      else ! useGPUid

        ! make sure GPU setup is only done once (per ELPA OBJECTect)

        if (.not.(OBJECT%gpu_setup%gpuAlreadySet)) then

          !TODO: have to set this somewhere
          !if (gpuIsInitialized) then
          !  gpuAvailable = .true.
          !  numberOfDevices = -1
          !  if (myid == 0 .and. wantDebugMessage) then
          !    write(error_unit,*)  "Skipping GPU init, should have already been initialized "
          !  endif
          !  return
          !else
            if (myid == 0 .and. wantDebugMessage) then
              write(error_unit,*) "Initializing the GPU devices"
            endif
          !endif


          success = .true.
#ifndef WITH_SYCL_GPU_VERSION
          success = gpu_getdevicecount(numberOfDevices)
#endif
#ifdef WITH_SYCL_GPU_VERSION
          call OBJECT%get("sycl_show_all_devices", syclShowAllDevices, error)
          if (error .ne. ELPA_OK) then
            write(error_unit,*) "Problem getting option for sycl_show_all_devices. Aborting..."
            stop 1
          endif

          if (syclShowAllDevices == 1) then
            syclShowOnlyIntelGpus = 0
            OBJECT%gpu_setup%syclCPU = 1
          else
            syclShowOnlyIntelGpus = 1
            OBJECT%gpu_setup%syclCPU = 0
          endif
          if (myid == 0 .and. wantDebugMessage) then
            write(error_unit,*) "SYCL: syclShowOnlyIntelGpus =  ", syclShowOnlyIntelGpus
          endif
          success = sycl_getdevicecount(numberOfDevices, syclShowOnlyIntelGpus)
          if (myid == 0 .and. wantDebugMessage) then
            write(error_unit,*) "SYCL: numberOfDevices =  ", numberOfDevices
          endif
          if (wantDebugMessage) then
            call sycl_printdevices()
          endif
          !OBJECT%gpu_setup%syclCPU=.false.
          !success = sycl_getdevicecount(numberOfDevices)
#endif
          if (.not.(success)) then
#ifdef WITH_NVIDIA_GPU_VERSION
            write(error_unit,*) "error in cuda_getdevicecount"
#endif
#ifdef WITH_AMD_GPU_VERSION
            write(error_unit,*) "error in hip_getdevicecount"
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
            write(error_unit,*) "error in openmp_offload_getdevicecount"
#endif
#ifdef WITH_SYCL_GPU_VERSION
            write(error_unit,*) "error in sycl_getdevicecount"
#endif
            stop 1
          endif

!#ifdef  WITH_INTEL_GPU_VERSION
!        gpuAvailable = .false.
!        numberOfDevices = -1
!
!        numberOfDevices = 1
!        write(error_unit,*) "Manually setting",numberOfDevices," of GPUs"
!        if (numberOfDevices .ge. 1) then
!          gpuAvailable = .true.
!        endif
!#endif


#ifdef WITH_SYCL_GPU_VERSION
        ! special case: maybe we want to run the sycl code path on cpu ?
        if (numberOfDevices .eq. 0) then
          success = sycl_getcpucount(numberOfDevices)
          if (.not.(success)) then
#ifdef WITH_SYCL_GPU_VERSION
            write(error_unit,*) "error in sycl_getdevicecount"
#endif
            stop 1
          endif
          if (numberOfDevices .ge. 0) then
            OBJECT%gpu_setup%syclCPU=.true.
          endif
        endif
#endif

          ! make sure that all nodes have the same number of GPU's, otherwise
          ! we run into loadbalancing trouble
#ifdef WITH_MPI
          call mpi_allreduce(numberOfDevices, maxNumberOfDevices, 1, MPI_INTEGER, MPI_MAX, mpi_comm_all, mpierr)

          if (maxNumberOfDevices .ne. numberOfDevices) then
            write(error_unit,*) "Different number of GPU devices on MPI tasks!"
            write(error_unit,*) "GPUs will NOT be used!"
            gpuAvailable = .false.
            return
          endif
#endif
          if (numberOfDevices .ne. 0) then
            gpuAvailable = .true.
            ! Usage of GPU is possible since devices have been detected

            if (myid==0) then
              if (wantDebugMessage) then
                print *
                print '(3(a,i0))','Found ', numberOfDevices, ' GPUs'
              endif
            endif

            fmt = '(I5.5)'

            write (gpu_string,fmt) numberOfDevices

#ifdef ADDITIONAL_OBJECT_CODE
            call OBJECT%timer%start("check_gpu_"//gpu_string)
#endif
            deviceNumber = mod(myid, numberOfDevices)
#ifdef ADDITIONAL_OBJECT_CODE
            call OBJECT%timer%start("set_device")
#endif

            !include device arrays here
#include "./device_arrays_template.F90"

#ifdef ADDITIONAL_OBJECT_CODE
            call OBJECT%timer%stop("set_device")
#endif

            call OBJECT%set("use_gpu_id",deviceNumber, error)
            if (error .ne. ELPA_OK) then
              write(error_unit,*) "Cannot set use_gpu_id. Aborting..."
              stop 1
            endif
 
#include "./handle_creation_template.F90"
          
            OBJECT%gpu_setup%gpuAlreadySet = .true.


#ifdef WITH_NVIDIA_NCCL
            write(error_unit,*) "Setting up nccl"


            ! mpi_comm_all
            if (myid .eq. 0) then
              success = nccl_get_unique_id(ncclId) 
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

            success = nccl_group_start()
            if (.not.success) then
              write(error_unit,*) "Error in setting up nccl_group_start!"
              stop 1
            endif

            success = nccl_comm_init_rank(ccl_comm_all, nprocs, ncclId, myid)
            if (.not.success) then
              write(error_unit,*) "Error in setting up communicator nccl_comm_all id!"
              stop 1
            endif

            success = nccl_group_end()
            if (.not.success) then
              write(error_unit,*) "Error in setting up nccl_group_end!"
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
              success = nccl_get_unique_id(ncclId) 
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

            success = nccl_group_start()
            if (.not.success) then
              write(error_unit,*) "Error in setting up nccl_group_start!"
              stop 1
            endif

            success = nccl_comm_init_rank(ccl_comm_rows, nprows, ncclId, myid_rows)
            if (.not.success) then
              write(error_unit,*) "Error in setting up communicator nccl_comm_rows id!"
              stop 1
            endif

            success = nccl_group_end()
            if (.not.success) then
              write(error_unit,*) "Error in setting up nccl_group_end!"
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
              success = nccl_get_unique_id(ncclId) 
              if (.not.success) then
                write(error_unit,*) "Error in setting up unique nccl id for cols!"
                stop 1
              endif
            endif
            call mpi_comm_size(mpi_comm_cols, npcols, mpierr)
            call MPI_Bcast(ncclId, 128, MPI_BYTE, 0, mpi_comm_cols, mpierr)
            if (mpierr .ne. MPI_SUCCESS) then
              write(error_unit,*) "Error when sending unique id for cols"
              stop 1
            endif

            success = nccl_group_start()
            if (.not.success) then
              write(error_unit,*) "Error in setting up nccl_group_start!"
              stop 1
            endif

            success = nccl_comm_init_rank(ccl_comm_cols, npcols, ncclId, myid_cols)
            if (.not.success) then
              write(error_unit,*) "Error in setting up communicator nccl_comm_cols id!"
              stop 1
            endif

            success = nccl_group_end()
            if (.not.success) then
              write(error_unit,*) "Error in setting up nccl_group_end!"
              stop 1
            endif

            OBJECT%gpu_setup%ccl_comm_cols = ccl_comm_cols





            !success = nccl_comm_destroy(ccl_comm_all)
            !if (.not.success) then
            !  write(error_unit,*) "Error in destroying ccl_comm_all!"
            !  stop 1
            !endif
#endif
#ifdef ADDITIONAL_OBJECT_CODE
            call OBJECT%timer%stop("check_gpu_"//gpu_string)
#endif
          endif ! numberOfDevices .ne. 0
          !gpuIsInitialized = .true.
        endif !OBJECT%gpu_setup%gpuAlreadySet
        OBJECT%gpu_setup%gpuIsAssigned = .true.
      endif ! useGPUid

