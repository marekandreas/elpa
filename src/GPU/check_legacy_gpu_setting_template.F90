    ! check legacy GPU setings
    if (obj%is_set("gpu") == 1) then
      write(error_unit,*) "You still use the deprecated option 'gpu', consider switching to 'nvidia-gpu'"
      call obj%get("gpu", gpu_old, error)     
      if (error .ne. ELPA_OK) then
        write(error_unit,*) "GPU_SOLVER : Problem getting option for gpu. Aborting..."
#if GPU_SOLVER == ELPA2
#include "../elpa2/elpa2_aborting_template.F90"
#endif
#if GPU_SOLVER == ELPA1
#include "../elpa1/elpa1_aborting_template.F90"
#endif
      endif
      if (obj%is_set("nvidia-gpu") == 0) then
        ! set gpu and nvidia-gpu consistent
        call obj%set("nvidia-gpu", gpu_old, error)
        if (error .ne. ELPA_OK) then
          write(error_unit,*) "GPU_SOLVER : Problem setting option for nvidia-gpu. Aborting..."
#if GPU_SOLVER == ELPA2
#include "../elpa2/elpa2_aborting_template.F90"
#endif
#if GPU_SOLVER == ELPA1
#include "../elpa1/elpa1_aborting_template.F90"
#endif
        endif
        write(error_unit,*) "Will set the new keyword 'nvidia-gpu' now."
        write(error_unit,*) "And ignore the 'gpu' keyword from now on"
      else ! obj%is_set("nvidia-gpu") == 0
        call obj%get("nvidia-gpu", gpu_new, error)
        if (error .ne. ELPA_OK) then
          write(error_unit,*) "GPU_SOLVER : Problem getting option for nvidia-gpu. Aborting..."
#if GPU_SOLVER == ELPA2
#include "../elpa2/elpa2_aborting_template.F90"
#endif
#if GPU_SOLVER == ELPA1
#include "../elpa1/elpa1_aborting_template.F90"
#endif
        endif
        if (gpu_old .ne. gpu_new) then
                write(error_unit,*) "Please do not set 'gpu' but 'nvidia-gpu' instead"
          write(error_unit,*) "You cannot set gpu = ",gpu_old," and nvidia-gpu=",gpu_new,". Aborting..."
#if GPU_SOLVER == ELPA2
#include "../elpa2/elpa2_aborting_template.F90"
#endif
#if GPU_SOLVER == ELPA1
#include "../elpa1/elpa1_aborting_template.F90"
#endif
        endif
      endif ! obj%is_set("nvidia-gpu") == 0
      if (obj%is_set("amd-gpu") == 0) then
        ! amd-gpu is not set, but gpu is set
        ! this is ok in anycase
      else ! obj%is_set("amd-gpu") == 0
        call obj%get("amd-gpu", gpu_new, error)
        if (error .ne. ELPA_OK) then
          write(error_unit,*) "GPU_SOLVER : Problem getting option for amd-gpu. Aborting..."
#if GPU_SOLVER == ELPA2
#include "../elpa2/elpa2_aborting_template.F90"
#endif
#if GPU_SOLVER == ELPA1
#include "../elpa1/elpa1_aborting_template.F90"
#endif
        endif
        ! this is ok, if gpu == 0 and amd-gpu == 1 or
        !             if gpu == 0 and amd-gpu == 0 or
        !             if gpu == 1 and amd-gpu == 0
        if (gpu_old .eq. 1 .and. gpu_new .eq. 1) then
          write(error_unit,*) "GPU_SOLVER : You cannot set gpu = 1 and amd-gpu = 1. Aborting..."
#if GPU_SOLVER == ELPA2
#include "../elpa2/elpa2_aborting_template.F90"
#endif
#if GPU_SOLVER == ELPA1
#include "../elpa1/elpa1_aborting_template.F90"
#endif
        endif
      endif ! amd-gpu
      if (obj%is_set("intel-gpu") == 0) then
        ! intel-gpu is not set, but gpu is set
        ! this is ok in anycase
      else ! obj%is_set("intel-gpu") == 0
        call obj%get("intel-gpu", gpu_new, error)
        if (error .ne. ELPA_OK) then
          write(error_unit,*) "GPU_SOLVER : Problem getting option for intel-gpu. Aborting..."
#if GPU_SOLVER == ELPA2
#include "../elpa2/elpa2_aborting_template.F90"
#endif
#if GPU_SOLVER == ELPA1
#include "../elpa1/elpa1_aborting_template.F90"
#endif
        endif
        ! this is ok, if gpu == 0 and intel-gpu == 1 or
        !             if gpu == 0 and intel-gpu == 0 or
        !             if gpu == 1 and intel-gpu == 0
        if (gpu_old .eq. 1 .and. gpu_new .eq. 1) then
          write(error_unit,*) "GPU_SOLVER : You cannot set gpu = 1 and intel-gpu = 1. Aborting..."
#if GPU_SOLVER == ELPA2
#include "../elpa2/elpa2_aborting_template.F90"
#endif
#if GPU_SOLVER == ELPA1
#include "../elpa1/elpa1_aborting_template.F90"
#endif
        endif
      endif ! intel-gpu
    else ! gpu not set
      ! nothing to do since the legacy option is not set
    endif ! gpu is set

    ! GPU settings
    if (gpu_vendor() == NVIDIA_GPU) then
      call obj%get("nvidia-gpu", gpu, error)
      if (error .ne. ELPA_OK) then
        write(error_unit,*) "GPU_SOLVER : Problem getting option for NVIDIA GPU. Aborting..."
#if GPU_SOLVER == ELPA2
#include "../elpa2/elpa2_aborting_template.F90"
#endif
#if GPU_SOLVER == ELPA1
#include "../elpa1/elpa1_aborting_template.F90"
#endif
      endif
    else if (gpu_vendor() == AMD_GPU) then
      call obj%get("amd-gpu", gpu, error)
      if (error .ne. ELPA_OK) then
        write(error_unit,*) "GPU_SOLVER : Problem getting option for AMD GPU. Aborting..."
#if GPU_SOLVER == ELPA2
#include "../elpa2/elpa2_aborting_template.F90"
#endif
#if GPU_SOLVER == ELPA1
#include "../elpa1/elpa1_aborting_template.F90"
#endif
      endif
    else if (gpu_vendor() == INTEL_GPU) then
      call obj%get("intel-gpu", gpu, error)
      if (error .ne. ELPA_OK) then
        write(error_unit,*) "GPU_SOLVER : Problem getting option for INTEL GPU. Aborting..."
#if GPU_SOLVER == ELPA2
#include "../elpa2/elpa2_aborting_template.F90"
#endif
#if GPU_SOLVER == ELPA1
#include "../elpa1/elpa1_aborting_template.F90"
#endif
      endif
    else if (gpu_vendor() == OPENMP_OFFLOAD_GPU) then
      call obj%get("intel-gpu", gpu, error)
      if (error .ne. ELPA_OK) then
        write(error_unit,*) "GPU_SOLVER : Problem getting option for INTEL GPU. Aborting..."
#if GPU_SOLVER == ELPA2
#include "../elpa2/elpa2_aborting_template.F90"
#endif
#if GPU_SOLVER == ELPA1
#include "../elpa1/elpa1_aborting_template.F90"
#endif
      endif
    else ! no supported gpu
      gpu = 0
    endif

   if (gpu .eq. 1) then
     useGPU =.true.
   else
#ifdef DEVICE_POINTER
     write(error_unit,*) "GPU_SOLVER : You used the interface for device pointers but did not specify GPU usage!. Aborting..."
#if GPU_SOLVER == ELPA2
#include "../elpa2/elpa2_aborting_template.F90"
#endif
#if GPU_SOLVER == ELPA1
#include "../elpa1/elpa1_aborting_template.F90"
#endif
#endif
     useGPU = .false.
   endif


