#ifdef WITH_OPENMP_TRADITIONAL
    ! store the number of OpenMP threads used in the calling function
    ! restore this at the end of the solver step
    omp_threads_caller = omp_get_max_threads()

   ! check the number of threads that ELPA should use internally
#if defined(THREADING_SUPPORT_CHECK) && defined(ALLOW_THREAD_LIMITING) && !defined(HAVE_SUFFICIENT_MPI_THREADING_SUPPORT)
   call obj%get("limit_openmp_threads",limitThreads,error)
   if (limitThreads .eq. 0) then
#endif
     if (obj%is_set("omp_threads") == 1) then
       ! user set omp_threads, honour this
       call obj%get("omp_threads", nrThreads, error)
       if (error .ne. ELPA_OK) then
         print *,"cannot get option for omp_threads. Aborting..."
         stop 1
       endif
       call omp_set_num_threads(nrThreads)
     else
       ! use the max threads
       call obj%set("omp_threads",omp_threads_caller, error)
       if (error .ne. ELPA_OK) then
         print *,"cannot set option for omp_threads. Aborting..."
         stop 1
       endif
       nrThreads = omp_threads_caller
       call omp_set_num_threads(omp_threads_caller)
     endif
#if defined(THREADING_SUPPORT_CHECK) && defined(ALLOW_THREAD_LIMITING) && !defined(HAVE_SUFFICIENT_MPI_THREADING_SUPPORT)
   else
     nrThreads = 1
     call omp_set_num_threads(nrThreads)
   endif
#endif

#else /* WITH_OPENMP_TRADITIONAL */
    nrThreads = 1
#endif /* WITH_OPENMP_TRADITIONAL */

