#ifdef WITH_OPENMP_TRADITIONAL
    ! store the number of OpenMP threads used in the calling function
    ! restore this at the end of the solver step
    omp_threads_caller = omp_get_max_threads()

   ! check the number of threads that ELPA should use internally
#if defined(THREADING_SUPPORT_CHECK) && defined(ALLOW_THREAD_LIMITING) && !defined(HAVE_SUFFICIENT_MPI_THREADING_SUPPORT)
   call obj%get("limit_openmp_threads",limitThreads,error)
   if (limitThreads .eq. 0) then
#endif
     call obj%get("omp_threads",nrThreads,error)
     call omp_set_num_threads(nrThreads)
#if defined(THREADING_SUPPORT_CHECK) && defined(ALLOW_THREAD_LIMITING) && !defined(HAVE_SUFFICIENT_MPI_THREADING_SUPPORT)
   else
     nrThreads = 1
     call omp_set_num_threads(nrThreads)
   endif
#endif

#else /* WITH_OPENMP_TRADITIONAL */
    nrThreads = 1
#endif /* WITH_OPENMP_TRADITIONAL */

