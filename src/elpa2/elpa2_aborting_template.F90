      write(error_unit,*) "Problem getting option for debug settings. Aborting..."
#ifdef ACTIVATE_SKEW
      call obj%timer%stop("elpa_solve_skew_evp_&
#else
      call obj%timer%stop("elpa_solve_evp_&
#endif
      &MATH_DATATYPE&
      &_2stage_&
      &PRECISION&
      &")
      success = .false.
      return

