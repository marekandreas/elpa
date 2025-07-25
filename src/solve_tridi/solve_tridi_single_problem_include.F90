#if 0
!    This file is part of ELPA.
!     
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!     
!    - Max Planck Computing and Data Facility (MPCDF), formerly known as
!      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
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
!    This particular source code file contains additions, changes and
!    enhancements authored by Intel Corporation which is not part of
!    the ELPA consortium.
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
!     
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!       
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
#endif  

      ! First try dstedc, this is normally faster but it may fail sometimes (why???)
      
      if (wantDebug) then
        call obj%timer%start("check_nans")
        has_nans = .false.

        do i = 1, nlen-1
          if (d(i) /= d(i) .or. e(i) /= e(i)) then
            has_nans = .true.
            exit
          endif
        enddo
        if (d(nlen) /= d(nlen)) has_nans = .true.

        if (has_nans) then
          write(error_unit,'(a)') 'ELPA1_solve_tridi_single: ERROR: NaNs found in input arrays d or e of STEDC. Aborting!'
          write(error_unit,'(a)') 'Check correctness of ELPA input!'
          stop 1
        endif
        
        call obj%timer%stop("check_nans")
      endif ! wantDebug

      lwork = 1 + 4*nlen + nlen**2
      liwork =  3 + 5*nlen
      allocate(work(lwork), iwork(liwork), stat=istat, errmsg=errorMessage)
      check_allocate("solve_tridi_single: work, iwork", istat, errorMessage)
      call obj%timer%start("lapack_stedc")
      call PRECISION_STEDC('I', int(nlen,kind=BLAS_KIND), d, e, q, int(ldq,kind=BLAS_KIND),    &
                          work, int(lwork,kind=BLAS_KIND), int(iwork,kind=BLAS_KIND), int(liwork,kind=BLAS_KIND), &
                          infoBLAS)
      info = int(infoBLAS,kind=ik)
      call obj%timer%stop("lapack_stedc")

      ! STEDC can affect the input arrays d and e, so we need to copy them if we want to use STEQR

      ! if (info /= 0) then
      !   ! DSTEDC failed, try DSTEQR. The workspace is enough for DSTEQR.
      !   write(error_unit,'(a,i8,a)') 'Warning: Lapack routine DSTEDC failed, info= ',info,', Trying DSTEQR!'

      !   d(:) = ds(:)
      !   e(:) = es(:)
      !   call obj%timer%start("lapack_steqr")
      !   call PRECISION_STEQR('I', int(nlen,kind=BLAS_KIND), d, e, q, int(ldq,kind=BLAS_KIND), work, infoBLAS )
      !   info = int(infoBLAS,kind=ik)
      !   call obj%timer%stop("lapack_steqr")
      ! end if

      ! If DSTEQR fails also, we don't know what to do further ...

      if (info /= 0) then
        if (wantDebug) then
          write(error_unit,'(a,i8,a)') 'ELPA1_solve_tridi_single: ERROR: Lapack routine STEDC failed, info= ',info,', Aborting!'
          write(error_unit,'(a)') 'Try to run ELPA in a debug mode, e.g. with export ELPA_DEFAULT_debug=1'
        endif
        !success = .false.
        !return ! this hangs
        stop 1
      endif


      deallocate(work,iwork,ds,es, stat=istat, errmsg=errorMessage)
      check_deallocate("solve_tridi_single: work, iwork, ds, es", istat, errorMessage)


