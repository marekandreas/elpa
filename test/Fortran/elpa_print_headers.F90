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

#ifdef WITH_OPENMP_TRADITIONAL
   if (myid .eq. 0) then
      print *,"Threaded version of test program"
      print *,"Using ",omp_get_max_threads()," threads"
      print *," "
   endif
#endif

#ifndef WITH_MPI
   if (myid .eq. 0) then
     print *,"This version of ELPA does not support MPI parallelisation"
     print *,"For MPI support re-build ELPA with appropiate flags"
     print *," "
   endif
#endif

#ifdef ELPA1

#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
   if (myid .eq. 0) then
     print *," "
     print *,"Real valued double-precision version of ELPA1 is used"
     print *," "
   endif
#else
   if (myid .eq. 0) then
     print *," "
     print *,"Real valued single-precision version of ELPA1 is used"
     print *," "
   endif
#endif

#endif

#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
   if (myid .eq. 0) then
     print *," "
     print *,"Complex valued double-precision version of ELPA1 is used"
     print *," "
   endif
#else
   if (myid .eq. 0) then
     print *," "
     print *,"Complex valued single-precision version of ELPA1 is used"
     print *," "
   endif
#endif

#endif /* DATATYPE */

#else /* ELPA1 */

#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
   if (myid .eq. 0) then
     print *," "
     print *,"Real valued double-precision version of ELPA2 is used"
     print *," "
   endif
#else
   if (myid .eq. 0) then
     print *," "
     print *,"Real valued single-precision version of ELPA2 is used"
     print *," "
   endif
#endif

#endif

#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
   if (myid .eq. 0) then
     print *," "
     print *,"Complex valued double-precision version of ELPA2 is used"
     print *," "
   endif
#else
   if (myid .eq. 0) then
     print *," "
     print *,"Complex valued single-precision version of ELPA2 is used"
     print *," "
   endif
#endif

#endif /* DATATYPE */

#endif /* ELPA1 */

#ifdef WITH_MPI
    call MPI_BARRIER(MPI_COMM_WORLD, mpierr)
#endif
#ifdef HAVE_REDIRECT
   if (check_redirect_environment_variable()) then
     if (myid .eq. 0) then
       print *," "
       print *,"Redirection of mpi processes is used"
       print *," "
       if (create_directories() .ne. 1) then
         write(error_unit,*) "Unable to create directory for stdout and stderr!"
         stop 1
       endif
     endif
#ifdef WITH_MPI
     call MPI_BARRIER(MPI_COMM_WORLD, mpierr)
#endif
     call redirect_stdout(myid)
   endif
#endif

#ifndef ELPA1

   if (myid .eq. 0) then
      print *," "
      print *,"This ELPA2 is build with"
#ifdef WITH_NVIDIA_GPU_KERNEL
        print *,"CUDA GPU support"
#endif
#ifdef WITH_INTEL_GPU_KERNEL
        print *,"INTEL GPU support"
#endif
#ifdef WITH_AMD_GPU_KERNEL
        print *,"AMD GPU support"
#endif
      print *," "
#ifdef REALCASE

#ifdef HAVE_AVX2

#ifdef WITH_REAL_AVX_BLOCK2_KERNEL
      print *,"AVX2 optimized kernel (2 blocking) for real matrices"
#endif
#ifdef WITH_REAL_AVX_BLOCK4_KERNEL
      print *,"AVX2 optimized kernel (4 blocking) for real matrices"
#endif
#ifdef WITH_REAL_AVX_BLOCK6_KERNEL
      print *,"AVX2 optimized kernel (6 blocking) for real matrices"
#endif

#else /* no HAVE_AVX2 */

#ifdef HAVE_AVX

#ifdef WITH_REAL_AVX_BLOCK2_KERNEL
      print *,"AVX optimized kernel (2 blocking) for real matrices"
#endif
#ifdef WITH_REAL_AVX_BLOCK4_KERNEL
      print *,"AVX optimized kernel (4 blocking) for real matrices"
#endif
#ifdef WITH_REAL_AVX_BLOCK6_KERNEL
      print *,"AVX optimized kernel (6 blocking) for real matrices"
#endif

#endif

#endif /* HAVE_AVX2 */


#ifdef WITH_REAL_GENERIC_KERNEL
     print *,"GENERIC kernel for real matrices"
#endif
#ifdef WITH_REAL_GENERIC_SIMPLE_KERNEL
     print *,"GENERIC SIMPLE kernel for real matrices"
#endif
#ifdef WITH_REAL_SSE_ASSEMBLY_KERNEL
     print *,"SSE ASSEMBLER kernel for real matrices"
#endif
#ifdef WITH_REAL_BGP_KERNEL
     print *,"BGP kernel for real matrices"
#endif
#ifdef WITH_REAL_BGQ_KERNEL
     print *,"BGQ kernel for real matrices"
#endif

#endif /* DATATYPE == REAL */

#ifdef COMPLEXCASE

#ifdef HAVE_AVX2

#ifdef  WITH_COMPLEX_AVX_BLOCK2_KERNEL
      print *,"AVX2 optimized kernel (2 blocking) for complex matrices"
#endif
#ifdef WITH_COMPLEX_AVX_BLOCK1_KERNEL
      print *,"AVX2 optimized kernel (1 blocking) for complex matrices"
#endif

#else /* no HAVE_AVX2 */

#ifdef HAVE_AVX

#ifdef  WITH_COMPLEX_AVX_BLOCK2_KERNEL
      print *,"AVX optimized kernel (2 blocking) for complex matrices"
#endif
#ifdef WITH_COMPLEX_AVX_BLOCK1_KERNEL
      print *,"AVX optimized kernel (1 blocking) for complex matrices"
#endif

#endif

#endif /* HAVE_AVX2 */


#ifdef WITH_COMPLEX_GENERIC_KERNEL
     print *,"GENERIC kernel for complex matrices"
#endif
#ifdef WITH_COMPLEX_GENERIC_SIMPLE_KERNEL
     print *,"GENERIC SIMPLE kernel for complex matrices"
#endif
#ifdef WITH_COMPLEX_SSE_ASSEMBLY_KERNEL
     print *,"SSE ASSEMBLER kernel for complex matrices"
#endif

#endif /* DATATYPE == COMPLEX */

   endif
#endif /* ELPA1 */

   if (write_to_file%eigenvectors) then
     if (myid .eq. 0) print *,"Writing Eigenvectors to files"
   endif

   if (write_to_file%eigenvalues) then
     if (myid .eq. 0) print *,"Writing Eigenvalues to files"
   endif


