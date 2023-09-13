//    Copyright 2023, P. Karpov, MPCDF
//
//    This file is part of ELPA.
//
//    The ELPA library was originally created by the ELPA consortium,
//    consisting of the following organizations:
//
//    - Max Planck Computing and Data Facility (MPCDF), formerly known as
//      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
//    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
//      Informatik,
//    - Technische Universität München, Lehrstuhl für Informatik mit
//      Schwerpunkt Wissenschaftliches Rechnen ,
//    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
//    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
//      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
//      and
//    - IBM Deutschland GmbH
//
//
//    This particular source code file contains additions, changes and
//    enhancements authored by Intel Corporation which is not part of
//    the ELPA consortium.
//
//    More information can be found here:
//    http://elpa.mpcdf.mpg.de/
//
//    ELPA is free software: you can redistribute it and/or modify
//    it under the terms of the version 3 of the license of the
//    GNU Lesser General Public License as published by the Free
//    Software Foundation.
//
//    ELPA is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU Lesser General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with ELPA. If not, see <http://www.gnu.org/licenses/>
//
//    ELPA reflects a substantial effort on the part of the original
//    ELPA consortium, and we ask you to respect the spirit of the
//    license that we chose: i.e., please contribute any changes you
//    may have back to the original ELPA library distribution, and keep
//    any derivatives of ELPA under the same license that we chose for
//    the original distribution, the GNU Lesser General Public License.
//

#include <cstdio>
#include <CL/sycl.hpp>
#include "syclCommon.hpp"


extern "C" {

  int is_device_ptr(void *a_void_ptr) {

    sycl::queue q{sycl::default_selector()};
    sycl::usm::alloc a_void_ptr_alloc = sycl::get_pointer_type(a_void_ptr, q.get_context());

    if (a_void_ptr_alloc==sycl::usm::alloc::host)
        {
        //printf("is_device_ptr:  a_void_ptr_alloc==sycl::usm::alloc::host \n");
        return 0;
        }
    else if (a_void_ptr_alloc==sycl::usm::alloc::device)
        {
        //printf("is_device_ptr:  a_void_ptr_alloc==sycl::usm::alloc::device \n");
        return 1;
        }
    else if (a_void_ptr_alloc==sycl::usm::alloc::shared)
        {
        //printf("is_device_ptr:  a_void_ptr_alloc==sycl::usm::alloc::shared \n");
        return 1;
        }
    else if (a_void_ptr_alloc==sycl::usm::alloc::unknown)
        {
        //printf("is_device_ptr:  a_void_ptr_alloc==sycl::usm::alloc::unknown \n");
        return 0;
        }
    else
        {
        //printf("is_device_ptr: allocated in unknown fashion \n");
        return 0;
        }

  }
}
