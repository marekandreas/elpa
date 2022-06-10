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
//    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
//
//    ELPA reflects a substantial effort on the part of the original
//    ELPA consortium, and we ask you to respect the spirit of the
//    license that we chose: i.e., please contribute any changes you
//    may have back to the original ELPA library distribution, and keep
//    any derivatives of ELPA under the same license that we chose for
//    the original distribution, the GNU Lesser General Public License.
//
//

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <complex>

template <typename T, unsigned int blk> void warp_reduce_complex(volatile T *s_block,
                                                                 sycl::nd_item<3> item_ct1)
{
    unsigned int tid = item_ct1.get_local_id(2);

    if (blk >= 64)
    {
        if (tid < 32)
        {
            s_block[tid].x() += s_block[tid + 32].x();
            s_block[tid].y() += s_block[tid + 32].y();
        }
    }

    if (blk >= 32)
    {
        if (tid < 16)
        {
            s_block[tid].x() += s_block[tid + 16].x();
            s_block[tid].y() += s_block[tid + 16].y();
        }
    }

    if (blk >= 16)
    {
        if (tid < 8)
        {
            s_block[tid].x() += s_block[tid + 8].x();
            s_block[tid].y() += s_block[tid + 8].y();
        }
    }

    if (blk >= 8)
    {
        if (tid < 4)
        {
            s_block[tid].x() += s_block[tid + 4].x();
            s_block[tid].y() += s_block[tid + 4].y();
        }
    }

    if (blk >= 4)
    {
        if (tid < 2)
        {
            s_block[tid].x() += s_block[tid + 2].x();
            s_block[tid].y() += s_block[tid + 2].y();
        }
    }

    if (blk >= 2)
    {
        if (tid < 1)
        {
            s_block[tid].x() += s_block[tid + 1].x();
            s_block[tid].y() += s_block[tid + 1].y();
        }
    }
}

template <typename T, unsigned int blk> void reduce_complex(T *s_block,
                                                            sycl::nd_item<3> item_ct1)
{
    unsigned int tid = item_ct1.get_local_id(2);

    if (blk >= 1024)
    {
        if (tid < 512)
        {
            s_block[tid].x() += s_block[tid + 512].x();
            s_block[tid].y() += s_block[tid + 512].y();
        }

        /*
        DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    if (blk >= 512)
    {
        if (tid < 256)
        {
            s_block[tid].x() += s_block[tid + 256].x();
            s_block[tid].y() += s_block[tid + 256].y();
        }

        /*
        DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    if (blk >= 256)
    {
        if (tid < 128)
        {
            s_block[tid].x() += s_block[tid + 128].x();
            s_block[tid].y() += s_block[tid + 128].y();
        }

        /*
        DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    if (blk >= 128)
    {
        if (tid < 64)
        {
            s_block[tid].x() += s_block[tid + 64].x();
            s_block[tid].y() += s_block[tid + 64].y();
        }

        /*
        DPCT1065:3: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    if (tid < 32)
    {
        warp_reduce_complex<T, blk>(s_block, item_ct1);
    }

}

template <unsigned int blk>
void compute_hh_trafo_cuda_kernel_complex_double(
    sycl::double2 *__restrict__ q, const sycl::double2 *__restrict__ hh,
    const sycl::double2 *__restrict__ hh_tau, const int nb, const int ldq,
    const int ncols, sycl::nd_item<3> item_ct1, sycl::double2 *q_s,
    sycl::double2 *dotp_s)
{

    sycl::double2 q_v2;

    int q_off, h_off, j;

    unsigned int tid = item_ct1.get_local_id(2);
    unsigned int bid = item_ct1.get_group(2);

    j = ncols;
    q_off = bid + (j + tid - 1) * ldq;
    h_off = tid + (j - 1) * nb;
    q_s[tid] = q[q_off];

    while (j >= 1)
    {
        if (tid == 0)
        {
            q_s[tid] = q[q_off];
        }

        q_v2 = q_s[tid];
        dotp_s[tid] = cuCmul(q_v2, cuConj(hh[h_off]));

        /*
        DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        reduce_complex<sycl::double2, blk>(dotp_s, item_ct1);

        /*
        DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        q_v2 = cuCsub(q_v2, cuCmul(cuCmul(dotp_s[0], hh_tau[j - 1]), hh[h_off]));
        q_s[tid + 1] = q_v2;

        if ((j == 1) || (tid == item_ct1.get_local_range(2) - 1))
        {
            q[q_off] = q_v2;
        }

        /*
        DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        q_off -= ldq;
        h_off -= nb;
        j -= 1;
    }
}

extern "C" void launch_compute_hh_trafo_c_cuda_kernel_complex_double(
    sycl::double2 *q, const sycl::double2 *hh, const sycl::double2 *hh_tau,
    const int nev, const int nb, const int ldq, const int ncols)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    int err;

    switch (nb)
    {
    case 1024:
        /*
        DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(1024 + 1), cgh);
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(1024), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                      sycl::range<3>(1, 1, nb),
                                  sycl::range<3>(1, 1, nb)),
                [=](sycl::nd_item<3> item_ct1) {
                    compute_hh_trafo_cuda_kernel_complex_double<1024>(
                        q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                        q_s_acc_ct1.get_pointer(),
                        dotp_s_acc_ct1.get_pointer());
                });
        });
        break;
    case 512:
        /*
        DPCT1049:8: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(512 + 1), cgh);
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(512), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                      sycl::range<3>(1, 1, nb),
                                  sycl::range<3>(1, 1, nb)),
                [=](sycl::nd_item<3> item_ct1) {
                    compute_hh_trafo_cuda_kernel_complex_double<512>(
                        q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                        q_s_acc_ct1.get_pointer(),
                        dotp_s_acc_ct1.get_pointer());
                });
        });
        break;
    case 256:
        /*
        DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(256 + 1), cgh);
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(256), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                      sycl::range<3>(1, 1, nb),
                                  sycl::range<3>(1, 1, nb)),
                [=](sycl::nd_item<3> item_ct1) {
                    compute_hh_trafo_cuda_kernel_complex_double<256>(
                        q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                        q_s_acc_ct1.get_pointer(),
                        dotp_s_acc_ct1.get_pointer());
                });
        });
        break;
    case 128:
        /*
        DPCT1049:10: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(128 + 1), cgh);
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(128), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                      sycl::range<3>(1, 1, nb),
                                  sycl::range<3>(1, 1, nb)),
                [=](sycl::nd_item<3> item_ct1) {
                    compute_hh_trafo_cuda_kernel_complex_double<128>(
                        q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                        q_s_acc_ct1.get_pointer(),
                        dotp_s_acc_ct1.get_pointer());
                });
        });
        break;
    case 64:
        /*
        DPCT1049:11: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(64 + 1), cgh);
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(64), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                      sycl::range<3>(1, 1, nb),
                                  sycl::range<3>(1, 1, nb)),
                [=](sycl::nd_item<3> item_ct1) {
                    compute_hh_trafo_cuda_kernel_complex_double<64>(
                        q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                        q_s_acc_ct1.get_pointer(),
                        dotp_s_acc_ct1.get_pointer());
                });
        });
        break;
    case 32:
        /*
        DPCT1049:12: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(32 + 1), cgh);
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(32), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                      sycl::range<3>(1, 1, nb),
                                  sycl::range<3>(1, 1, nb)),
                [=](sycl::nd_item<3> item_ct1) {
                    compute_hh_trafo_cuda_kernel_complex_double<32>(
                        q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                        q_s_acc_ct1.get_pointer(),
                        dotp_s_acc_ct1.get_pointer());
                });
        });
        break;
    case 16:
        /*
        DPCT1049:13: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(16 + 1), cgh);
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(16), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                      sycl::range<3>(1, 1, nb),
                                  sycl::range<3>(1, 1, nb)),
                [=](sycl::nd_item<3> item_ct1) {
                    compute_hh_trafo_cuda_kernel_complex_double<16>(
                        q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                        q_s_acc_ct1.get_pointer(),
                        dotp_s_acc_ct1.get_pointer());
                });
        });
        break;
    case 8:
        /*
        DPCT1049:14: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(8 + 1), cgh);
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(8), cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                                   sycl::range<3>(1, 1, nb),
                                               sycl::range<3>(1, 1, nb)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 compute_hh_trafo_cuda_kernel_complex_double<8>(
                                     q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                                     q_s_acc_ct1.get_pointer(),
                                     dotp_s_acc_ct1.get_pointer());
                             });
        });
        break;
    case 4:
        /*
        DPCT1049:15: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(4 + 1), cgh);
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(4), cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                                   sycl::range<3>(1, 1, nb),
                                               sycl::range<3>(1, 1, nb)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 compute_hh_trafo_cuda_kernel_complex_double<4>(
                                     q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                                     q_s_acc_ct1.get_pointer(),
                                     dotp_s_acc_ct1.get_pointer());
                             });
        });
        break;
    case 2:
        /*
        DPCT1049:16: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(2 + 1), cgh);
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(2), cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                                   sycl::range<3>(1, 1, nb),
                                               sycl::range<3>(1, 1, nb)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 compute_hh_trafo_cuda_kernel_complex_double<2>(
                                     q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                                     q_s_acc_ct1.get_pointer(),
                                     dotp_s_acc_ct1.get_pointer());
                             });
        });
        break;
    case 1:
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(1 + 1), cgh);
            sycl::accessor<sycl::double2, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(1), cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                                   sycl::range<3>(1, 1, nb),
                                               sycl::range<3>(1, 1, nb)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 compute_hh_trafo_cuda_kernel_complex_double<1>(
                                     q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                                     q_s_acc_ct1.get_pointer(),
                                     dotp_s_acc_ct1.get_pointer());
                             });
        });
        break;
    }

    /*
    DPCT1010:18: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    err = 0;
}

template <unsigned int blk>
void compute_hh_trafo_cuda_kernel_complex_single(cuFloatComplex * __restrict__ q, const cuFloatComplex * __restrict__ hh, const cuFloatComplex * __restrict__ hh_tau, const int nb, const int ldq, const int ncols,
                                                 sycl::nd_item<3> item_ct1,
                                                 cuFloatComplex *q_s,
                                                 cuFloatComplex *dotp_s)
{

    cuFloatComplex q_v2;

    int q_off, h_off, j;

    unsigned int tid = item_ct1.get_local_id(2);
    unsigned int bid = item_ct1.get_group(2);

    j = ncols;
    q_off = bid + (j + tid - 1) * ldq;
    h_off = tid + (j - 1) * nb;
    q_s[tid] = q[q_off];

    while (j >= 1)
    {
        if (tid == 0)
        {
            q_s[tid] = q[q_off];
        }

        q_v2 = q_s[tid];
        dotp_s[tid] = cuCmulf(q_v2, cuConjf(hh[h_off]));

        /*
        DPCT1065:20: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        reduce_complex<cuFloatComplex, blk>(dotp_s, item_ct1);

        /*
        DPCT1065:21: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        q_v2 = cuCsubf(q_v2, cuCmulf(cuCmulf(dotp_s[0], hh_tau[j - 1]), hh[h_off]));
        q_s[tid + 1] = q_v2;

        if ((j == 1) || (tid == item_ct1.get_local_range(2) - 1))
        {
            q[q_off] = q_v2;
        }

        /*
        DPCT1065:22: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        q_off -= ldq;
        h_off -= nb;
        j -= 1;
    }
}

extern "C" void launch_compute_hh_trafo_c_cuda_kernel_complex_single(cuFloatComplex *q, const cuFloatComplex *hh, const cuFloatComplex *hh_tau, const int nev, const int nb, const int ldq, const int ncols)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    int err;

    switch (nb)
    {
    case 1024:
        /*
        DPCT1049:23: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(1024 + 1), cgh);
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(1024), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                      sycl::range<3>(1, 1, nb),
                                  sycl::range<3>(1, 1, nb)),
                [=](sycl::nd_item<3> item_ct1) {
                    compute_hh_trafo_cuda_kernel_complex_single<1024>(
                        q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                        q_s_acc_ct1.get_pointer(),
                        dotp_s_acc_ct1.get_pointer());
                });
        });
        break;
    case 512:
        /*
        DPCT1049:24: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(512 + 1), cgh);
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(512), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                      sycl::range<3>(1, 1, nb),
                                  sycl::range<3>(1, 1, nb)),
                [=](sycl::nd_item<3> item_ct1) {
                    compute_hh_trafo_cuda_kernel_complex_single<512>(
                        q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                        q_s_acc_ct1.get_pointer(),
                        dotp_s_acc_ct1.get_pointer());
                });
        });
        break;
    case 256:
        /*
        DPCT1049:25: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(256 + 1), cgh);
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(256), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                      sycl::range<3>(1, 1, nb),
                                  sycl::range<3>(1, 1, nb)),
                [=](sycl::nd_item<3> item_ct1) {
                    compute_hh_trafo_cuda_kernel_complex_single<256>(
                        q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                        q_s_acc_ct1.get_pointer(),
                        dotp_s_acc_ct1.get_pointer());
                });
        });
        break;
    case 128:
        /*
        DPCT1049:26: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(128 + 1), cgh);
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(128), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                      sycl::range<3>(1, 1, nb),
                                  sycl::range<3>(1, 1, nb)),
                [=](sycl::nd_item<3> item_ct1) {
                    compute_hh_trafo_cuda_kernel_complex_single<128>(
                        q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                        q_s_acc_ct1.get_pointer(),
                        dotp_s_acc_ct1.get_pointer());
                });
        });
        break;
    case 64:
        /*
        DPCT1049:27: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(64 + 1), cgh);
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(64), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                      sycl::range<3>(1, 1, nb),
                                  sycl::range<3>(1, 1, nb)),
                [=](sycl::nd_item<3> item_ct1) {
                    compute_hh_trafo_cuda_kernel_complex_single<64>(
                        q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                        q_s_acc_ct1.get_pointer(),
                        dotp_s_acc_ct1.get_pointer());
                });
        });
        break;
    case 32:
        /*
        DPCT1049:28: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(32 + 1), cgh);
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(32), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                      sycl::range<3>(1, 1, nb),
                                  sycl::range<3>(1, 1, nb)),
                [=](sycl::nd_item<3> item_ct1) {
                    compute_hh_trafo_cuda_kernel_complex_single<32>(
                        q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                        q_s_acc_ct1.get_pointer(),
                        dotp_s_acc_ct1.get_pointer());
                });
        });
        break;
    case 16:
        /*
        DPCT1049:29: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(16 + 1), cgh);
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(16), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                      sycl::range<3>(1, 1, nb),
                                  sycl::range<3>(1, 1, nb)),
                [=](sycl::nd_item<3> item_ct1) {
                    compute_hh_trafo_cuda_kernel_complex_single<16>(
                        q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                        q_s_acc_ct1.get_pointer(),
                        dotp_s_acc_ct1.get_pointer());
                });
        });
        break;
    case 8:
        /*
        DPCT1049:30: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(8 + 1), cgh);
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(8), cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                                   sycl::range<3>(1, 1, nb),
                                               sycl::range<3>(1, 1, nb)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 compute_hh_trafo_cuda_kernel_complex_single<8>(
                                     q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                                     q_s_acc_ct1.get_pointer(),
                                     dotp_s_acc_ct1.get_pointer());
                             });
        });
        break;
    case 4:
        /*
        DPCT1049:31: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(4 + 1), cgh);
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(4), cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                                   sycl::range<3>(1, 1, nb),
                                               sycl::range<3>(1, 1, nb)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 compute_hh_trafo_cuda_kernel_complex_single<4>(
                                     q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                                     q_s_acc_ct1.get_pointer(),
                                     dotp_s_acc_ct1.get_pointer());
                             });
        });
        break;
    case 2:
        /*
        DPCT1049:32: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(2 + 1), cgh);
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(2), cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                                   sycl::range<3>(1, 1, nb),
                                               sycl::range<3>(1, 1, nb)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 compute_hh_trafo_cuda_kernel_complex_single<2>(
                                     q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                                     q_s_acc_ct1.get_pointer(),
                                     dotp_s_acc_ct1.get_pointer());
                             });
        });
        break;
    case 1:
        /*
        DPCT1049:33: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                q_s_acc_ct1(sycl::range<1>(1 + 1), cgh);
            sycl::accessor<cuFloatComplex, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                dotp_s_acc_ct1(sycl::range<1>(1), cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nev) *
                                                   sycl::range<3>(1, 1, nb),
                                               sycl::range<3>(1, 1, nb)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 compute_hh_trafo_cuda_kernel_complex_single<1>(
                                     q, hh, hh_tau, nb, ldq, ncols, item_ct1,
                                     q_s_acc_ct1.get_pointer(),
                                     dotp_s_acc_ct1.get_pointer());
                             });
        });
        break;
    }

    /*
    DPCT1010:34: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    err = 0;
}
