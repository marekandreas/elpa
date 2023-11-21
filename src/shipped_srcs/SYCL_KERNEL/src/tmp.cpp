
template <typename T, int wg_size, int sg_size>
void compute_hh_trafo_c_sycl_kernel(T *q, T const *hh, T const *hh_tau, int const nev, int const nb, int const ldq, int const ncols) {
  using local_buffer = sycl::local_accessor<T>;

  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  queue.submit([&](sycl::handler &h) {
    local_buffer q_s(sycl::range(nb+1), h);
    local_buffer q_s_reserve(sycl::range(nb), h);

    // For real numbers, using the custom reduction is still a lot faster, for complex ones, the SYCL one is better.
    // And for the custom reduction we need the SLM.
    sycl::range<1> r(0);
    if constexpr (!is_complex_number<T>::value) {
      r = sycl::range<1>(nb + 1);
    }
    local_buffer dotp_s(r, h);

    sycl::range<1> global_range(nev * nb);
    sycl::range<1> local_range(nb);
    h.parallel_for(sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]]{
      using sf = sycl::access::fence_space;
      int tid = it.get_local_id(0);
      int local_range = it.get_local_range(0);
      auto g = it.get_group();

      int j = ncols;
      int reserve_counter = wg_size + 2;
      int q_off = (j + tid - 1) * ldq + it.get_group(0);
      int q_off_res = (j - 1) * ldq + it.get_group(0);
      int h_off = tid + (j - 1) * nb;

      q_s[tid] = q[q_off];

      for (; j >= 1; j--) {
        // We can preload the q values into shared local memoory every X iterations
        // instead of doing it in every iteration. This seems to save some time.
        if (reserve_counter > sg_size) {
          q_off_res = sycl::group_broadcast(g, q_off);
          if (j - tid >= 1 && tid <= sg_size) {
            //int idx = (q_off_res - tid * ldq >= 0) ? (q_off_res - tid * ldq) : 0;
            //int idx = q_off_res - tid * ldq;
            //q_s_reserve[tid] = q[idx];
            q_s_reserve[tid] = q[q_off_res - tid * ldq];
          }
          reserve_counter = 0;
        }

        // All work items use the same value of hh_tau. Be explicit about only loading it once,
        // and broadcast it to the other work items in the group. (Also, in this loop,
        // in continuation of the above, a q value from the reserve is consumed.
        T hh_tau_jm1;
        if (tid == 0) {
            q_s[tid] = q[q_off];//q_s_reserve[reserve_counter];
            hh_tau_jm1 = hh_tau[j-1];
        }
        reserve_counter++;

        hh_tau_jm1 = sycl::group_broadcast(it.get_group(), hh_tau_jm1);
        T q_v2 = q_s[tid];
        T hh_h_off = hh[h_off];

        // For Complex numbers, the native SYCL implementation of the reduction is faster than the hand-coded one. But for
        // real numbers, there's still a significant advantage to using the hand-crafted solution. The correct variant is
        // picked at template instantiation time.
        T dotp_res;
        if constexpr (is_complex_number<T>::value) {
          // I don't get it. Is it now faster or slower?!
          //dotp_s[tid] = q_v2 * std::conj(hh_h_off); //hh_h_off;
          //it.barrier(sf::local_space);
          it.barrier();
          dotp_res = sycl::reduce_over_group(g, q_v2 * std::conj(hh_h_off), sycl::plus<>());
          //dotp_res = parallel_sum_group<T, wg_size, sg_size>(it, dotp_s.get_pointer());
        } else {
          dotp_s[tid] = q_v2 * hh_h_off;
          //it.barrier(sf::local_space);
          it.barrier();
          //dotp_res = parallel_sum_group<T, wg_size, sg_size>(it, dotp_s.get_pointer());
          dotp_res = sycl::reduce_over_group(g, q_v2 * hh_h_off, sycl::plus<>());
        }

        q_v2 -= dotp_res * hh_tau_jm1 * hh_h_off;
        q_s[tid + 1] = q_v2;

        if ((j == 1) || (tid == it.get_local_range()[0] - 1)) {
           q[q_off] = q_v2;
        }

        q_off -= ldq;
        q_off_res -= ldq;
        h_off -= nb;
      }
    });
  });
  queue.wait_and_throw();
}
