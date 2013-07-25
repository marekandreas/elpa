extern "C" __device__ void reset_shared_block(signed char* _ps_block,signed char* _pb_size);
extern "C" __device__ void reset_shared_block_pair(signed char* _ps_block_1,signed char* _ps_block_2,signed char* _pb_size);
extern "C" __device__ void warp_reduce(signed char* _ps_block);
extern "C" __global__ void compute_hh_dotp_kernel(signed char* _phh,signed char* _pv_dot,int __V_nb,int __V_n);
extern "C" __global__ void extract_hh_tau_kernel(signed char* _phh,signed char* _phh_tau,int __V_nb,int __V_n,unsigned int __V_is_zero);
extern "C" __global__ void compute_hh_trafo_single_kernel(signed char* _pq,signed char* _phh,signed char* _phh_tau,int __V_nb,int __V_ldq,int __V_ncols);
extern "C" __global__ void compute_hh_trafo_kernel(signed char* _pq,signed char* _phh,signed char* _phh_dot,signed char* _phh_tau,int __V_nb,int __V_ldq,int __V_off,int __V_ncols);
extern "C" __global__ void my_pack_kernel(int __V_n_offset,int __V_max_idx,int __V_stripe_width,int __V_a_dim2,int __V_stripe_count,signed char* _psrc,signed char* _pdst,signed char* _pdst_sd);
extern "C" __global__ void my_unpack_kernel(int __V_n_offset,int __V_max_idx,int __V_stripe_width,int __V_a_dim2,int __V_stripe_count,signed char* _psrc,signed char* _pdst,signed char* _psrc_sd);
