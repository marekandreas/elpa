// Yes, I am aware this is bad style. However, I do not want to modify the original file, and it contains a template.
#include "compute_hh_cuda_complex.cu"

extern "C" void compute_hh_cuda_gpu_complex_kernel(cuDoubleComplex *q, const cuDoubleComplex *hh, const cuDoubleComplex *hh_tau, const int nev, const int nb, const int ldq, const int ncols) {
    cudaError_t err;

    switch (nb) {
        case 1024: compute_hh_trafo_cuda_kernel_complex_double<1024><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols); break;
        case 512:  compute_hh_trafo_cuda_kernel_complex_double<512><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);  break;
        case 256:  compute_hh_trafo_cuda_kernel_complex_double<256><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);  break;
        case 128:  compute_hh_trafo_cuda_kernel_complex_double<128><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);  break;
        case 64:   compute_hh_trafo_cuda_kernel_complex_double<64><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);   break;
        case 32:   compute_hh_trafo_cuda_kernel_complex_double<32><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);   break;
        case 16:   compute_hh_trafo_cuda_kernel_complex_double<16><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);   break;
        case 8:    compute_hh_trafo_cuda_kernel_complex_double<8><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);    break;
        case 4:    compute_hh_trafo_cuda_kernel_complex_double<4><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);    break;
        case 2:    compute_hh_trafo_cuda_kernel_complex_double<2><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);    break;
        case 1:    compute_hh_trafo_cuda_kernel_complex_double<1><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);    break;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("\n compute_hh_trafo CUDA kernel failed: %s \n",cudaGetErrorString(err));
    }
}

