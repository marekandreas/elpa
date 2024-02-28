// TODO: license

#define WITH_HIPCUB
// These two are actually not really header files, but symlinks to their respective versions in ELPA.
// This allows the proxy application to stay current while not requireing any changes to the original files.
#include "compute_hh_kernel_complex.hpp"
#include "compute_hh_kernel_real.hpp"

extern "C" void compute_hh_hip_gpu_complex_kernel(hipDoubleComplex *q, const hipDoubleComplex *hh, const hipDoubleComplex *hh_tau, const int nev, const int nb, const int ldq, const int ncols) {
    hipError_t err;
    switch (nb)
    {
    case 1024:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<1024>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 512:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<512>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 256:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<256>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 128:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<128>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 64:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<64>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 32:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<32>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<16>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 8:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<8>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 4:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<4>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 2:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<2>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 1:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<1>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    }

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        printf("\n compute_hh_trafo CUDA kernel failed: %s \n",hipGetErrorString(err));
    }
}


extern "C" void compute_hh_hip_gpu_real_kernel(double *q, const double *hh, const double *hh_tau, const int nev, const int nb, const int ldq, const int ncols)
{
    hipError_t err;

    switch (nb)
    {
    case 1024:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 1024>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 512:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 512>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 256:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 256>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 128:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 128>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 64:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 64>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 32:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 32>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 16:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 16>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 8:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 8>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 4:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 4>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 2:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 2>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 1:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 1>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
        break;
    }

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        printf("\n compute_hh_trafo CUDA kernel failed: %s \n",hipGetErrorString(err));
    }
}
