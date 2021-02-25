extern "C" {
  int amd_gpu_count() {
    int count;
    hipError_t hiperr = hipGetDeviceCount(&count);
    if (hiperr != hipSuccess) {
      count = -1000;
    }
    return count;
  }
}
