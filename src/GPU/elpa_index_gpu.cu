extern "C" {
  int gpu_count() {
    int count;
    cudaError_t cuerr = cudaGetDeviceCount(&count);
    if (cuerr != cudaSuccess) {
      count = -1000;
    }
    return count;
  }
}
