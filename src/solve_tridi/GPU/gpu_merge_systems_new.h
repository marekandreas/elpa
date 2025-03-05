//    Copyright 2024, P. Karpov
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
//    This file was written by P. Karpov, MPCDF

//________________________________________________________________

// Device function that implements the bisection algorithm as in solve_secular_equation.
template <typename T>
__device__ void device_solve_secular_equation(int n, int i_f, const T* d1, const T* z1, T* delta, T* rho, T* dlam) {
    // i_f is the Fortran index (1-indexed); convert to C index:
    int i = i_f - 1;
    T dshift, a, b, x, y;
    const int maxIter = 200;
    // Use a tolerance value (for simplicity we use one value; in practice you might choose different ones for float/double)
    T tol = (sizeof(T) == sizeof(double)) ? (T)1e-200 : (T)1e-20;

    if(i_f == n) { 
         // Special case: last eigenvalue.
         dshift = d1[n-1];
         // delta[j] = d1[j] - dshift for all j.
         for (int j = 0; j < n; j++) {
              delta[j] = d1[j] - dshift;
         }
         // a = 0; b = rho*SUM(z1^2) + 1.
         a = 0;
         T sum_zsq = 0;
         for (int j = 0; j < n; j++) {
              sum_zsq += z1[j]*z1[j];
         }
         b = rho[0] * sum_zsq + 1;
    } else {
         // Other eigenvalues: lower bound is d1[i] and upper bound is d1[i+1]
         x = 0.5 * (d1[i] + d1[i+1]);
         T sum_term = 0;
         for (int j = 0; j < n; j++) {
              // Avoid division by zero (assume d1[j] != x)
              sum_term += z1[j]*z1[j] / (d1[j] - x);
         }
         y = 1.0 + rho[0] * sum_term;
         if (y > 0)
             dshift = d1[i];
         else
             dshift = d1[i+1];
         for (int j = 0; j < n; j++) {
              delta[j] = d1[j] - dshift;
         }
         a = delta[i];
         b = delta[i+1];
    }

    // Bisection loop
    for (int iter = 0; iter < maxIter; iter++) {
         x = 0.5 * (a + b);
         if (x == a || x == b)
             break;  // no further subdivision possible
         if (fabs(x) < tol)
             break;  // x is too close to zero (i.e. near a pole)
         T sum_term = 0;
         for (int j = 0; j < n; j++) {
              sum_term += z1[j]*z1[j] / (delta[j] - x);
         }
         y = 1.0 + rho[0] * sum_term;
         if (y == 0)
             break;  // exact solution found
         else if (y > 0)
             b = x;
         else
             a = x;
    }
    // dlam = x + dshift would be the computed eigenvalue (not stored here).
    

    // Update delta: delta[j] = delta[j] - x for all j.
    for (int j = 0; j < n; j++) {
         delta[j] = delta[j] - x;
    }
}

//________________________________________________________________

template <typename T>
__global__ void gpu_solve_secular_equation_loop_kernel(T *d1_dev, T *z1_dev, T *delta_dev, T *rho_dev, T *s_dev, 
                                      T *z_dev, T *dbase_dev, T *ddiff_dev, 
                                      int my_proc, int na1, int n_procs, int SM_count, int debug){

  int i_loc = threadIdx.x;
  int j_loc = blockIdx.x ;

  // do i = my_procs+1, na1, n_procs
  //   call solve_secular_equation_&
  //                               &PRECISION&
  //                               &(obj, na1, i, d1, z1, delta, rho, s) 

  //   ! Compute updated z
  //   do j=1,na1
  //     if (i/=j)  z(j) = z(j)*( delta(j) / (d1(j)-d1(i)) )
  //   enddo

  //   z(i) = z(i)*delta(i)
    
  //   ! Store dbase/ddiff

  //   if (i<na1) then
  //     if (abs(delta(i+1)) < abs(delta(i))) then
  //       dbase(i) = d1(i+1)
  //       ddiff(i) = delta(i+1)
  //     else
  //       dbase(i) = d1(i)
  //       ddiff(i) = delta(i)
  //     endif
  //   else
  //     dbase(i) = d1(i)
  //     ddiff(i) = delta(i)
  //   endif
  // enddo

  for (int i=my_proc + n_procs*blockIdx.x; i<na1; i += n_procs*gridDim.x)
    {
    //int i = i_f - 1;
    int i_f = i + 1;

    device_solve_secular_equation(na1, i_f, d1_dev, z1_dev, delta_dev, rho_dev, s_dev);
    
    // Compute updated z. PETERDEBUG111: this part can't be parallelized! But it can with MPI!
    for (int j = 0; j < na1; j++)
      {
      if (j != i) z_dev[j] = z_dev[j] * ( delta_dev[j] / (d1_dev[j] - d1_dev[i]) );  
      }

    z_dev[i] = z_dev[i] * delta_dev[i];

    // Store dbase/ddiff

    if (i_f < na1) 
      {
      if (fabs(delta_dev[i+1]) < fabs(delta_dev[i])) 
        {
        dbase_dev[i] = d1_dev[i+1];
        ddiff_dev[i] = delta_dev[i+1];
        }
      else 
        {
        dbase_dev[i] = d1_dev[i];
        ddiff_dev[i] = delta_dev[i];
        }
      } 
    else 
      {
      dbase_dev[i] = d1_dev[i];
      ddiff_dev[i] = delta_dev[i];
      }
    }
}

template <typename T>
void gpu_solve_secular_equation_loop (T *d1_dev, T *z1_dev, T *delta_dev, T *rho_dev, T *s_dev, 
                                      T *z_dev, T *dbase_dev, T *ddiff_dev, 
                                      int my_proc, int na1, int n_procs, int SM_count, int debug, gpuStream_t my_stream){

  //dim3 blocks = dim3(SM_count,1,1);
  //dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1); // PETERDEBUG111

  dim3 blocks = dim3(1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

  if (debug) // PETERDEBUG111
    {
    printf("gpu_solve_secular_equation_loop: blocks.x=%d, threadsPerBlock.x=%d\n",blocks.x,threadsPerBlock.x); //PETERDEBUG111
    }

#ifdef WITH_GPU_STREAMS
  gpu_solve_secular_equation_loop_kernel<<<blocks,threadsPerBlock,0,my_stream>>> (d1_dev, z1_dev, delta_dev, rho_dev, s_dev, 
                                                                                  z_dev, dbase_dev, ddiff_dev, my_proc, na1, n_procs, SM_count, debug);
#else
  gpu_solve_secular_equation_loop_kernel<<<blocks,threadsPerBlock>>>             (d1_dev, z1_dev, delta_dev, rho_dev, s_dev, 
                                                                                  z_dev, dbase_dev, ddiff_dev, my_proc, na1, n_procs, SM_count, debug);
#endif

  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_solve_secular_equation_loop: %s\n",gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _solve_secular_equation_loop_FromC) (char dataType, intptr_t d1_dev, intptr_t z1_dev, intptr_t delta_dev, intptr_t rho_dev, intptr_t s_dev,
                                                                            intptr_t z_dev, intptr_t dbase_dev, intptr_t ddiff_dev, 
                                                                            int my_proc, int na1, int n_procs, int SM_count, int debug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_solve_secular_equation_loop<double>((double *) d1_dev, (double *) z1_dev, (double *) delta_dev, (double *) rho_dev, (double *) s_dev,
                                                                  (double *) z_dev, (double *) dbase_dev, (double *) ddiff_dev,
                                                                  my_proc, na1, n_procs, SM_count, debug, my_stream);
  else if (dataType=='S') gpu_solve_secular_equation_loop<float> ((float  *) d1_dev, (float  *) z1_dev, (float  *) delta_dev, (float  *) rho_dev, (float  *) s_dev,
                                                                  (float  *) z_dev, (float  *) dbase_dev, (float  *) ddiff_dev,
                                                                  my_proc, na1, n_procs, SM_count, debug, my_stream);
  else {
    printf("Error in gpu_solve_secular_equation_loop: Unsupported data type\n");
  }
}

//________________________________________________________________
