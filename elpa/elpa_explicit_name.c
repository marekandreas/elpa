//    Copyright 2022, P. Karpov, MPCDF
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


#include <stdio.h>
#include "elpa_explicit_name.h"



/*! \brief generic C method for elpa_eigenvectors_double
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       double host/device pointer to matrix a in CPU/GPU memory
 *  \param  ev      on return: double pointer to eigenvalues in CPU/GPU memory
 *  \param  q       on return: double pointer to eigenvectors in CPU/GPU memory
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_eigenvectors_double(elpa_t handle, double *a, double *ev, double *q, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1 || ELPA_WITH_SYCL_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		//printf("elpa_eigenvectors_double dev\n");
		elpa_eigenvectors_d_ptr_d(handle, a, ev, q, error);
		}
	else {
		//printf("elpa_eigenvectors_double host\n");
		elpa_eigenvectors_a_h_a_d(handle, a, ev, q, error);
		}	
#else
	//printf("elpa_eigenvectors_double non-gpu version\n");
	elpa_eigenvectors_a_h_a_d(handle, a, ev, q, error);
#endif		
	}

/*! \brief generic C method for elpa_eigenvectors_float
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       float host/device pointer to matrix a in CPU/GPU memory
 *  \param  ev      on return: float pointer to eigenvalues in CPU/GPU memory
 *  \param  q       on return: float pointer to eigenvectors in CPU/GPU memory
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_eigenvectors_float(elpa_t handle, float *a, float *ev, float *q, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		//printf("elpa_eigenvectors_float dev\n");
		elpa_eigenvectors_d_ptr_f(handle, a, ev, q, error);
		}
	else {
		//printf("elpa_eigenvectors_float host\n");
		elpa_eigenvectors_a_h_a_f(handle, a, ev, q, error);
		}	
#else
   //printf("elpa_eigenvectors_float non-nvidia version\n");
	elpa_eigenvectors_a_h_a_f(handle, a, ev, q, error);
#endif		
	}

/*! \brief generic C method for elpa_eigenvectors_double_complex
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       double complex host/device pointer to matrix a in CPU/GPU memory
 *  \param  ev      on return: double complex pointer to eigenvalues in CPU/GPU memory
 *  \param  q       on return: double complex pointer to eigenvectors in CPU/GPU memory
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_eigenvectors_double_complex(elpa_t handle, double complex *a, double *ev, double complex *q, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
   
	if (IsDevicePtr){
		//printf("elpa_eigenvectors_double_complex\n");
		elpa_eigenvectors_d_ptr_dc(handle, a, ev, q, error);
		}
	else {
		//printf("elpa_eigenvectors_double_complex\n");
		elpa_eigenvectors_a_h_a_dc(handle, a, ev, q, error);
		}	
		
#else
	//printf("elpa_eigenvectors_double_complex non-nvidia version\n");
	elpa_eigenvectors_a_h_a_dc(handle, a, ev, q, error);
#endif		
	}

/*! \brief generic C method for elpa_eigenvectors_float_complex
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       float complex host/device pointer to matrix a in CPU/GPU memory
 *  \param  ev      on return: float complex pointer to eigenvalues in CPU/GPU memory
 *  \param  q       on return: float complex pointer to eigenvectors in CPU/GPU memory
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_eigenvectors_float_complex(elpa_t handle, float complex *a, float *ev, float complex *q, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
   
	if (IsDevicePtr){
		//printf("elpa_eigenvectors_float_complex\n");
		elpa_eigenvectors_d_ptr_fc(handle, a, ev, q, error);
		}
	else {
		//printf("elpa_eigenvectors_float_complex\n");
		elpa_eigenvectors_a_h_a_fc(handle, a, ev, q, error);
		}	
		
#else
	//printf("elpa_eigenvectors_float_complex non-nvidia version\n");
	elpa_eigenvectors_a_h_a_fc(handle, a, ev, q, error);
#endif		
	}


#ifdef HAVE_SKEWSYMMETRIC
/*! \brief generic C method for elpa_skew_eigenvectors_double
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       double pointer to matrix a
 *  \param  ev      on return: double pointer to eigenvalues
 *  \param  q       on return: double pointer to eigenvectors
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_skew_eigenvectors_double(elpa_t handle, double *a, double *ev, double *q, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_skew_eigenvectors_d_ptr_d(handle, a, ev, q, error);
		}
	else {
		elpa_skew_eigenvectors_a_h_a_d(handle, a, ev, q, error);
		}	
#else
   elpa_skew_eigenvectors_a_h_a_d(handle, a, ev, q, error);
#endif		
	}

/*! \brief generic C method for elpa_skew_eigenvectors_float
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       float pointer to matrix a
 *  \param  ev      on return: float pointer to eigenvalues
 *  \param  q       on return: float pointer to eigenvectors
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_skew_eigenvectors_float(elpa_t handle, float *a, float *ev, float *q, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_skew_eigenvectors_d_ptr_f(handle, a, ev, q, error);
		}
	else {
		elpa_skew_eigenvectors_a_h_a_f(handle, a, ev, q, error);
		}	
#else
   elpa_skew_eigenvectors_a_h_a_f(handle, a, ev, q, error);
#endif		
	}
#endif /* HAVE_SKEWSYMMETRIC */

/*! \brief generic C method for elpa_eigenvalues_double
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       double pointer to matrix a in GPU memory
 *  \param  ev      on return: double pointer to eigenvalues in GPU memory
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_eigenvalues_double(elpa_t handle, double *a, double *ev, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_eigenvalues_d_ptr_d(handle, a, ev, error);
		}
	else {
		elpa_eigenvalues_a_h_a_d(handle, a, ev, error);
		}	
#else
   elpa_eigenvalues_a_h_a_d(handle, a, ev, error);
#endif		
	}

/*! \brief generic C method for elpa_eigenvalues_float
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       float pointer to matrix a in GPU memory
 *  \param  ev      on return: float pointer to eigenvalues in GPU memory
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_eigenvalues_float(elpa_t handle, float *a, float *ev, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_eigenvalues_d_ptr_f(handle, a, ev, error);
		}
	else {
		elpa_eigenvalues_a_h_a_f(handle, a, ev, error);
		}	
#else
   elpa_eigenvalues_a_h_a_f(handle, a, ev, error);
#endif		
	}

/*! \brief generic C method for elpa_eigenvalues_double_complex
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       double complex pointer to matrix a in GPU memory
 *  \param  ev      on return: double pointer to eigenvalues in GPU memory
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_eigenvalues_double_complex(elpa_t handle, double complex *a, double *ev, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_eigenvalues_d_ptr_dc(handle, a, ev, error);
		}
	else {
		elpa_eigenvalues_a_h_a_dc(handle, a, ev, error);
		}	
#else
   elpa_eigenvalues_a_h_a_dc(handle, a, ev, error);
#endif		
	}

/*! \brief generic C method for elpa_eigenvalues_float_complex
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       float complex pointer to matrix a in GPU memory
 *  \param  ev      on return: float pointer to eigenvalues in GPU memory
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */
 
void elpa_eigenvalues_float_complex(elpa_t handle, float complex *a, float *ev, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_eigenvalues_d_ptr_fc(handle, a, ev, error);
		}
	else {
		elpa_eigenvalues_a_h_a_fc(handle, a, ev, error);
		}	
#else
   elpa_eigenvalues_a_h_a_fc(handle, a, ev, error);
#endif		
	}

#ifdef HAVE_SKEWSYMMETRIC
/*! \brief generic C method for elpa_skew_eigenvalues_double
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       double pointer to matrix a
 *  \param  ev      on return: double pointer to eigenvalues
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_skew_eigenvalues_double(elpa_t handle, double *a, double *ev, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_skew_eigenvalues_d_ptr_d(handle, a, ev, error);
		}
	else {
		elpa_skew_eigenvalues_a_h_a_d(handle, a, ev, error);
		}	
#else
   elpa_skew_eigenvalues_a_h_a_d(handle, a, ev, error);
#endif		
	}

/*! \brief generic C method for elpa_skew_eigenvalues_float
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       float pointer to matrix a
 *  \param  ev      on return: float pointer to eigenvalues
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_skew_eigenvalues_float(elpa_t handle, float *a, float *ev, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_skew_eigenvalues_d_ptr_f(handle, a, ev, error);
		}
	else {
		elpa_skew_eigenvalues_a_h_a_f(handle, a, ev, error);
		}	
#else
   elpa_skew_eigenvalues_a_h_a_f(handle, a, ev, error);
#endif		
	}
#endif /* HAVE_SKEWSYMMETRIC */

/*! \brief generic C method for elpa_cholesky_double
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       double pointer to matrix a in GPU memory
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_cholesky_double(elpa_t handle, double *a, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_cholesky_d_ptr_d(handle, a, error);
		}
	else {
		elpa_cholesky_a_h_a_d(handle, a, error);
		}	
#else
   elpa_cholesky_a_h_a_d(handle, a, error);
#endif		
	}

/*! \brief generic C method for elpa_cholesky_float
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       float pointer to matrix a in GPU memory
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_cholesky_float(elpa_t handle, float *a, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_cholesky_d_ptr_f(handle, a, error);
		}
	else {
		elpa_cholesky_a_h_a_f(handle, a, error);
		}	
#else
   elpa_cholesky_a_h_a_f(handle, a, error);
#endif		
	}

/*! \brief generic C method for elpa_cholesky_double_complex
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       double complex pointer to matrix a in GPU memory
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_cholesky_double_complex(elpa_t handle, double complex *a, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_cholesky_d_ptr_dc(handle, a, error);
		}
	else {
		elpa_cholesky_a_h_a_dc(handle, a, error);
		}	
#else
   elpa_cholesky_a_h_a_dc(handle, a, error);
#endif		
	}

/*! \brief generic C method for elpa_cholesky_float_complex
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       float complex pointer to matrix a in GPU memory
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_cholesky_float_complex(elpa_t handle, float complex *a, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_cholesky_d_ptr_fc(handle, a, error);
		}
	else {
		elpa_cholesky_a_h_a_fc(handle, a, error);
		}	
#else
   elpa_cholesky_a_h_a_fc(handle, a, error);
#endif		
	}


/*! \brief generic C method for elpa_hermitian_multiply_double
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  uplo_a  descriptor for matrix a
 *  \param  uplo_c  descriptor for matrix c
 *  \param  ncb     int
 *  \param  a       double pointer to matrix a
 *  \param  b       double pointer to matrix b
 *  \param  nrows_b number of rows for matrix b
 *  \param  ncols_b number of cols for matrix b
 *  \param  c       double pointer to matrix c
 *  \param  nrows_c number of rows for matrix c
 *  \param  ncols_c number of cols for matrix c
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */
 
void elpa_hermitian_multiply_double(elpa_t handle, char uplo_a, char uplo_c, int ncb, double *a, double *b, int nrows_b, int ncols_b, double *c, int nrows_c, int ncols_c, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_hermitian_multiply_d_ptr_d(handle, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, c, nrows_c, ncols_c, error);
		}
	else {
		elpa_hermitian_multiply_a_h_a_d(handle, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, c, nrows_c, ncols_c, error);
		}	
#else
   elpa_hermitian_multiply_a_h_a_d(handle, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, c, nrows_c, ncols_c, error);
#endif		
	}
   
/*! \brief generic C method for elpa_hermitian_multiply_float
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  uplo_a  descriptor for matrix a
 *  \param  uplo_c  descriptor for matrix c
 *  \param  ncb     int
 *  \param  a       float pointer to matrix a
 *  \param  b       float pointer to matrix b
 *  \param  nrows_b number of rows for matrix b
 *  \param  ncols_b number of cols for matrix b
 *  \param  c       float pointer to matrix c
 *  \param  nrows_c number of rows for matrix c
 *  \param  ncols_c number of cols for matrix c
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */
 
void elpa_hermitian_multiply_float(elpa_t handle, char uplo_a, char uplo_c, int ncb, float *a, float *b, int nrows_b, int ncols_b, float *c, int nrows_c, int ncols_c, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_hermitian_multiply_d_ptr_f(handle, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, c, nrows_c, ncols_c, error);
		}
	else {
		elpa_hermitian_multiply_a_h_a_f(handle, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, c, nrows_c, ncols_c, error);
		}	
#else
   elpa_hermitian_multiply_a_h_a_f(handle, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, c, nrows_c, ncols_c, error);
#endif		
	}

/*! \brief generic C method for elpa_hermitian_multiply_double_complex
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  uplo_a  descriptor for matrix a
 *  \param  uplo_c  descriptor for matrix c
 *  \param  ncb     int
 *  \param  a       double complex pointer to matrix a
 *  \param  b       double complex pointer to matrix b
 *  \param  nrows_b number of rows for matrix b
 *  \param  ncols_b number of cols for matrix b
 *  \param  c       double complex pointer to matrix c
 *  \param  nrows_c number of rows for matrix c
 *  \param  ncols_c number of cols for matrix c
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */
 
void elpa_hermitian_multiply_double_complex(elpa_t handle, char uplo_a, char uplo_c, int ncb, double complex *a, double complex *b, int nrows_b, int ncols_b, double complex *c, int nrows_c, int ncols_c, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_hermitian_multiply_d_ptr_dc(handle, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, c, nrows_c, ncols_c, error);
		}
	else {
		elpa_hermitian_multiply_a_h_a_dc(handle, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, c, nrows_c, ncols_c, error);
		}	
#else
   elpa_hermitian_multiply_a_h_a_dc(handle, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, c, nrows_c, ncols_c, error);
#endif		
	}
   
/*! \brief generic C method for elpa_hermitian_multiply_float_complex
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  uplo_a  descriptor for matrix a
 *  \param  uplo_c  descriptor for matrix c
 *  \param  ncb     int
 *  \param  a       float complex pointer to matrix a
 *  \param  b       float complex pointer to matrix b
 *  \param  nrows_b number of rows for matrix b
 *  \param  ncols_b number of cols for matrix b
 *  \param  c       float complex pointer to matrix c
 *  \param  nrows_c number of rows for matrix c
 *  \param  ncols_c number of cols for matrix c
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */
 
void elpa_hermitian_multiply_float_complex(elpa_t handle, char uplo_a, char uplo_c, int ncb, float complex *a, float complex *b, int nrows_b, int ncols_b, float complex *c, int nrows_c, int ncols_c, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_hermitian_multiply_d_ptr_fc(handle, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, c, nrows_c, ncols_c, error);
		}
	else {
		elpa_hermitian_multiply_a_h_a_fc(handle, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, c, nrows_c, ncols_c, error);
		}	
#else
   elpa_hermitian_multiply_a_h_a_fc(handle, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, c, nrows_c, ncols_c, error);
#endif		
	}
   
/*! \brief generic C method for elpa_invert_triangular_double
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       double host/device pointer to matrix a in CPU/GPU memory
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_invert_triangular_double(elpa_t handle, double *a, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_invert_trm_d_ptr_d(handle, a, error);
		}
	else {
		elpa_invert_trm_a_h_a_d(handle, a, error);
		}	
#else
   elpa_invert_trm_a_h_a_d(handle, a, error);
#endif		
	}

/*! \brief generic C method for elpa_invert_triangular_float
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       float host/device pointer to matrix a in CPU/GPU memory
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_invert_triangular_float(elpa_t handle, float *a, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_invert_trm_d_ptr_f(handle, a, error);
		}
	else {
		elpa_invert_trm_a_h_a_f(handle, a, error);
		}	
#else
   elpa_invert_trm_a_h_a_f(handle, a, error);
#endif		
	}

/*! \brief generic C method for elpa_invert_triangular_double_complex
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       double complex host/device pointer to matrix a in CPU/GPU memory
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_invert_triangular_double_complex(elpa_t handle, double complex *a, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_invert_trm_d_ptr_dc(handle, a, error);
		}
	else {
		elpa_invert_trm_a_h_a_dc(handle, a, error);
		}	
#else
   elpa_invert_trm_a_h_a_dc(handle, a, error);
#endif		
	}

/*! \brief generic C method for elpa_invert_triangular_float_complex
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       float complex host/device pointer to matrix a in CPU/GPU memory
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */

void elpa_invert_triangular_float_complex(elpa_t handle, float complex *a, int *error)
	{
#if ELPA_WITH_NVIDIA_GPU_VERSION==1 || ELPA_WITH_AMD_GPU_VERSION==1
   void *a_void_ptr = (void*) a;
   int IsDevicePtr = is_device_ptr(a_void_ptr);
	
	if (IsDevicePtr){
		elpa_invert_trm_d_ptr_fc(handle, a, error);
		}
	else {
		elpa_invert_trm_a_h_a_fc(handle, a, error);
		}	
#else
   elpa_invert_trm_a_h_a_fc(handle, a, error);
#endif		
	}
