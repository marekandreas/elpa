//    This file is part of ELPA.
//
//    The ELPA library was originally created by the ELPA consortium, 
//    consisting of the following organizations:
//
//    - Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG), 
//    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
//      Informatik,
//    - Technische Universität München, Lehrstuhl für Informatik mit
//      Schwerpunkt Wissenschaftliches Rechnen , 
//    - Fritz-Haber-Institut, Berlin, Abt. Theorie, 
//    - Max-Plack-Institut für Mathematik in den Naturwissenschaftrn, 
//      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition, 
//      and  
//    - IBM Deutschland GmbH
//
//
//    More information can be found here:
//    http://elpa.rzg.mpg.de/
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
// --------------------------------------------------------------------------------------------------
//
// This file contains the compute intensive kernels for the Householder transformations.
// It should be compiled with the highest possible optimization level.
//
// On Intel Nehalem or Intel Westmere or AMD Magny Cours use -O3 -msse3
// On Intel Sandy Bridge use -O3 -mavx
//
// Copyright of the original code rests with the authors inside the ELPA
// consortium. The copyright of any additional modifications shall rest
// with their original authors, but shall adhere to the licensing terms
// distributed along with the original code in the file "COPYING".
//
// Author: Alexander Heinecke (alexander.heinecke@mytum.de)
// Adapted for building a shared-library by Andreas Marek, RZG (andreas.marek@rzg.mpg.de)
// --------------------------------------------------------------------------------------------------

#include <complex>
#include <x86intrin.h>

#define __forceinline __attribute__((always_inline))

#ifdef __USE_AVX128__
#undef __AVX__
#endif


extern "C" {

//Forward declaration
#ifdef __AVX__
static  __forceinline void hh_trafo_complex_kernel_12_AVX_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq);
static  __forceinline void hh_trafo_complex_kernel_8_AVX_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq);
static  __forceinline void hh_trafo_complex_kernel_4_AVX_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq);

#else
static __forceinline void hh_trafo_complex_kernel_6_SSE_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq);
static __forceinline void hh_trafo_complex_kernel_4_SSE_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq);
static __forceinline void hh_trafo_complex_kernel_2_SSE_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq);
#endif

#if 0
static __forceinline void hh_trafo_complex_kernel_4_C_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq)
{
	std::complex<double> x0;
	std::complex<double> x1;
	std::complex<double> x2;
	std::complex<double> x3;
	std::complex<double> h0;
	std::complex<double> tau0;
	int i=0;

	x0 = q[0];
	x1 = q[1];
	x2 = q[2];
	x3 = q[3];

	for (i = 1; i < nb; i++)
	{
		h0 = conj(hh[i]);
		x0 += (q[(i*ldq)+0] * h0);
		x1 += (q[(i*ldq)+1] * h0);
		x2 += (q[(i*ldq)+2] * h0);
		x3 += (q[(i*ldq)+3] * h0);
	}

	tau0 = hh[0];

	h0 = (-1.0)*tau0;

	x0 *= h0;
	x1 *= h0;
	x2 *= h0;
	x3 *= h0;

	q[0] += x0;
	q[1] += x1;
	q[2] += x2;
	q[3] += x3;

	for (i = 1; i < nb; i++)
	{
		h0 = hh[i];
		q[(i*ldq)+0] += (x0*h0);
		q[(i*ldq)+1] += (x1*h0);
		q[(i*ldq)+2] += (x2*h0);
		q[(i*ldq)+3] += (x3*h0);
	}
}
#endif // if 0

void single_hh_trafo_complex_(std::complex<double>* q, std::complex<double>* hh, int* pnb, int* pnq, int* pldq)
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	//int ldh = *pldh;

#ifdef __AVX__
	for (i = 0; i < nq-8; i+=12)
	{
		hh_trafo_complex_kernel_12_AVX_1hv(&q[i], hh, nb, ldq);
	}
	if (nq-i > 4)
	{
		hh_trafo_complex_kernel_8_AVX_1hv(&q[i], hh, nb, ldq);
	}
	else if (nq-i > 0)
	{
		hh_trafo_complex_kernel_4_AVX_1hv(&q[i], hh, nb, ldq);
	}
#else
	for (i = 0; i < nq-4; i+=6)
	{
		hh_trafo_complex_kernel_6_SSE_1hv(&q[i], hh, nb, ldq);
	}
	if (nq-i > 2)
	{
		hh_trafo_complex_kernel_4_SSE_1hv(&q[i], hh, nb, ldq);
	}
	else if (nq-i > 0)
	{
		hh_trafo_complex_kernel_2_SSE_1hv(&q[i], hh, nb, ldq);
	}
#endif
}

#ifdef __AVX__
 static __forceinline void hh_trafo_complex_kernel_12_AVX_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq)
{
	double* q_dbl = (double*)q;
	double* hh_dbl = (double*)hh;

	__m256d x1, x2, x3, x4, x5, x6;
	__m256d q1, q2, q3, q4, q5, q6;
	__m256d h1_real, h1_imag;
	__m256d tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
	int i=0;

	__m256d sign = (__m256d)_mm256_set_epi64x(0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000);

	x1 = _mm256_load_pd(&q_dbl[0]);
	x2 = _mm256_load_pd(&q_dbl[4]);
	x3 = _mm256_load_pd(&q_dbl[8]);
	x4 = _mm256_load_pd(&q_dbl[12]);
	x5 = _mm256_load_pd(&q_dbl[16]);
	x6 = _mm256_load_pd(&q_dbl[20]);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm256_broadcast_sd(&hh_dbl[i*2]);
		h1_imag = _mm256_broadcast_sd(&hh_dbl[(i*2)+1]);
#ifndef __FMA4__		
		// conjugate
		h1_imag = _mm256_xor_pd(h1_imag, sign);
#endif

		q1 = _mm256_load_pd(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm256_load_pd(&q_dbl[(2*i*ldq)+4]);
		q3 = _mm256_load_pd(&q_dbl[(2*i*ldq)+8]);
		q4 = _mm256_load_pd(&q_dbl[(2*i*ldq)+12]);
		q5 = _mm256_load_pd(&q_dbl[(2*i*ldq)+16]);
		q6 = _mm256_load_pd(&q_dbl[(2*i*ldq)+20]);

		tmp1 = _mm256_mul_pd(h1_imag, q1);
#ifdef __FMA4__
		x1 = _mm256_add_pd(x1, _mm256_msubadd_pd(h1_real, q1, _mm256_shuffle_pd(tmp1, tmp1, 0x5)));
#else
		x1 = _mm256_add_pd(x1, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q1), _mm256_shuffle_pd(tmp1, tmp1, 0x5)));
#endif
		tmp2 = _mm256_mul_pd(h1_imag, q2);
#ifdef __FMA4__
		x2 = _mm256_add_pd(x2, _mm256_msubadd_pd(h1_real, q2, _mm256_shuffle_pd(tmp2, tmp2, 0x5)));
#else
		x2 = _mm256_add_pd(x2, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q2), _mm256_shuffle_pd(tmp2, tmp2, 0x5)));
#endif
		tmp3 = _mm256_mul_pd(h1_imag, q3);
#ifdef __FMA4__
		x3 = _mm256_add_pd(x3, _mm256_msubadd_pd(h1_real, q3, _mm256_shuffle_pd(tmp3, tmp3, 0x5)));
#else
		x3 = _mm256_add_pd(x3, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q3), _mm256_shuffle_pd(tmp3, tmp3, 0x5)));
#endif
		tmp4 = _mm256_mul_pd(h1_imag, q4);
#ifdef __FMA4__
		x4 = _mm256_add_pd(x4, _mm256_msubadd_pd(h1_real, q4, _mm256_shuffle_pd(tmp4, tmp4, 0x5)));
#else
		x4 = _mm256_add_pd(x4, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q4), _mm256_shuffle_pd(tmp4, tmp4, 0x5)));
#endif
		tmp5 = _mm256_mul_pd(h1_imag, q5);
#ifdef __FMA4__
		x5 = _mm256_add_pd(x5, _mm256_msubadd_pd(h1_real, q5, _mm256_shuffle_pd(tmp5, tmp5, 0x5)));
#else
		x5 = _mm256_add_pd(x5, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q5), _mm256_shuffle_pd(tmp5, tmp5, 0x5)));
#endif
		tmp6 = _mm256_mul_pd(h1_imag, q6);
#ifdef __FMA4__
		x6 = _mm256_add_pd(x6, _mm256_msubadd_pd(h1_real, q6, _mm256_shuffle_pd(tmp6, tmp6, 0x5)));
#else
		x6 = _mm256_add_pd(x6, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q6), _mm256_shuffle_pd(tmp6, tmp6, 0x5)));
#endif
	}

	h1_real = _mm256_broadcast_sd(&hh_dbl[0]);
	h1_imag = _mm256_broadcast_sd(&hh_dbl[1]);
	h1_real = _mm256_xor_pd(h1_real, sign);
	h1_imag = _mm256_xor_pd(h1_imag, sign);

	tmp1 = _mm256_mul_pd(h1_imag, x1);
#ifdef __FMA4__
	x1 = _mm256_maddsub_pd(h1_real, x1, _mm256_shuffle_pd(tmp1, tmp1, 0x5));
#else
	x1 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x1), _mm256_shuffle_pd(tmp1, tmp1, 0x5));
#endif
	tmp2 = _mm256_mul_pd(h1_imag, x2);
#ifdef __FMA4__
	x2 = _mm256_maddsub_pd(h1_real, x2, _mm256_shuffle_pd(tmp2, tmp2, 0x5));
#else
	x2 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x2), _mm256_shuffle_pd(tmp2, tmp2, 0x5));
#endif
	tmp3 = _mm256_mul_pd(h1_imag, x3);
#ifdef __FMA4__
	x3 = _mm256_maddsub_pd(h1_real, x3, _mm256_shuffle_pd(tmp3, tmp3, 0x5));
#else
	x3 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x3), _mm256_shuffle_pd(tmp3, tmp3, 0x5));
#endif
	tmp4 = _mm256_mul_pd(h1_imag, x4);
#ifdef __FMA4__
	x4 = _mm256_maddsub_pd(h1_real, x4, _mm256_shuffle_pd(tmp4, tmp4, 0x5));
#else
	x4 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x4), _mm256_shuffle_pd(tmp4, tmp4, 0x5));
#endif
	tmp5 = _mm256_mul_pd(h1_imag, x5);
#ifdef __FMA4__
	x5 = _mm256_maddsub_pd(h1_real, x5, _mm256_shuffle_pd(tmp5, tmp5, 0x5));
#else
	x5 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x5), _mm256_shuffle_pd(tmp5, tmp5, 0x5));
#endif
	tmp6 = _mm256_mul_pd(h1_imag, x6);
#ifdef __FMA4__
	x6 = _mm256_maddsub_pd(h1_real, x6, _mm256_shuffle_pd(tmp6, tmp6, 0x5));
#else
	x6 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x6), _mm256_shuffle_pd(tmp6, tmp6, 0x5));
#endif

	q1 = _mm256_load_pd(&q_dbl[0]);
	q2 = _mm256_load_pd(&q_dbl[4]);
	q3 = _mm256_load_pd(&q_dbl[8]);
	q4 = _mm256_load_pd(&q_dbl[12]);
	q5 = _mm256_load_pd(&q_dbl[16]);
	q6 = _mm256_load_pd(&q_dbl[20]);

	q1 = _mm256_add_pd(q1, x1);
	q2 = _mm256_add_pd(q2, x2);
	q3 = _mm256_add_pd(q3, x3);
	q4 = _mm256_add_pd(q4, x4);
	q5 = _mm256_add_pd(q5, x5);
	q6 = _mm256_add_pd(q6, x6);

	_mm256_store_pd(&q_dbl[0], q1);
	_mm256_store_pd(&q_dbl[4], q2);
	_mm256_store_pd(&q_dbl[8], q3);
	_mm256_store_pd(&q_dbl[12], q4);
	_mm256_store_pd(&q_dbl[16], q5);
	_mm256_store_pd(&q_dbl[20], q6);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm256_broadcast_sd(&hh_dbl[i*2]);
		h1_imag = _mm256_broadcast_sd(&hh_dbl[(i*2)+1]);

		q1 = _mm256_load_pd(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm256_load_pd(&q_dbl[(2*i*ldq)+4]);
		q3 = _mm256_load_pd(&q_dbl[(2*i*ldq)+8]);
		q4 = _mm256_load_pd(&q_dbl[(2*i*ldq)+12]);
		q5 = _mm256_load_pd(&q_dbl[(2*i*ldq)+16]);
		q6 = _mm256_load_pd(&q_dbl[(2*i*ldq)+20]);

		tmp1 = _mm256_mul_pd(h1_imag, x1);
#ifdef __FMA4__
		q1 = _mm256_add_pd(q1, _mm256_maddsub_pd(h1_real, x1, _mm256_shuffle_pd(tmp1, tmp1, 0x5)));
#else
		q1 = _mm256_add_pd(q1, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x1), _mm256_shuffle_pd(tmp1, tmp1, 0x5)));
#endif
		tmp2 = _mm256_mul_pd(h1_imag, x2);
#ifdef __FMA4__
		q2 = _mm256_add_pd(q2, _mm256_maddsub_pd(h1_real, x2, _mm256_shuffle_pd(tmp2, tmp2, 0x5)));
#else
		q2 = _mm256_add_pd(q2, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x2), _mm256_shuffle_pd(tmp2, tmp2, 0x5)));
#endif
		tmp3 = _mm256_mul_pd(h1_imag, x3);
#ifdef __FMA4__	
		q3 = _mm256_add_pd(q3, _mm256_maddsub_pd(h1_real, x3, _mm256_shuffle_pd(tmp3, tmp3, 0x5)));
#else
		q3 = _mm256_add_pd(q3, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x3), _mm256_shuffle_pd(tmp3, tmp3, 0x5)));
#endif
		tmp4 = _mm256_mul_pd(h1_imag, x4);
#ifdef __FMA4__
		q4 = _mm256_add_pd(q4, _mm256_maddsub_pd(h1_real, x4, _mm256_shuffle_pd(tmp4, tmp4, 0x5)));
#else
		q4 = _mm256_add_pd(q4, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x4), _mm256_shuffle_pd(tmp4, tmp4, 0x5)));
#endif
		tmp5 = _mm256_mul_pd(h1_imag, x5);
#ifdef __FMA4__
		q5 = _mm256_add_pd(q5, _mm256_maddsub_pd(h1_real, x5, _mm256_shuffle_pd(tmp5, tmp5, 0x5)));
#else
		q5 = _mm256_add_pd(q5, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x5), _mm256_shuffle_pd(tmp5, tmp5, 0x5)));
#endif
		tmp6 = _mm256_mul_pd(h1_imag, x6);
#ifdef __FMA4__
		q6 = _mm256_add_pd(q6, _mm256_maddsub_pd(h1_real, x6, _mm256_shuffle_pd(tmp6, tmp6, 0x5)));
#else
		q6 = _mm256_add_pd(q6, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x6), _mm256_shuffle_pd(tmp6, tmp6, 0x5)));
#endif

		_mm256_store_pd(&q_dbl[(2*i*ldq)+0], q1);
		_mm256_store_pd(&q_dbl[(2*i*ldq)+4], q2);
		_mm256_store_pd(&q_dbl[(2*i*ldq)+8], q3);
		_mm256_store_pd(&q_dbl[(2*i*ldq)+12], q4);
		_mm256_store_pd(&q_dbl[(2*i*ldq)+16], q5);
		_mm256_store_pd(&q_dbl[(2*i*ldq)+20], q6);
	}
}

static __forceinline void hh_trafo_complex_kernel_8_AVX_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq)
{
	double* q_dbl = (double*)q;
	double* hh_dbl = (double*)hh;

	__m256d x1, x2, x3, x4;
	__m256d q1, q2, q3, q4;
	__m256d h1_real, h1_imag;
	__m256d tmp1, tmp2, tmp3, tmp4;
	int i=0;

	__m256d sign = (__m256d)_mm256_set_epi64x(0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000);

	x1 = _mm256_load_pd(&q_dbl[0]);
	x2 = _mm256_load_pd(&q_dbl[4]);
	x3 = _mm256_load_pd(&q_dbl[8]);
	x4 = _mm256_load_pd(&q_dbl[12]);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm256_broadcast_sd(&hh_dbl[i*2]);
		h1_imag = _mm256_broadcast_sd(&hh_dbl[(i*2)+1]);
#ifndef __FMA4__		
		// conjugate
		h1_imag = _mm256_xor_pd(h1_imag, sign);
#endif

		q1 = _mm256_load_pd(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm256_load_pd(&q_dbl[(2*i*ldq)+4]);
		q3 = _mm256_load_pd(&q_dbl[(2*i*ldq)+8]);
		q4 = _mm256_load_pd(&q_dbl[(2*i*ldq)+12]);

		tmp1 = _mm256_mul_pd(h1_imag, q1);
#ifdef __FMA4__
		x1 = _mm256_add_pd(x1, _mm256_msubadd_pd(h1_real, q1, _mm256_shuffle_pd(tmp1, tmp1, 0x5)));
#else
		x1 = _mm256_add_pd(x1, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q1), _mm256_shuffle_pd(tmp1, tmp1, 0x5)));
#endif
		tmp2 = _mm256_mul_pd(h1_imag, q2);
#ifdef __FMA4__
		x2 = _mm256_add_pd(x2, _mm256_msubadd_pd(h1_real, q2, _mm256_shuffle_pd(tmp2, tmp2, 0x5)));
#else
		x2 = _mm256_add_pd(x2, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q2), _mm256_shuffle_pd(tmp2, tmp2, 0x5)));
#endif
		tmp3 = _mm256_mul_pd(h1_imag, q3);
#ifdef __FMA4__
		x3 = _mm256_add_pd(x3, _mm256_msubadd_pd(h1_real, q3, _mm256_shuffle_pd(tmp3, tmp3, 0x5)));
#else
		x3 = _mm256_add_pd(x3, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q3), _mm256_shuffle_pd(tmp3, tmp3, 0x5)));
#endif
		tmp4 = _mm256_mul_pd(h1_imag, q4);
#ifdef __FMA4__
		x4 = _mm256_add_pd(x4, _mm256_msubadd_pd(h1_real, q4, _mm256_shuffle_pd(tmp4, tmp4, 0x5)));
#else
		x4 = _mm256_add_pd(x4, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q4), _mm256_shuffle_pd(tmp4, tmp4, 0x5)));
#endif
	}

	h1_real = _mm256_broadcast_sd(&hh_dbl[0]);
	h1_imag = _mm256_broadcast_sd(&hh_dbl[1]);
	h1_real = _mm256_xor_pd(h1_real, sign);
	h1_imag = _mm256_xor_pd(h1_imag, sign);

	tmp1 = _mm256_mul_pd(h1_imag, x1);
#ifdef __FMA4__
	x1 = _mm256_maddsub_pd(h1_real, x1, _mm256_shuffle_pd(tmp1, tmp1, 0x5));
#else
	x1 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x1), _mm256_shuffle_pd(tmp1, tmp1, 0x5));
#endif
	tmp2 = _mm256_mul_pd(h1_imag, x2);
#ifdef __FMA4__
	x2 = _mm256_maddsub_pd(h1_real, x2, _mm256_shuffle_pd(tmp2, tmp2, 0x5));
#else
	x2 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x2), _mm256_shuffle_pd(tmp2, tmp2, 0x5));
#endif
	tmp3 = _mm256_mul_pd(h1_imag, x3);
#ifdef __FMA4__
	x3 = _mm256_maddsub_pd(h1_real, x3, _mm256_shuffle_pd(tmp3, tmp3, 0x5));
#else
	x3 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x3), _mm256_shuffle_pd(tmp3, tmp3, 0x5));
#endif
	tmp4 = _mm256_mul_pd(h1_imag, x4);
#ifdef __FMA4__
	x4 = _mm256_maddsub_pd(h1_real, x4, _mm256_shuffle_pd(tmp4, tmp4, 0x5));
#else
	x4 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x4), _mm256_shuffle_pd(tmp4, tmp4, 0x5));
#endif

	q1 = _mm256_load_pd(&q_dbl[0]);
	q2 = _mm256_load_pd(&q_dbl[4]);
	q3 = _mm256_load_pd(&q_dbl[8]);
	q4 = _mm256_load_pd(&q_dbl[12]);

	q1 = _mm256_add_pd(q1, x1);
	q2 = _mm256_add_pd(q2, x2);
	q3 = _mm256_add_pd(q3, x3);
	q4 = _mm256_add_pd(q4, x4);

	_mm256_store_pd(&q_dbl[0], q1);
	_mm256_store_pd(&q_dbl[4], q2);
	_mm256_store_pd(&q_dbl[8], q3);
	_mm256_store_pd(&q_dbl[12], q4);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm256_broadcast_sd(&hh_dbl[i*2]);
		h1_imag = _mm256_broadcast_sd(&hh_dbl[(i*2)+1]);

		q1 = _mm256_load_pd(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm256_load_pd(&q_dbl[(2*i*ldq)+4]);
		q3 = _mm256_load_pd(&q_dbl[(2*i*ldq)+8]);
		q4 = _mm256_load_pd(&q_dbl[(2*i*ldq)+12]);

		tmp1 = _mm256_mul_pd(h1_imag, x1);
#ifdef __FMA4__
		q1 = _mm256_add_pd(q1, _mm256_maddsub_pd(h1_real, x1, _mm256_shuffle_pd(tmp1, tmp1, 0x5)));
#else
		q1 = _mm256_add_pd(q1, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x1), _mm256_shuffle_pd(tmp1, tmp1, 0x5)));
#endif
		tmp2 = _mm256_mul_pd(h1_imag, x2);
#ifdef __FMA4__
		q2 = _mm256_add_pd(q2, _mm256_maddsub_pd(h1_real, x2, _mm256_shuffle_pd(tmp2, tmp2, 0x5)));
#else	
		q2 = _mm256_add_pd(q2, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x2), _mm256_shuffle_pd(tmp2, tmp2, 0x5)));
#endif
		tmp3 = _mm256_mul_pd(h1_imag, x3);
#ifdef __FMA4__
		q3 = _mm256_add_pd(q3, _mm256_maddsub_pd(h1_real, x3, _mm256_shuffle_pd(tmp3, tmp3, 0x5)));
#else		
		q3 = _mm256_add_pd(q3, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x3), _mm256_shuffle_pd(tmp3, tmp3, 0x5)));
#endif
		tmp4 = _mm256_mul_pd(h1_imag, x4);
#ifdef __FMA4__
		q4 = _mm256_add_pd(q4, _mm256_maddsub_pd(h1_real, x4, _mm256_shuffle_pd(tmp4, tmp4, 0x5)));
#else
		q4 = _mm256_add_pd(q4, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x4), _mm256_shuffle_pd(tmp4, tmp4, 0x5)));
#endif

		_mm256_store_pd(&q_dbl[(2*i*ldq)+0], q1);
		_mm256_store_pd(&q_dbl[(2*i*ldq)+4], q2);
		_mm256_store_pd(&q_dbl[(2*i*ldq)+8], q3);
		_mm256_store_pd(&q_dbl[(2*i*ldq)+12], q4);
	}
}

static __forceinline void hh_trafo_complex_kernel_4_AVX_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq)
{
	double* q_dbl = (double*)q;
	double* hh_dbl = (double*)hh;

	__m256d x1, x2;
	__m256d q1, q2;
	__m256d h1_real, h1_imag;
	__m256d tmp1, tmp2;
	int i=0;

	__m256d sign = (__m256d)_mm256_set_epi64x(0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000);

	x1 = _mm256_load_pd(&q_dbl[0]);
	x2 = _mm256_load_pd(&q_dbl[4]);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm256_broadcast_sd(&hh_dbl[i*2]);
		h1_imag = _mm256_broadcast_sd(&hh_dbl[(i*2)+1]);
#ifndef __FMA4__
		// conjugate
		h1_imag = _mm256_xor_pd(h1_imag, sign);
#endif

		q1 = _mm256_load_pd(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm256_load_pd(&q_dbl[(2*i*ldq)+4]);

		tmp1 = _mm256_mul_pd(h1_imag, q1);
#ifdef __FMA4__
		x1 = _mm256_add_pd(x1, _mm256_msubadd_pd(h1_real, q1, _mm256_shuffle_pd(tmp1, tmp1, 0x5)));
#else
		x1 = _mm256_add_pd(x1, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q1), _mm256_shuffle_pd(tmp1, tmp1, 0x5)));
#endif
		tmp2 = _mm256_mul_pd(h1_imag, q2);
#ifdef __FMA4__
		x2 = _mm256_add_pd(x2, _mm256_msubadd_pd(h1_real, q2, _mm256_shuffle_pd(tmp2, tmp2, 0x5)));
#else
		x2 = _mm256_add_pd(x2, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q2), _mm256_shuffle_pd(tmp2, tmp2, 0x5)));
#endif
	}

	h1_real = _mm256_broadcast_sd(&hh_dbl[0]);
	h1_imag = _mm256_broadcast_sd(&hh_dbl[1]);
	h1_real = _mm256_xor_pd(h1_real, sign);
	h1_imag = _mm256_xor_pd(h1_imag, sign);

	tmp1 = _mm256_mul_pd(h1_imag, x1);
#ifdef __FMA4__
	x1 = _mm256_maddsub_pd(h1_real, x1, _mm256_shuffle_pd(tmp1, tmp1, 0x5));
#else
	x1 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x1), _mm256_shuffle_pd(tmp1, tmp1, 0x5));
#endif
	tmp2 = _mm256_mul_pd(h1_imag, x2);
#ifdef __FMA4__
	x2 = _mm256_maddsub_pd(h1_real, x2, _mm256_shuffle_pd(tmp2, tmp2, 0x5));
#else
	x2 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x2), _mm256_shuffle_pd(tmp2, tmp2, 0x5));
#endif

	q1 = _mm256_load_pd(&q_dbl[0]);
	q2 = _mm256_load_pd(&q_dbl[4]);

	q1 = _mm256_add_pd(q1, x1);
	q2 = _mm256_add_pd(q2, x2);

	_mm256_store_pd(&q_dbl[0], q1);
	_mm256_store_pd(&q_dbl[4], q2);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm256_broadcast_sd(&hh_dbl[i*2]);
		h1_imag = _mm256_broadcast_sd(&hh_dbl[(i*2)+1]);

		q1 = _mm256_load_pd(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm256_load_pd(&q_dbl[(2*i*ldq)+4]);

		tmp1 = _mm256_mul_pd(h1_imag, x1);
#ifdef __FMA4__
		q1 = _mm256_add_pd(q1, _mm256_maddsub_pd(h1_real, x1, _mm256_shuffle_pd(tmp1, tmp1, 0x5)));
#else
		q1 = _mm256_add_pd(q1, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x1), _mm256_shuffle_pd(tmp1, tmp1, 0x5)));
#endif
		tmp2 = _mm256_mul_pd(h1_imag, x2);
#ifdef __FMA4__
		q2 = _mm256_add_pd(q2, _mm256_maddsub_pd(h1_real, x2, _mm256_shuffle_pd(tmp2, tmp2, 0x5)));
#else
		q2 = _mm256_add_pd(q2, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x2), _mm256_shuffle_pd(tmp2, tmp2, 0x5)));
#endif

		_mm256_store_pd(&q_dbl[(2*i*ldq)+0], q1);
		_mm256_store_pd(&q_dbl[(2*i*ldq)+4], q2);
	}
}

#else
static __forceinline void hh_trafo_complex_kernel_6_SSE_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq)
{
	double* q_dbl = (double*)q;
	double* hh_dbl = (double*)hh;

	__m128d x1, x2, x3, x4, x5, x6;
	__m128d q1, q2, q3, q4, q5, q6;
	__m128d h1_real, h1_imag;
	__m128d tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
	int i=0;

	__m128d sign = (__m128d)_mm_set_epi64x(0x8000000000000000, 0x8000000000000000);

	x1 = _mm_load_pd(&q_dbl[0]);
	x2 = _mm_load_pd(&q_dbl[2]);
	x3 = _mm_load_pd(&q_dbl[4]);
	x4 = _mm_load_pd(&q_dbl[6]);
	x5 = _mm_load_pd(&q_dbl[8]);
	x6 = _mm_load_pd(&q_dbl[10]);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm_loaddup_pd(&hh_dbl[i*2]);
		h1_imag = _mm_loaddup_pd(&hh_dbl[(i*2)+1]);
#ifndef __FMA4__		
		// conjugate
		h1_imag = _mm_xor_pd(h1_imag, sign);
#endif

		q1 = _mm_load_pd(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm_load_pd(&q_dbl[(2*i*ldq)+2]);
		q3 = _mm_load_pd(&q_dbl[(2*i*ldq)+4]);
		q4 = _mm_load_pd(&q_dbl[(2*i*ldq)+6]);
		q5 = _mm_load_pd(&q_dbl[(2*i*ldq)+8]);
		q6 = _mm_load_pd(&q_dbl[(2*i*ldq)+10]);

		tmp1 = _mm_mul_pd(h1_imag, q1);
#ifdef __FMA4__
		x1 = _mm_add_pd(x1, _mm_msubadd_pd(h1_real, q1, _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#else
		x1 = _mm_add_pd(x1, _mm_addsub_pd( _mm_mul_pd(h1_real, q1), _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#endif
		tmp2 = _mm_mul_pd(h1_imag, q2);
#ifdef __FMA4__
		x2 = _mm_add_pd(x2, _mm_msubadd_pd(h1_real, q2, _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#else
		x2 = _mm_add_pd(x2, _mm_addsub_pd( _mm_mul_pd(h1_real, q2), _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#endif
		tmp3 = _mm_mul_pd(h1_imag, q3);
#ifdef __FMA4__
		x3 = _mm_add_pd(x3, _mm_msubadd_pd(h1_real, q3, _mm_shuffle_pd(tmp3, tmp3, _MM_SHUFFLE2(0,1))));
#else
		x3 = _mm_add_pd(x3, _mm_addsub_pd( _mm_mul_pd(h1_real, q3), _mm_shuffle_pd(tmp3, tmp3, _MM_SHUFFLE2(0,1))));
#endif
		tmp4 = _mm_mul_pd(h1_imag, q4);
#ifdef __FMA4__
		x4 = _mm_add_pd(x4, _mm_msubadd_pd(h1_real, q4, _mm_shuffle_pd(tmp4, tmp4, _MM_SHUFFLE2(0,1))));
#else
		x4 = _mm_add_pd(x4, _mm_addsub_pd( _mm_mul_pd(h1_real, q4), _mm_shuffle_pd(tmp4, tmp4, _MM_SHUFFLE2(0,1))));
#endif
		tmp5 = _mm_mul_pd(h1_imag, q5);
#ifdef __FMA4__
		x5 = _mm_add_pd(x5, _mm_msubadd_pd(h1_real, q5, _mm_shuffle_pd(tmp5, tmp5, _MM_SHUFFLE2(0,1))));
#else
		x5 = _mm_add_pd(x5, _mm_addsub_pd( _mm_mul_pd(h1_real, q5), _mm_shuffle_pd(tmp5, tmp5, _MM_SHUFFLE2(0,1))));
#endif
		tmp6 = _mm_mul_pd(h1_imag, q6);
#ifdef __FMA4__
		x6 = _mm_add_pd(x6, _mm_msubadd_pd(h1_real, q6, _mm_shuffle_pd(tmp6, tmp6, _MM_SHUFFLE2(0,1))));
#else
		x6 = _mm_add_pd(x6, _mm_addsub_pd( _mm_mul_pd(h1_real, q6), _mm_shuffle_pd(tmp6, tmp6, _MM_SHUFFLE2(0,1))));
#endif
	}

	h1_real = _mm_loaddup_pd(&hh_dbl[0]);
	h1_imag = _mm_loaddup_pd(&hh_dbl[1]);
	h1_real = _mm_xor_pd(h1_real, sign);
	h1_imag = _mm_xor_pd(h1_imag, sign);

	tmp1 = _mm_mul_pd(h1_imag, x1);
#ifdef __FMA4__	
	x1 = _mm_maddsub_pd(h1_real, x1, _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1)));
#else
	x1 = _mm_addsub_pd( _mm_mul_pd(h1_real, x1), _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1)));
#endif
	tmp2 = _mm_mul_pd(h1_imag, x2);
#ifdef __FMA4__
	x2 = _mm_maddsub_pd(h1_real, x2, _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1)));
#else
	x2 = _mm_addsub_pd( _mm_mul_pd(h1_real, x2), _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1)));
#endif
	tmp3 = _mm_mul_pd(h1_imag, x3);
#ifdef __FMA4__
	x3 = _mm_maddsub_pd(h1_real, x3, _mm_shuffle_pd(tmp3, tmp3, _MM_SHUFFLE2(0,1)));
#else
	x3 = _mm_addsub_pd( _mm_mul_pd(h1_real, x3), _mm_shuffle_pd(tmp3, tmp3, _MM_SHUFFLE2(0,1)));
#endif
	tmp4 = _mm_mul_pd(h1_imag, x4);
#ifdef __FMA4__
	x4 = _mm_maddsub_pd(h1_real, x4, _mm_shuffle_pd(tmp4, tmp4, _MM_SHUFFLE2(0,1)));
#else
	x4 = _mm_addsub_pd( _mm_mul_pd(h1_real, x4), _mm_shuffle_pd(tmp4, tmp4, _MM_SHUFFLE2(0,1)));
#endif
	tmp5 = _mm_mul_pd(h1_imag, x5);
#ifdef __FMA4__
	x5 = _mm_maddsub_pd(h1_real, x5, _mm_shuffle_pd(tmp5, tmp5, _MM_SHUFFLE2(0,1)));
#else
	x5 = _mm_addsub_pd( _mm_mul_pd(h1_real, x5), _mm_shuffle_pd(tmp5, tmp5, _MM_SHUFFLE2(0,1)));
#endif
	tmp6 = _mm_mul_pd(h1_imag, x6);
#ifdef __FMA4__
	x6 = _mm_maddsub_pd(h1_real, x6, _mm_shuffle_pd(tmp6, tmp6, _MM_SHUFFLE2(0,1)));
#else
	x6 = _mm_addsub_pd( _mm_mul_pd(h1_real, x6), _mm_shuffle_pd(tmp6, tmp6, _MM_SHUFFLE2(0,1)));
#endif

	q1 = _mm_load_pd(&q_dbl[0]);
	q2 = _mm_load_pd(&q_dbl[2]);
	q3 = _mm_load_pd(&q_dbl[4]);
	q4 = _mm_load_pd(&q_dbl[6]);
	q5 = _mm_load_pd(&q_dbl[8]);
	q6 = _mm_load_pd(&q_dbl[10]);

	q1 = _mm_add_pd(q1, x1);
	q2 = _mm_add_pd(q2, x2);
	q3 = _mm_add_pd(q3, x3);
	q4 = _mm_add_pd(q4, x4);
	q5 = _mm_add_pd(q5, x5);
	q6 = _mm_add_pd(q6, x6);

	_mm_store_pd(&q_dbl[0], q1);
	_mm_store_pd(&q_dbl[2], q2);
	_mm_store_pd(&q_dbl[4], q3);
	_mm_store_pd(&q_dbl[6], q4);
	_mm_store_pd(&q_dbl[8], q5);
	_mm_store_pd(&q_dbl[10], q6);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm_loaddup_pd(&hh_dbl[i*2]);
		h1_imag = _mm_loaddup_pd(&hh_dbl[(i*2)+1]);

		q1 = _mm_load_pd(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm_load_pd(&q_dbl[(2*i*ldq)+2]);
		q3 = _mm_load_pd(&q_dbl[(2*i*ldq)+4]);
		q4 = _mm_load_pd(&q_dbl[(2*i*ldq)+6]);
		q5 = _mm_load_pd(&q_dbl[(2*i*ldq)+8]);
		q6 = _mm_load_pd(&q_dbl[(2*i*ldq)+10]);

		tmp1 = _mm_mul_pd(h1_imag, x1);
#ifdef __FMA4__
		q1 = _mm_add_pd(q1, _mm_maddsub_pd(h1_real, x1, _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#else
		q1 = _mm_add_pd(q1, _mm_addsub_pd( _mm_mul_pd(h1_real, x1), _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#endif
		tmp2 = _mm_mul_pd(h1_imag, x2);
#ifdef __FMA4__
		q2 = _mm_add_pd(q2, _mm_maddsub_pd(h1_real, x2, _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#else
		q2 = _mm_add_pd(q2, _mm_addsub_pd( _mm_mul_pd(h1_real, x2), _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#endif
		tmp3 = _mm_mul_pd(h1_imag, x3);
#ifdef __FMA4__
		q3 = _mm_add_pd(q3, _mm_maddsub_pd(h1_real, x3, _mm_shuffle_pd(tmp3, tmp3, _MM_SHUFFLE2(0,1))));
#else
		q3 = _mm_add_pd(q3, _mm_addsub_pd( _mm_mul_pd(h1_real, x3), _mm_shuffle_pd(tmp3, tmp3, _MM_SHUFFLE2(0,1))));
#endif
		tmp4 = _mm_mul_pd(h1_imag, x4);
#ifdef __FMA4__
		q4 = _mm_add_pd(q4, _mm_maddsub_pd(h1_real, x4, _mm_shuffle_pd(tmp4, tmp4, _MM_SHUFFLE2(0,1))));
#else
		q4 = _mm_add_pd(q4, _mm_addsub_pd( _mm_mul_pd(h1_real, x4), _mm_shuffle_pd(tmp4, tmp4, _MM_SHUFFLE2(0,1))));
#endif
		tmp5 = _mm_mul_pd(h1_imag, x5);
#ifdef __FMA4__
		q5 = _mm_add_pd(q5, _mm_maddsub_pd(h1_real, x5, _mm_shuffle_pd(tmp5, tmp5, _MM_SHUFFLE2(0,1))));
#else
		q5 = _mm_add_pd(q5, _mm_addsub_pd( _mm_mul_pd(h1_real, x5), _mm_shuffle_pd(tmp5, tmp5, _MM_SHUFFLE2(0,1))));
#endif
		tmp6 = _mm_mul_pd(h1_imag, x6);
#ifdef __FMA4__
		q6 = _mm_add_pd(q6, _mm_maddsub_pd(h1_real, x6, _mm_shuffle_pd(tmp6, tmp6, _MM_SHUFFLE2(0,1))));
#else
		q6 = _mm_add_pd(q6, _mm_addsub_pd( _mm_mul_pd(h1_real, x6), _mm_shuffle_pd(tmp6, tmp6, _MM_SHUFFLE2(0,1))));
#endif

		_mm_store_pd(&q_dbl[(2*i*ldq)+0], q1);
		_mm_store_pd(&q_dbl[(2*i*ldq)+2], q2);
		_mm_store_pd(&q_dbl[(2*i*ldq)+4], q3);
		_mm_store_pd(&q_dbl[(2*i*ldq)+6], q4);
		_mm_store_pd(&q_dbl[(2*i*ldq)+8], q5);
		_mm_store_pd(&q_dbl[(2*i*ldq)+10], q6);
	}
}

static __forceinline void hh_trafo_complex_kernel_4_SSE_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq)
{
	double* q_dbl = (double*)q;
	double* hh_dbl = (double*)hh;

	__m128d x1, x2, x3, x4;
	__m128d q1, q2, q3, q4;
	__m128d h1_real, h1_imag;
	__m128d tmp1, tmp2, tmp3, tmp4;
	int i=0;

	__m128d sign = (__m128d)_mm_set_epi64x(0x8000000000000000, 0x8000000000000000);

	x1 = _mm_load_pd(&q_dbl[0]);
	x2 = _mm_load_pd(&q_dbl[2]);
	x3 = _mm_load_pd(&q_dbl[4]);
	x4 = _mm_load_pd(&q_dbl[6]);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm_loaddup_pd(&hh_dbl[i*2]);
		h1_imag = _mm_loaddup_pd(&hh_dbl[(i*2)+1]);
#ifndef __FMA4__
		// conjugate
		h1_imag = _mm_xor_pd(h1_imag, sign);
#endif

		q1 = _mm_load_pd(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm_load_pd(&q_dbl[(2*i*ldq)+2]);
		q3 = _mm_load_pd(&q_dbl[(2*i*ldq)+4]);
		q4 = _mm_load_pd(&q_dbl[(2*i*ldq)+6]);

		tmp1 = _mm_mul_pd(h1_imag, q1);
#ifdef __FMA4__
		x1 = _mm_add_pd(x1, _mm_msubadd_pd(h1_real, q1, _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#else
		x1 = _mm_add_pd(x1, _mm_addsub_pd( _mm_mul_pd(h1_real, q1), _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#endif
		tmp2 = _mm_mul_pd(h1_imag, q2);
#ifdef __FMA4__
		x2 = _mm_add_pd(x2, _mm_msubadd_pd(h1_real, q2, _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#else
		x2 = _mm_add_pd(x2, _mm_addsub_pd( _mm_mul_pd(h1_real, q2), _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#endif
		tmp3 = _mm_mul_pd(h1_imag, q3);
#ifdef __FMA4__
		x3 = _mm_add_pd(x3, _mm_msubadd_pd(h1_real, q3, _mm_shuffle_pd(tmp3, tmp3, _MM_SHUFFLE2(0,1))));
#else
		x3 = _mm_add_pd(x3, _mm_addsub_pd( _mm_mul_pd(h1_real, q3), _mm_shuffle_pd(tmp3, tmp3, _MM_SHUFFLE2(0,1))));
#endif
		tmp4 = _mm_mul_pd(h1_imag, q4);
#ifdef __FMA4__
		x4 = _mm_add_pd(x4, _mm_msubadd_pd(h1_real, q4, _mm_shuffle_pd(tmp4, tmp4, _MM_SHUFFLE2(0,1))));
#else
		x4 = _mm_add_pd(x4, _mm_addsub_pd( _mm_mul_pd(h1_real, q4), _mm_shuffle_pd(tmp4, tmp4, _MM_SHUFFLE2(0,1))));
#endif
	}

	h1_real = _mm_loaddup_pd(&hh_dbl[0]);
	h1_imag = _mm_loaddup_pd(&hh_dbl[1]);
	h1_real = _mm_xor_pd(h1_real, sign);
	h1_imag = _mm_xor_pd(h1_imag, sign);

	tmp1 = _mm_mul_pd(h1_imag, x1);
#ifdef __FMA4__
	x1 = _mm_maddsub_pd(h1_real, x1, _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1)));
#else
	x1 = _mm_addsub_pd( _mm_mul_pd(h1_real, x1), _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1)));
#endif
	tmp2 = _mm_mul_pd(h1_imag, x2);
#ifdef __FMA4__
	x2 = _mm_maddsub_pd(h1_real, x2, _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1)));
#else
	x2 = _mm_addsub_pd( _mm_mul_pd(h1_real, x2), _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1)));
#endif
	tmp3 = _mm_mul_pd(h1_imag, x3);
#ifdef __FMA4__
	x3 = _mm_maddsub_pd(h1_real, x3, _mm_shuffle_pd(tmp3, tmp3, _MM_SHUFFLE2(0,1)));
#else
	x3 = _mm_addsub_pd( _mm_mul_pd(h1_real, x3), _mm_shuffle_pd(tmp3, tmp3, _MM_SHUFFLE2(0,1)));
#endif
	tmp4 = _mm_mul_pd(h1_imag, x4);
#ifdef __FMA4__
	x4 = _mm_maddsub_pd(h1_real, x4, _mm_shuffle_pd(tmp4, tmp4, _MM_SHUFFLE2(0,1)));
#else
	x4 = _mm_addsub_pd( _mm_mul_pd(h1_real, x4), _mm_shuffle_pd(tmp4, tmp4, _MM_SHUFFLE2(0,1)));
#endif

	q1 = _mm_load_pd(&q_dbl[0]);
	q2 = _mm_load_pd(&q_dbl[2]);
	q3 = _mm_load_pd(&q_dbl[4]);
	q4 = _mm_load_pd(&q_dbl[6]);

	q1 = _mm_add_pd(q1, x1);
	q2 = _mm_add_pd(q2, x2);
	q3 = _mm_add_pd(q3, x3);
	q4 = _mm_add_pd(q4, x4);

	_mm_store_pd(&q_dbl[0], q1);
	_mm_store_pd(&q_dbl[2], q2);
	_mm_store_pd(&q_dbl[4], q3);
	_mm_store_pd(&q_dbl[6], q4);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm_loaddup_pd(&hh_dbl[i*2]);
		h1_imag = _mm_loaddup_pd(&hh_dbl[(i*2)+1]);

		q1 = _mm_load_pd(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm_load_pd(&q_dbl[(2*i*ldq)+2]);
		q3 = _mm_load_pd(&q_dbl[(2*i*ldq)+4]);
		q4 = _mm_load_pd(&q_dbl[(2*i*ldq)+6]);

		tmp1 = _mm_mul_pd(h1_imag, x1);
#ifdef __FMA4__
		q1 = _mm_add_pd(q1, _mm_maddsub_pd(h1_real, x1, _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#else
		q1 = _mm_add_pd(q1, _mm_addsub_pd( _mm_mul_pd(h1_real, x1), _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#endif
		tmp2 = _mm_mul_pd(h1_imag, x2);
#ifdef __FMA4__
		q2 = _mm_add_pd(q2, _mm_maddsub_pd(h1_real, x2, _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#else
		q2 = _mm_add_pd(q2, _mm_addsub_pd( _mm_mul_pd(h1_real, x2), _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#endif
		tmp3 = _mm_mul_pd(h1_imag, x3);
#ifdef __FMA4__
		q3 = _mm_add_pd(q3, _mm_maddsub_pd(h1_real, x3, _mm_shuffle_pd(tmp3, tmp3, _MM_SHUFFLE2(0,1))));
#else
		q3 = _mm_add_pd(q3, _mm_addsub_pd( _mm_mul_pd(h1_real, x3), _mm_shuffle_pd(tmp3, tmp3, _MM_SHUFFLE2(0,1))));
#endif
		tmp4 = _mm_mul_pd(h1_imag, x4);
#ifdef __FMA4__
		q4 = _mm_add_pd(q4, _mm_maddsub_pd(h1_real, x4, _mm_shuffle_pd(tmp4, tmp4, _MM_SHUFFLE2(0,1))));
#else
		q4 = _mm_add_pd(q4, _mm_addsub_pd( _mm_mul_pd(h1_real, x4), _mm_shuffle_pd(tmp4, tmp4, _MM_SHUFFLE2(0,1))));
#endif

		_mm_store_pd(&q_dbl[(2*i*ldq)+0], q1);
		_mm_store_pd(&q_dbl[(2*i*ldq)+2], q2);
		_mm_store_pd(&q_dbl[(2*i*ldq)+4], q3);
		_mm_store_pd(&q_dbl[(2*i*ldq)+6], q4);
	}
}

static __forceinline void hh_trafo_complex_kernel_2_SSE_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq)
{
	double* q_dbl = (double*)q;
	double* hh_dbl = (double*)hh;

	__m128d x1, x2;
	__m128d q1, q2;
	__m128d h1_real, h1_imag;
	__m128d tmp1, tmp2;
	int i=0;

	__m128d sign = (__m128d)_mm_set_epi64x(0x8000000000000000, 0x8000000000000000);

	x1 = _mm_load_pd(&q_dbl[0]);
	x2 = _mm_load_pd(&q_dbl[2]);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm_loaddup_pd(&hh_dbl[i*2]);
		h1_imag = _mm_loaddup_pd(&hh_dbl[(i*2)+1]);
#ifndef __FMA4__	
		// conjugate
		h1_imag = _mm_xor_pd(h1_imag, sign);
#endif

		q1 = _mm_load_pd(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm_load_pd(&q_dbl[(2*i*ldq)+2]);

		tmp1 = _mm_mul_pd(h1_imag, q1);
#ifdef __FMA4__
		x1 = _mm_add_pd(x1, _mm_msubadd_pd(h1_real, q1, _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#else
		x1 = _mm_add_pd(x1, _mm_addsub_pd( _mm_mul_pd(h1_real, q1), _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#endif
		tmp2 = _mm_mul_pd(h1_imag, q2);
#ifdef __FMA4__
		x2 = _mm_add_pd(x2, _mm_msubadd_pd(h1_real, q2, _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#else
		x2 = _mm_add_pd(x2, _mm_addsub_pd( _mm_mul_pd(h1_real, q2), _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#endif
	}

	h1_real = _mm_loaddup_pd(&hh_dbl[0]);
	h1_imag = _mm_loaddup_pd(&hh_dbl[1]);
	h1_real = _mm_xor_pd(h1_real, sign);
	h1_imag = _mm_xor_pd(h1_imag, sign);

	tmp1 = _mm_mul_pd(h1_imag, x1);
#ifdef __FMA4__
	x1 = _mm_maddsub_pd(h1_real, x1, _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1)));
#else
	x1 = _mm_addsub_pd( _mm_mul_pd(h1_real, x1), _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1)));
#endif
	tmp2 = _mm_mul_pd(h1_imag, x2);
#ifdef __FMA4__
	x2 = _mm_maddsub_pd(h1_real, x2, _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1)));
#else
	x2 = _mm_addsub_pd( _mm_mul_pd(h1_real, x2), _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1)));
#endif

	q1 = _mm_load_pd(&q_dbl[0]);
	q2 = _mm_load_pd(&q_dbl[2]);

	q1 = _mm_add_pd(q1, x1);
	q2 = _mm_add_pd(q2, x2);

	_mm_store_pd(&q_dbl[0], q1);
	_mm_store_pd(&q_dbl[2], q2);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm_loaddup_pd(&hh_dbl[i*2]);
		h1_imag = _mm_loaddup_pd(&hh_dbl[(i*2)+1]);

		q1 = _mm_load_pd(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm_load_pd(&q_dbl[(2*i*ldq)+2]);

		tmp1 = _mm_mul_pd(h1_imag, x1);
#ifdef __FMA4__
		q1 = _mm_add_pd(q1, _mm_maddsub_pd(h1_real, x1, _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#else
		q1 = _mm_add_pd(q1, _mm_addsub_pd( _mm_mul_pd(h1_real, x1), _mm_shuffle_pd(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#endif
		tmp2 = _mm_mul_pd(h1_imag, x2);
#ifdef __FMA4__
		q2 = _mm_add_pd(q2, _mm_maddsub_pd(h1_real, x2, _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#else
		q2 = _mm_add_pd(q2, _mm_addsub_pd( _mm_mul_pd(h1_real, x2), _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#endif

		_mm_store_pd(&q_dbl[(2*i*ldq)+0], q1);
		_mm_store_pd(&q_dbl[(2*i*ldq)+2], q2);
	}
}
#endif
} // extern C
