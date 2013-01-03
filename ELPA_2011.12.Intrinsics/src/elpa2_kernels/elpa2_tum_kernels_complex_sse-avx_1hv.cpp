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
// --------------------------------------------------------------------------------------------------

#include <complex>
#include <x86intrin.h>

#define __forceinline __attribute__((always_inline))

#ifdef __USE_AVX128__
#undef __AVX__
#endif

//Forward declaration
#ifdef __AVX__
extern "C" __forceinline void hh_trafo_complex_kernel_4_AVX_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq);
#else
extern "C" __forceinline void hh_trafo_complex_kernel_4_SSE_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq);
#endif

extern "C" void single_hh_trafo_complex_(std::complex<double>* q, std::complex<double>* hh, int* pnb, int* pnq, int* pldq)
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	//int ldh = *pldh;

#ifdef __AVX__
	for (i = 0; i < nq; i+=4)
	{
		hh_trafo_complex_kernel_4_AVX_1hv(&q[i], hh, nb, ldq);
	}
#else
	for (i = 0; i < nq; i+=4)
	{
		hh_trafo_complex_kernel_4_SSE_1hv(&q[i], hh, nb, ldq);
	}
#endif
}

extern "C" __forceinline void hh_trafo_complex_kernel_4_C_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq)
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

#ifdef __AVX__
extern "C" __forceinline void hh_trafo_complex_kernel_4_AVX_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq)
{
	hh_trafo_complex_kernel_4_C_1hv(q, hh, nb, ldq);
}
#else
extern "C" __forceinline void hh_trafo_complex_kernel_4_SSE_1hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq)
{
	hh_trafo_complex_kernel_4_C_1hv(q, hh, nb, ldq);
}
#endif
