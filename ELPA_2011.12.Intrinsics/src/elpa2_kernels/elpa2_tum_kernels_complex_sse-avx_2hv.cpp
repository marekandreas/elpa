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
// --------------------------------------------------------------------------------------------------

#include <complex>
#include <x86intrin.h>

#define __forceinline __attribute__((always_inline))

#ifdef __USE_AVX128__
#undef __AVX__
#endif

//Forward declaration
#ifdef __AVX__
extern "C" __forceinline void hh_trafo_complex_kernel_4_AVX_2hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq, int ldh, std::complex<double> s);
#else
extern "C" __forceinline void hh_trafo_complex_kernel_4_SSE_2hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq, int ldh, std::complex<double> s);
#endif

extern "C" void double_hh_trafo_complex_(std::complex<double>* q, std::complex<double>* hh, int* pnb, int* pnq, int* pldq, int* pldh)
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;

	std::complex<double> s = conj(hh[(ldh)+1])*1.0;
	for (i = 2; i < nb; i++)
	{
		s += hh[i-1] * conj(hh[(i+ldh)]);
	}

#ifdef __AVX__
	for (i = 0; i < nq; i+=4)
	{
		hh_trafo_complex_kernel_4_AVX_2hv(&q[i], hh, nb, ldq, ldh, s);
	}
#else
	for (i = 0; i < nq; i+=4)
	{
		hh_trafo_complex_kernel_4_SSE_2hv(&q[i], hh, nb, ldq, ldh, s);
	}
#endif
}

extern "C" __forceinline void hh_trafo_complex_kernel_4_C_2hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq, int ldh, std::complex<double> s)
{
	std::complex<double> x1;
	std::complex<double> x2;
	std::complex<double> x3;
	std::complex<double> x4;
	std::complex<double> y1;
	std::complex<double> y2;
	std::complex<double> y3;
	std::complex<double> y4;
	std::complex<double> h1;
	std::complex<double> h2;
	std::complex<double> tau1;
	std::complex<double> tau2;
	int i=0;

	x1 = q[ldq+0];
	x2 = q[ldq+1];
	x3 = q[ldq+2];
	x4 = q[ldq+3];

	h2 = conj(hh[ldh+1]);

	y1 = q[0] + (x1*h2);
	y2 = q[1] + (x2*h2);
	y3 = q[2] + (x3*h2);
	y4 = q[3] + (x4*h2);

	for (i = 2; i < nb; i++)
	{
		h1 = conj(hh[i-1]);
		h2 = conj(hh[ldh+i]);

		x1 += (q[(i*ldq)+0] * h1);
		y1 += (q[(i*ldq)+0] * h2);
		x2 += (q[(i*ldq)+1] * h1);
		y2 += (q[(i*ldq)+1] * h2);
		x3 += (q[(i*ldq)+2] * h1);
		y3 += (q[(i*ldq)+2] * h2);
		x4 += (q[(i*ldq)+3] * h1);
		y4 += (q[(i*ldq)+3] * h2);
	}
	h1 = conj(hh[nb-1]);

	x1 += (q[(nb*ldq)+0] * h1);
	x2 += (q[(nb*ldq)+1] * h1);
	x3 += (q[(nb*ldq)+2] * h1);
	x4 += (q[(nb*ldq)+3] * h1);

	tau1 = hh[0];
	tau2 = hh[ldh];

	h1 = (-1.0)*tau1;

	x1 *= h1;
	x2 *= h1;
	x3 *= h1;
	x4 *= h1;

	h1 = (-1.0)*tau2;
	h2 = (-1.0)*tau2;
	h2 *= s;
	y1 = y1*h1 +x1*h2;
	y2 = y2*h1 +x2*h2;
	y3 = y3*h1 +x3*h2;
	y4 = y4*h1 +x4*h2;

	q[0] += y1;
	q[1] += y2;
	q[2] += y3;
	q[3] += y4;

	h2 = hh[ldh+1];
	q[ldq+0] += (x1 + (y1*h2));
	q[ldq+1] += (x2 + (y2*h2));
	q[ldq+2] += (x3 + (y3*h2));
	q[ldq+3] += (x4 + (y4*h2));

	for (i = 2; i < nb; i++)
	{
		h1 = hh[i-1];
		h2 = hh[ldh+i];

		q[(i*ldq)+0] += ((x1*h1) + (y1*h2));
		q[(i*ldq)+1] += ((x2*h1) + (y2*h2));
		q[(i*ldq)+2] += ((x3*h1) + (y3*h2));
		q[(i*ldq)+3] += ((x4*h1) + (y4*h2));
	}

	h1 = hh[nb-1];
	q[(nb*ldq)+0] += (x1*h1);
	q[(nb*ldq)+1] += (x2*h1);
	q[(nb*ldq)+2] += (x3*h1);
	q[(nb*ldq)+3] += (x4*h1);
}

#ifdef __AVX__
extern "C" __forceinline void hh_trafo_complex_kernel_4_AVX_2hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq, int ldh, std::complex<double> s)
{
	hh_trafo_complex_kernel_4_C_2hv(q, hh, nb, ldq, ldh, s);
}
#else
extern "C" __forceinline void hh_trafo_complex_kernel_4_SSE_2hv(std::complex<double>* q, std::complex<double>* hh, int nb, int ldq, int ldh, std::complex<double> s)
{
	hh_trafo_complex_kernel_4_C_2hv(q, hh, nb, ldq, ldh, s);
}
#endif
