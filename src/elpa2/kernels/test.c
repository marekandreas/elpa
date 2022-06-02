#define offset 2

#define _SIMD_LOAD _mm_load_pd
#define _SIMD_STORE _mm_store_pd
#define _SIMD_XOR _mm_xor_pd


#undef _LOAD
#undef _STORE
#undef _XOR
#define _LOAD(x) _SIMD_LOAD(x)
#define _LOAD(x) _SIMD_LOAD(0, (unsigned long int *) x)
#define _XOR(a ,b) _SIMD_XOR(a, b)

#undef _LOAD
//#undef _STORE
#undef _XOR
#define _STORE(a, b) _SIMD_STORE(a, b)
#define _STORE(a, b) _SIMD_STORE((__vector unsigned int) b, 0, (unsigned int *) a)
//#define _XOR(a, b) vec_mul(b, a)



_LOAD(&q[ldq]);
_STORE(&q[offset],q2)
_XOR(tau1, sign)
