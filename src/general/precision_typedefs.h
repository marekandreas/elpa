#ifdef REALCASE
#ifdef DOUBLE_PRECISION
//typedef double math_type;
#undef math_type
#define math_type double
#endif
#ifdef SINGLE_PRECISION
//typedef float math_type;
#undef math_type
#define math_type float
#endif
#endif

#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION
#undef math_type
#define math_type double complex
#endif
#ifdef SINGLE_PRECISION
#undef math_type
#define math_type float complex
#endif
#endif
