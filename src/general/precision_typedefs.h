#ifdef REALCASE
#ifdef DOUBLE_PRECISION
//typedef double math_type;
#define math_type double
#endif
#ifdef SINGLE_PRECISION
//typedef float math_type;
#define math_type float
#endif
#endif

#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION
#endif
#ifdef SINGLE_PRECISION
#endif
#endif
