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
#ifdef _WIN32
#define ELPA_DOUBLE_COMPLEX_TYPE double _Complex
#define ELPA_FLOAT_COMPLEX_TYPE float _Complex
#else
#define ELPA_DOUBLE_COMPLEX_TYPE double complex
#define ELPA_FLOAT_COMPLEX_TYPE float complex
#endif
#ifdef DOUBLE_PRECISION
#undef math_type
#define math_type ELPA_DOUBLE_COMPLEX_TYPE
#endif
#ifdef SINGLE_PRECISION
#undef math_type
#define math_type ELPA_FLOAT_COMPLEX_TYPE
#endif
#endif
