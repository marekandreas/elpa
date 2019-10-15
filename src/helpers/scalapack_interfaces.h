#ifdef HAVE_64BIT_INTEGER_MATH_SUPPORT
#define C_INT_TYPE_PTR long int*
#define C_INT_TYPE long int
#else
#define C_INT_TYPE_PTR int*
#define C_INT_TYPE int
#endif

C_INT_TYPE numroc_(C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR);


void pdlacpy_(char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, double*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, double*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR);
void pdtran_(C_INT_TYPE_PTR, C_INT_TYPE_PTR, double*, double*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, double*, double*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR);


void pslacpy_(char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, float*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, float*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR);
void pstran_(C_INT_TYPE_PTR, C_INT_TYPE_PTR, float*, float*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, float*, float*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR);



void pzlacpy_(char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, double complex*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, double complex*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR);
void pztranc_(C_INT_TYPE_PTR, C_INT_TYPE_PTR, double complex*, double complex*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, double complex*, double complex*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR);


void pclacpy_(char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, float complex*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, float complex*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR);
void pctranc_(C_INT_TYPE_PTR, C_INT_TYPE_PTR, float complex*, float complex*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, float complex*, float complex*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR);


