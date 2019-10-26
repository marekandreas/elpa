#ifdef HAVE_64BIT_INTEGER_MATH_SUPPORT
#define C_INT_TYPE_PTR long int*
#define C_INT_TYPE long int
#else
#define C_INT_TYPE_PTR int*
#define C_INT_TYPE int
#endif

void dlacpy_(char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, double*, C_INT_TYPE_PTR, double*, C_INT_TYPE_PTR);
void dgemm_(char*, char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, double*, double*, C_INT_TYPE_PTR, double*, C_INT_TYPE_PTR, double*, double*, C_INT_TYPE_PTR); 


void slacpy_(char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, float*, C_INT_TYPE_PTR, float*, C_INT_TYPE_PTR);
void sgemm_(char*, char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, float*, float*, C_INT_TYPE_PTR, float*, C_INT_TYPE_PTR, float*, float*, C_INT_TYPE_PTR); 




void zlacpy_(char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, double complex*, C_INT_TYPE_PTR, double complex*, C_INT_TYPE_PTR);
void zgemm_(char*, char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, double complex*, double complex*, C_INT_TYPE_PTR, double complex*, C_INT_TYPE_PTR, double complex*, double complex*, C_INT_TYPE_PTR); 


void clacpy_(char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, float complex*, C_INT_TYPE_PTR, float complex*, C_INT_TYPE_PTR);
void cgemm_(char*, char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, float complex*, float complex*, C_INT_TYPE_PTR, float complex*, C_INT_TYPE_PTR, float complex*, float complex*, C_INT_TYPE_PTR); 


