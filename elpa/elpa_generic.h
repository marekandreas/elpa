#pragma once

/*! \brief generic C method for elpa_set
 *
 *  \details
 *  \param  handle  handle of the ELPA object for which a key/value pair should be set
 *  \param  name    the name of the key
 *  \param  value   integer/double value to be set for the key
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */
#define elpa_set(e, name, value, error) _Generic((value), \
                int: \
                  elpa_set_integer, \
                \
                double: \
                  elpa_set_double \
        )(e, name, value, error)


/*! \brief generic C method for elpa_get
 *
 *  \details
 *  \param  handle  handle of the ELPA object for which a key/value pair should be queried
 *  \param  name    the name of the key
 *  \param  value   integer/double value to be queried
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */
#define elpa_get(e, name, value, error) _Generic((value), \
                int*: \
                  elpa_get_integer, \
                \
                double*: \
                  elpa_get_double \
        )(e, name, value, error)


/*! \brief generic C method for elpa_eigenvectors
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       float/double float complex/double complex pointer to matrix a
 *  \param  ev      on return: float/double pointer to eigenvalues
 *  \param  q       on return: float/double float complex/double complex pointer to eigenvectors
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */
#define elpa_eigenvectors(handle, a, ev, q, error) _Generic((a), \
                double*: \
                  elpa_eigenvectors_d, \
                \
                float*: \
                  elpa_eigenvectors_f, \
                \
                double complex*: \
                  elpa_eigenvectors_dc, \
                \
                float complex*: \
                  elpa_eigenvectors_fc \
        )(handle, a, ev, q, error)

/*! \brief generic C method for elpa_skew_eigenvectors
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       float/double float complex/double complex pointer to matrix a
 *  \param  ev      on return: float/double pointer to eigenvalues
 *  \param  q       on return: float/double float complex/double complex pointer to eigenvectors
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */
#define elpa_skew_eigenvectors(handle, a, ev, q, error) _Generic((a), \
                double*: \
                  elpa_eigenvectors_d, \
                \
                float*: \
                  elpa_eigenvectors_f, \
        )(handle, a, ev, q, error)



/*! \brief generic C method for elpa_generalized_eigenvectors
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       float/double float complex/double complex pointer to matrix a
 *  \param  b       float/double float complex/double complex pointer to matrix b
 *  \param  ev      on return: float/double pointer to eigenvalues
 *  \param  q       on return: float/double float complex/double complex pointer to eigenvectors
 *  \param  is_already_decomposed   set to 1, if b already decomposed by previous call to elpa_generalized
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */
#define elpa_generalized_eigenvectors(handle, a, b, ev, q, is_already_decomposed, error) _Generic((a), \
                double*: \
                  elpa_generalized_eigenvectors_d, \
                \
                float*: \
                  elpa_generalized_eigenvectors_f, \
                \
                double complex*: \
                  elpa_generalized_eigenvectors_dc, \
                \
                float complex*: \
                  elpa_generalized_eigenvectors_fc \
        )(handle, a, b, ev, q, is_already_decomposed, error)


/*! \brief generic C method for elpa_eigenvalues
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       float/double float complex/double complex pointer to matrix a
 *  \param  ev      on return: float/double pointer to eigenvalues
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */
#define elpa_eigenvalues(handle, a, ev, error) _Generic((a), \
                double*: \
                  elpa_eigenvalues_d, \
                \
                float*: \
                  elpa_eigenvalues_f, \
                \
                double complex*: \
                  elpa_eigenvalues_dc, \
                \
                float complex*: \
                  elpa_eigenvalues_fc \
        )(handle, a, ev, error)

/*! \brief generic C method for elpa_skew_eigenvalues
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       float/double float complex/double complex pointer to matrix a
 *  \param  ev      on return: float/double pointer to eigenvalues
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */
#define elpa_skew_eigenvalues(handle, a, ev, error) _Generic((a), \
                double*: \
                  elpa_eigenvalues_d, \
                \
                float*: \
                  elpa_eigenvalues_f, \
        )(handle, a, ev, error)


/*  \brief generic C method for elpa_cholesky
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       float/double float complex/double complex pointer to matrix a, for which
 *                  the cholesky factorizaion will be computed
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */
#define elpa_cholesky(handle, a, error) _Generic((a), \
                double*: \
                  elpa_cholesky_d, \
                \
                float*: \
                  elpa_cholesky_f, \
                \
                double complex*: \
                  elpa_cholesky_dc, \
                \
                float complex*: \
                  elpa_cholesky_fc \
        )(handle, a, error)


/*! \brief generic C method for elpa_hermitian_multiply
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  uplo_a  descriptor for matrix a
 *  \param  uplo_c  descriptor for matrix c
 *  \param  ncb     int
 *  \param  a       float/double float complex/double complex pointer to matrix a
 *  \param  b       float/double float complex/double complex pointer to matrix b
 *  \param  nrows_b number of rows for matrix b
 *  \param  ncols_b number of cols for matrix b
 *  \param  c       float/double float complex/double complex pointer to matrix c
 *  \param  nrows_c number of rows for matrix c
 *  \param  ncols_c number of cols for matrix c
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */
#define elpa_hermitian_multiply(handle, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, c, nrows_c, ncols_c, error) _Generic((a), \
                double*: \
                  elpa_hermitian_multiply_d, \
                \
                float*: \
                  elpa_hermitian_multiply_f, \
                \
                double complex*: \
                  elpa_hermitian_multiply_dc, \
                \
                float complex*: \
                  elpa_hermitian_multiply_fc \
        )(handle, a, error)


/*! \brief generic C method for elpa_invert_triangular
 *
 *  \details
 *  \param  handle  handle of the ELPA object, which defines the problem
 *  \param  a       float/double float complex/double complex pointer to matrix a, which
 *                  should be inverted
 *  \param  error   on return the error code, which can be queried with elpa_strerr()
 *  \result void
 */
#define elpa_invert_triangular(handle, a, error) _Generic((a), \
                double*: \
                  elpa_invert_trm_d, \
                \
                float*: \
                  elpa_invert_trm_f, \
                \
                double complex*: \
                  elpa_invert_trm_dc, \
                \
                float complex*: \
                  elpa_invert_trm_fc \
        )(handle, a, error)
