      SUBROUTINE DSSMV( UPLO, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY )
*
      IMPLICIT NONE
*
*     .. Scalar Arguments ..
      CHARACTER          UPLO
      INTEGER            N, LDA, INCX, INCY
      DOUBLE PRECISION   ALPHA, BETA
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   A( LDA, * ), X( * ), Y( * )
*     ..
*
*  Purpose
*  =======
*
*  DSSMV performs the matrix-vector operation
*
*     y := alpha*A*x + beta*y,
*
*  where alpha and beta are scalars, x and y are n element vectors and
*  A is an n by n skew-symmetric matrix.
*
*  Arguments
*  ==========
*
*  UPLO   - CHARACTER*1.
*           On entry, UPLO specifies whether the upper or lower
*           triangular part of the array A is to be referenced as
*           follows:
*
*              UPLO = 'U' or 'u'   Only the upper triangular part of A
*                                  is to be referenced.
*
*              UPLO = 'L' or 'l'   Only the lower triangular part of A
*                                  is to be referenced.
*
*           Unchanged on exit.
*
*  N      - INTEGER.
*           On entry, N specifies the order of the matrix A.
*           N must be at least zero.
*           Unchanged on exit.
*
*  ALPHA  - DOUBLE PRECISION.
*           On entry, ALPHA specifies the scalar alpha.
*           Unchanged on exit.
*
*  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
*           Before entry with UPLO = 'U' or 'u', the leading n by n
*           upper triangular part of the array A must contain the upper
*           triangular part of the skew-symmetric matrix and the
*           strictly lower triangular part of A is not referenced.
*           Before entry with UPLO = 'L' or 'l', the leading n by n
*           lower triangular part of the array A must contain the lower
*           triangular part of the skew-symmetric matrix and the
*           strictly upper triangular part of A is not referenced.
*           Unchanged on exit.
*
*  LDA    - INTEGER.
*           On entry, LDA specifies the first dimension of A as declared
*           in the calling (sub) program. LDA must be at least
*           max( 1, n ).
*           Unchanged on exit.
*
*  X      - DOUBLE PRECISION array of dimension at least
*           ( 1 + ( n - 1 )*abs( INCX ) ).
*           Before entry, the incremented array X must contain the n
*           element vector x.
*           Unchanged on exit.
*
*  INCX   - INTEGER.
*           On entry, INCX specifies the increment for the elements of
*           X. INCX must not be zero.
*           Unchanged on exit.
*
*  BETA   - DOUBLE PRECISION.
*           On entry, BETA specifies the scalar beta. When BETA is
*           supplied as zero then Y need not be set on input.
*           Unchanged on exit.
*
*  Y      - DOUBLE PRECISION array of dimension at least
*           ( 1 + ( n - 1 )*abs( INCY ) ).
*           Before entry, the incremented array Y must contain the n
*           element vector y. On exit, Y is overwritten by the updated
*           vector y.
*
*  INCY   - INTEGER.
*           On entry, INCY specifies the increment for the elements of
*           Y. INCY must not be zero.
*           Unchanged on exit.
*
*  Further Details
*  ===============
*
*  Level 2 BLAS-like routine.
*
*  Written by Meiyue Shao, Lawrence Berkeley National Laboratory.
*  Last change: October 2014
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE
      PARAMETER          ( ZERO = 0.0D+0, ONE = 1.0D+0 )
      INTEGER            NB
      PARAMETER          ( NB = 64 )
*     ..
*     .. Local Scalars ..
      INTEGER            II, JJ, IC, IY, JC, JX, KX, KY, INFO
      DOUBLE PRECISION   TEMP
      LOGICAL            UPPER
*     .. Local Arrays ..
      DOUBLE PRECISION   WORK( NB )
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     ..
*     .. External Subroutines ..
      EXTERNAL           DGEMV, DSSMV_SM, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MAX
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      UPPER = LSAME( UPLO, 'U' )
      INFO = 0
      IF ( .NOT. UPPER .AND. .NOT. LSAME( UPLO, 'L' ) ) THEN
         INFO = 1
      ELSE IF ( N .LT. 0 ) THEN
         INFO = 2
      ELSE IF ( LDA .LT. MAX( 1,N ) ) THEN
         INFO = 5
      ELSE IF ( INCX .EQ. 0 ) THEN
         INFO = 7
      ELSE IF ( INCY .EQ. 0 ) THEN
         INFO = 10
      END IF
      IF ( INFO .NE. 0 ) THEN
         CALL XERBLA( 'DSSMV ', INFO )
         RETURN
      END IF
*
*     Quick return if possible.
*
      IF ( ( N .EQ. 0 ) .OR. ( ( ALPHA .EQ. ZERO ) .AND.
     $     ( BETA .EQ. ONE ) ) ) RETURN
*
*     Set up the start points in X and Y.
*
      IF ( INCX .GT. 0 ) THEN
         KX = 1
      ELSE
         KX = 1 - (N-1)*INCX
      END IF
      IF ( INCY .GT. 0 ) THEN
         KY = 1
      ELSE
         KY = 1 - (N-1)*INCY
      END IF
*
*     Start the operations.
*
      IF ( UPPER ) THEN
*
*        Form y when A is stored in the upper triangle.
*
         TEMP = BETA
         DO JJ = 1, N, NB
            JC = MIN( NB, N-JJ+1 )
            JX = KX + (JJ-1)*INCX
            IF ( INCX .LT. 0 ) JX = JX + (JC-1)*INCX
            DO II = 1, N, NB
               IC = MIN( NB, N-II+1 )
               IY = KY + (II-1)*INCY
               IF ( INCY .LT. 0 ) IY = IY + (IC-1)*INCY
*
*              Call DSSMV_SM for diagonal blocks,
*              and DGEMV for off-diagonal blocks.
*
               IF ( II .GT. JJ ) THEN
                  CALL DGEMV( 'T', NB, IC, -ALPHA, A( JJ, II ), LDA,
     $                 X( JX ), INCX, TEMP, Y( IY ), INCY )
               ELSE IF ( II .LT. JJ ) THEN
                  CALL DGEMV( 'N', NB, JC, ALPHA, A( II, JJ ), LDA,
     $                 X( JX ), INCX, TEMP, Y( IY ), INCY )
               ELSE
                  CALL DSSMV_SM( UPPER, JC, ALPHA, A( JJ, JJ ), LDA,
     $                 X( JX ), INCX, TEMP, Y( IY ), INCY, WORK )
               END IF
            END DO
            TEMP = ONE
         END DO
      ELSE
*
*        Form y when A is stored in the lower triangle.
*
         TEMP = BETA
         DO JJ = 1, N, NB
            JC = MIN( NB, N-JJ+1 )
            JX = KX + (JJ-1)*INCX
            IF ( INCX .LT. 0 ) JX = JX + (JC-1)*INCX
            DO II = 1, N, NB
               IC = MIN( NB, N-II+1 )
               IY = KY + (II-1)*INCY
               IF ( INCY .LT. 0 ) IY = IY + (IC-1)*INCY
*
*              Call DSSMV_SM for diagonal blocks,
*              and DGEMV for off-diagonal blocks.
*
               IF ( II .LT. JJ ) THEN
                  CALL DGEMV( 'T', JC, NB, -ALPHA, A( JJ, II ), LDA,
     $                 X( JX ), INCX, TEMP, Y( IY ), INCY )
               ELSE IF ( II .GT. JJ ) THEN
                  CALL DGEMV( 'N', IC, NB, ALPHA, A( II, JJ ), LDA,
     $                 X( JX ), INCX, TEMP, Y( IY ), INCY )
               ELSE
                  CALL DSSMV_SM( UPPER, JC, ALPHA, A( JJ, JJ ), LDA,
     $                 X( JX ), INCX, TEMP, Y( IY ), INCY, WORK )
               END IF
            END DO
            TEMP = ONE
         END DO
      END IF
*
      RETURN
*
*     End of DSSMV.
*
      END
*
*
*
      SUBROUTINE DSSMV_SM( UPPER, N, ALPHA, A, LDA, X, INCX, BETA, Y,
     $                     INCY, WORK )
*
*     The length of WORK is at least n.
*
      IMPLICIT NONE
*
*     .. Scalar Arguments ..
      LOGICAL            UPPER
      INTEGER            N, LDA, INCX, INCY
      DOUBLE PRECISION   ALPHA, BETA
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   A( LDA, * ), X( * ), Y( * ), WORK( * )
*     ..
*     .. Parameters ..
      DOUBLE PRECISION   ONE
      PARAMETER          ( ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            I, IX, IY, KX, KY
      CHARACTER          UPLO
*     ..
*     .. External Subroutines ..
      EXTERNAL           DCOPY, DSCAL, DTRMV
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS
*     ..
*     .. Executable Statements ..
*
      IF ( UPPER ) THEN
         UPLO = 'U'
      ELSE
         UPLO = 'L'
      END IF
*
*     Set up the start points in X and Y.
*
      IF ( INCX .GT. 0 ) THEN
         KX = 1
      ELSE
         KX = 1 - (N-1)*INCX
      END IF
      IF ( INCY .GT. 0 ) THEN
         KY = 1
      ELSE
         KY = 1 - (N-1)*INCY
      END IF
*
*     Note: DSCAL does not work with negative INCY.
*
      CALL DSCAL( N, BETA, Y, ABS( INCY ) )
*
*     Start the operations.
*
      IF ( ALPHA .EQ. ONE ) THEN
         CALL DCOPY( N, X, INCX, WORK, 1 )
      ELSE
         IF ( INCX .EQ. 1 ) THEN
            DO I = 1, N
               WORK( I ) = ALPHA*X( I )
            END DO
         ELSE
            IX = KX
            DO I = 1, N
               WORK( I ) = ALPHA*X( IX )
               IX = IX + INCX
            END DO
         END IF
      END IF
      CALL DTRMV( UPLO, 'N', 'N', N, A, LDA, WORK, 1 )
      IF ( INCY .EQ. 1 ) THEN
         DO I = 1, N
            Y( I ) = Y( I ) + WORK( I )
         END DO
      ELSE
         IY = KY
         DO I = 1, N
            Y( IY ) = Y( IY ) + WORK( I )
            IY = IY + INCY
         END DO
      END IF
*
      IF ( ALPHA .EQ. ONE ) THEN
         CALL DCOPY( N, X, INCX, WORK, 1 )
      ELSE
         IF ( INCX .EQ. 1 ) THEN
            DO I = 1, N
               WORK( I ) = ALPHA*X( I )
            END DO
         ELSE
            IX = KX
            DO I = 1, N
               WORK( I ) = ALPHA*X( IX )
               IX = IX + INCX
            END DO
         END IF
      END IF
      CALL DTRMV( UPLO, 'T', 'N', N, A, LDA, WORK, 1 )
      IF ( INCY .EQ. 1 ) THEN
         DO I = 1, N
            Y( I ) = Y( I ) - WORK( I )
         END DO
      ELSE
         IY = KY
         DO I = 1, N
            Y( IY ) = Y( IY ) - WORK( I )
            IY = IY + INCY
         END DO
      END IF
*
      RETURN
*
*     End of DSSMV_SM.
*
      END
