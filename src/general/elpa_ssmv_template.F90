#if REALCASE == 1
#ifdef DOUBLE_PRECISION
subroutine elpa_dssmv(n, alpha, a, lda, x,  y)
#endif
#ifdef SINGLE_PRECISION
subroutine elpa_sssmv(n, alpha, a, lda, x,  y)
#endif
#endif /* REALCASE */
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION
subroutine elpa_zssmv(n, alpha, a, lda, x,  y)
#endif
#ifdef SINGLE_PRECISION
subroutine elpa_cssmv(n, alpha, a, lda, x,  y)
#endif
#endif /* COMPLEXCASE */

  use precision
  use elpa_utilities, only : error_unit
  use elpa_blas_interfaces  
  implicit none
#include "./precision_kinds.F90"

  integer(kind=BLAS_KIND)     :: n, lda
  MATH_DATATYPE(kind=rck)     :: alpha
  MATH_DATATYPE(kind=rck)     :: a( lda, * ), x( * ), y( * )
  integer(kind=ik), parameter :: nb = 64 
  integer(kind=ik)            :: ii, jj, ic, iy, jc, jx, info
  MATH_DATATYPE(kind=rck)     :: temp
  MATH_DATATYPE(kind=rck)     :: work( nb )

  ! Test the input parameters.
  info = 0
  if (n == 0) then
    return
  end if
  if ( n < 0 ) then
    info = 1
  else if ( lda < max( 1,n ) ) then
    info = 4
  end if
  if ( info /= 0 ) then
    write(error_unit,*) "wrong arguments in elpa_ssmv, info =", info
    return
  end if

  ! Access only lower triangular part of a

  temp = zero
  do jj = 1, n, nb
    jc = min( nb, n-jj+1 )
    jx = 1 + (jj-1)
    do ii = 1, n, nb
      ic = min( nb, n-ii+1 )
      iy = 1 + (ii-1)
        
      ! gemv for non-diagonal blocks. use 2x dtrmv for diagonal blocks
      if ( ii < jj ) then
       call PRECISION_GEMV('t', int(jc,kind=BLAS_KIND), int(nb,kind=BLAS_KIND), -alpha, &
                           a( jj, ii ), int(lda, kind=BLAS_KIND), &
                           x( jx ), 1_BLAS_KIND, temp, y( iy ), 1_BLAS_KIND )
      else if ( ii > jj ) then
       call PRECISION_GEMV('n', int(ic,kind=BLAS_KIND), int(nb,kind=BLAS_KIND), alpha, a( ii, jj ), &
                           int(lda,kind=BLAS_KIND), &
                           x( jx ), 1_BLAS_KIND, temp, y( iy ), 1_BLAS_KIND )
      else
        if (temp == zero) then
          y(1:n) = zero
        else if (temp /= one) then
          ! should not happen
          call PRECISION_SCAL( int(jc,kind=BLAS_KIND), temp, y( iy ), 1_BLAS_KIND)
        end if
        call PRECISION_COPY( int(jc,kind=BLAS_KIND), x( jx ), 1_BLAS_KIND, work, 1_BLAS_KIND )
        call PRECISION_TRMV( 'l', 'n', 'n', int(jc,kind=BLAS_KIND), a( jj, jj ), int(lda,kind=BLAS_KIND), work, 1_BLAS_KIND )
        call PRECISION_AXPY( int(jc,kind=BLAS_KIND),alpha, work, 1_BLAS_KIND, y( iy ), 1_BLAS_KIND)
           
        call PRECISION_COPY( int(jc,kind=BLAS_KIND), x( jx ), 1_BLAS_KIND, work, 1_BLAS_KIND )
        call PRECISION_TRMV( 'l', 't', 'n', int(jc,kind=BLAS_KIND), a( jj, jj ), int(lda,kind=BLAS_KIND), work, 1_BLAS_KIND )
        call PRECISION_AXPY(int(jc,kind=BLAS_KIND), -alpha, work, 1_BLAS_KIND, y( iy ), 1_BLAS_KIND)               
      end if
    end do
    temp = one
  end do

  return
end subroutine

