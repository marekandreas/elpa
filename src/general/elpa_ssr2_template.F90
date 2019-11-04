#if REALCASE == 1
#ifdef DOUBLE_PRECISION
subroutine elpa_dssr2(n, x, y,  a, lda )
#endif
#ifdef SINGLE_PRECISION
subroutine elpa_sssr2(n, x, y,  a, lda )
#endif
#endif /* REALCASE */
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION
subroutine elpa_zssr2(n, x, y,  a, lda )
#endif
#ifdef SINGLE_PRECISION
subroutine elpa_cssr2(n, x, y,  a, lda )
#endif
#endif /* COMPLEXCASE */

  use precision
  use elpa_utilities, only : error_unit
  use elpa_blas_interfaces
  implicit none
#include "./precision_kinds.F90"

  integer(kind=BLAS_KIND)     :: n, lda
  MATH_DATATYPE(kind=rck)     :: a( lda, * ), x( * ), y( * )
  integer(kind=ik), parameter :: nb = 64
  MATH_DATATYPE(kind=rck)     :: temp1, temp2
  integer(kind=ik)            :: i, j, ii, jj, ic, ix, iy, jc, jx, jy, info
  logical                     :: upper

  ! test the input parameters.
  info = 0
  if (n == 0) then
    return 
  end if
  if ( n < 0 ) then
    info = 1
  else if ( lda < max( 1,n ) ) then
    info = 5
  end if
  if ( info /= 0 ) then
    write(error_unit,*) "wrong arguments in elpa_ssmv, info =", info
    return
  end if

  ! Access A in lower triangular part.
  do jj = 1, n, nb
    jc = min( nb, n-jj+1 )
    jx = 1 + (jj-1)
    jy = 1 + (jj-1)
     
    do j = 1, jc-1
    ! Do local update for blocks on the diagonal
      if ( ( x( jx + j -1) /= zero ) .or. &
           ( y( jy + j -1 ) /= zero ) ) then
        temp1 = - y( jy + j - 1 )
        temp2 = - x( jy + j - 1 )
        do i = j+1, jc
          a( jj +  i -1 , jj +  j -1 ) = a(jj + i -1,jj +  j -1 ) + x( jx + i  -1)*temp1 - y(jj +  i -1 )*temp2
        end do
      end if
    end do
            
    ! Use dger for other blocks
    do ii = jj+nb, n, nb
      ic = min( nb, n-ii+1 )
      ix = 1 + (ii-1)
      iy = 1 + (ii-1)
#if REALCASE == 1
      call PRECISION_GER(int(ic,kind=BLAS_KIND), int(nb,kind=BLAS_KIND), -one, x( ix ), 1_BLAS_KIND, y( jy ), 1_BLAS_KIND, &
                          a( ii, jj ), int(lda,kind=BLAS_KIND) )
      call PRECISION_GER(int(ic,kind=BLAS_KIND), int(nb,kind=BLAS_KIND), one, y( iy ), 1_BLAS_KIND, x( jx ), 1_BLAS_KIND, &
                         a( ii, jj ), int(lda,kind=BLAS_KIND) )
#endif
    end do
  end do

  return
end subroutine
