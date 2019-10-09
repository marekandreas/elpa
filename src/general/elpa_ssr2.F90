subroutine elpa_dssr2(n, x, y,  a, lda )
!
      implicit none
!
!     .. scalar arguments ..
      integer            n, lda
!     ..
!     .. array arguments ..
      double precision   a( lda, * ), x( * ), y( * )
!     ..
!
!     .. parameters ..
      double precision   zero, one, temp1, temp2
      parameter          ( zero = 0.0d+0, one = 1.0d+0 )
      integer            nb
      parameter          ( nb = 64 )
!
!     .. local scalars ..
      integer            i, j, ii, jj, ic, ix, iy, jc, jx, jy, info
      logical            upper

!     .. external subroutines ..
      external           dger
!     ..
!     .. intrinsic functions ..
      intrinsic          max, min
!     ..
!     .. executable statements ..
!
!     test the input parameters.

      info = 0
      if (n .eq. 0) then
         return 
      end if
      if ( n .lt. 0 ) then
         info = 1
      else if ( lda .lt. max( 1,n ) ) then
         info = 5
      end if
      if ( info .ne. 0 ) then
         print *,"wrong arguments in elpa_ssmv, info =", info
         return
      end if
!
!        Access A in lower triangular part.
!
         do jj = 1, n, nb
            jc = min( nb, n-jj+1 )
            jx = 1 + (jj-1)
            jy = 1 + (jj-1)
     
            do j = 1, jc-1
!           Do local update for blocks on the diagonal
               if ( ( x( jx + j -1) .ne. zero ) .or. &
     &              ( y( jy + j -1 ) .ne. zero ) ) then
                  temp1 = - y( jy + j - 1 )
                  temp2 = - x( jy + j - 1 )
                  do i = j+1, jc
                     a( jj +  i -1 , jj +  j -1 ) = a(jj + i -1,jj +  j -1 ) + x( jx + i  -1)*temp1 - y(jj +  i -1 )*temp2
                  end do
               end if
            end do
            
!           Use dger for other blocks
            do ii = jj+nb, n, nb
               ic = min( nb, n-ii+1 )
               ix = 1 + (ii-1)
               iy = 1 + (ii-1)
               call dger( ic, nb, -one, x( ix ), 1, y( jy ), 1, &
     &              a( ii, jj ), lda )
               call dger( ic, nb, one, y( iy ), 1, x( jx ), 1, &
     &              a( ii, jj ), lda )
            end do
         end do

      return
!
!     end of elpa_dssr2.
!
end
!
!