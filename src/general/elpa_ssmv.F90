subroutine elpa_dssmv( n, alpha, a, lda, x,  y )
!
      implicit none
!
!     .. scalar arguments ..
      integer            n, lda
      double precision   alpha
!     ..
!     .. array arguments ..
      double precision   a( lda, * ), x( * ), y( * )
!     ..

!
!     .. parameters ..
      double precision   zero, one
      parameter          ( zero = 0.0d+0, one = 1.0d+0 )
      integer            nb
      parameter          ( nb = 64 )
!     ..
!     .. local scalars ..
      integer            ii, jj, ic, iy, jc, jx, info
      double precision   temp
!     .. local arrays ..
      double precision   work( nb )
!     ..
!     .. external subroutines ..
      external           dgemv, dtrmv, dcopy, daxpy
!     ..
!     .. intrinsic functions ..
      intrinsic          max, min
!     ..
!     .. executable statements ..


!     Test the input parameters.
      info = 0
      if (n .eq. 0) then
         return
      end if
      if ( n .lt. 0 ) then
         info = 1
      else if ( lda .lt. max( 1,n ) ) then
         info = 4
      end if
      if ( info .ne. 0 ) then
         print *,"wrong arguments in elpa_ssmv, info =", info
         return
      end if

!     Access only lower triangular part of a

      temp = zero
      do jj = 1, n, nb
         jc = min( nb, n-jj+1 )
         jx = 1 + (jj-1)
         do ii = 1, n, nb
            ic = min( nb, n-ii+1 )
            iy = 1 + (ii-1)
            
!           gemv for non-diagonal blocks. use 2x dtrmv for diagonal blocks
            if ( ii .lt. jj ) then
               call dgemv( 't', jc, nb, -alpha, a( jj, ii ), lda, &
&                 x( jx ), 1, temp, y( iy ), 1 )
            else if ( ii .gt. jj ) then
               call dgemv( 'n', ic, nb, alpha, a( ii, jj ), lda, &
&                 x( jx ), 1, temp, y( iy ), 1 )
            else
               if (temp .eq. zero) then
                  y(1:n) = zero
               else if (temp .ne. one) then
!                    should not happen
                  call dscal( jc, temp, y( iy ), 1)
               end if
               call dcopy( jc, x( jx ), 1, work, 1 )
               call dtrmv( 'l', 'n', 'n', jc, a( jj, jj ), lda, work, 1 )
               call daxpy(jc,alpha,work,1,y( iy ),1)
               
               call dcopy( jc, x( jx ), 1, work, 1 )
               call dtrmv( 'l', 't', 'n', jc, a( jj, jj ), lda, work, 1 )
               call daxpy(jc,-alpha,work,1,y( iy ),1)               
            end if
         end do
         temp = one
      end do
!
      return
!
!     end of elpa_dssmv.
!
end