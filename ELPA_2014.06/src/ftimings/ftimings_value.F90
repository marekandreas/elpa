! Copyright 2014 Lorenz Hüdepohl
!
! This file is part of ftimings.
!
! ftimings is free software: you can redistribute it and/or modify
! it under the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! ftimings is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU Lesser General Public License for more details.
!
! You should have received a copy of the GNU Lesser General Public License
! along with ftimings.  If not, see <http://www.gnu.org/licenses/>.

#ifdef HAVE_CONFIG_H
#include "config-f90.h"
#endif

module ftimings_value
  use ftimings_type
  implicit none
  public

  type value_t
    integer(kind=C_INT64_T) :: micros = 0         ! microseconds spent in this node
    integer(kind=C_LONG) :: virtualmem = 0        ! newly created virtual memory
    integer(kind=C_LONG) :: maxrsssize = 0            ! newly used max. resident mem ("high water mark")

    integer(kind=C_LONG) :: rsssize = 0           ! newly used resident memory

    integer(kind=C_LONG_LONG) :: flop_count = 0   ! floating point operations done in this node
    integer(kind=C_LONG_LONG) :: ldst = 0         ! number of loads and stores
  end type

  interface operator(+)
    module procedure value_add
  end interface

  interface operator(-)
    module procedure value_minus
    module procedure value_inverse
  end interface

  type(value_t), parameter :: null_value = value_t(micros = 0, &
                                                   rsssize = 0, &
                                                   virtualmem = 0, &
                                                   maxrsssize = 0, &
                                                   flop_count = 0)

  contains

  pure elemental function value_add(a,b) result(c)
    class(value_t), intent(in) :: a, b
    type(value_t) :: c
    c%micros = a%micros + b%micros
    c%rsssize = a%rsssize + b%rsssize
    c%virtualmem = a%virtualmem + b%virtualmem
    c%maxrsssize = a%maxrsssize + b%maxrsssize
#ifdef HAVE_LIBPAPI
    c%flop_count = a%flop_count + b%flop_count
    c%ldst = a%ldst + b%ldst
#endif
  end function

  pure elemental function value_minus(a,b) result(c)
    class(value_t), intent(in) :: a, b
    type(value_t) :: c
    c%micros = a%micros - b%micros
    c%rsssize = a%rsssize - b%rsssize
    c%virtualmem = a%virtualmem - b%virtualmem
    c%maxrsssize = a%maxrsssize - b%maxrsssize
#ifdef HAVE_LIBPAPI
    c%flop_count = a%flop_count - b%flop_count
    c%ldst = a%ldst - b%ldst
#endif
  end function

  pure elemental function value_inverse(a) result(neg_a)
    class(value_t), intent(in) :: a
    type(value_t) :: neg_a
    neg_a%micros = - a%micros
    neg_a%rsssize = - a%rsssize
    neg_a%virtualmem = - a%virtualmem
    neg_a%maxrsssize = - a%maxrsssize
#ifdef HAVE_LIBPAPI
    neg_a%flop_count = - a%flop_count
    neg_a%ldst = - a%ldst
#endif
  end function
end module
