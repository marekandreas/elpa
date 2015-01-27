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

module ftimings_type
  use, intrinsic :: iso_c_binding, only : C_INT64_T, C_DOUBLE, C_LONG_LONG, C_LONG, C_INT
  implicit none
  integer, parameter :: rk = C_DOUBLE
  integer, parameter :: name_length = 40
end module
