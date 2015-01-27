/* Copyright 2014 Lorenz Hüdepohl
 *
 * This file is part of ftimings.
 *
 * ftimings is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ftimings is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with ftimings.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifdef HAVE_CONFIG_H
#include "config-f90.h"
#endif

/* Return number of microseconds since 1.1.1970, in a 64 bit integer.
 * (with 2^64 us ~ 6 * 10^5 years, this should be sufficiently overflow safe)
 */
int64_t ftimings_microseconds_since_epoch(void) {
	struct timeval tv;
	if (gettimeofday(&tv, NULL) != 0) {
		perror("gettimeofday");
		exit(1);
	}
	return (int64_t) (tv.tv_sec) * ((int64_t) 1000000) + (int64_t)(tv.tv_usec);
}
