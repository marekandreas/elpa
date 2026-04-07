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

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

#ifdef HAVE_CONFIG_H
#include "config-f90.h"
#endif

/* Return number of microseconds since 1.1.1970, in a 64 bit integer.
 * (with 2^64 us ~ 6 * 10^5 years, this should be sufficiently overflow safe)
 */
int64_t ftimings_microseconds_since_epoch(void) {
#ifdef _WIN32
	FILETIME ft;
	ULARGE_INTEGER uli;
	const uint64_t epoch_diff_100ns = 116444736000000000ULL;

	GetSystemTimeAsFileTime(&ft);
	uli.LowPart = ft.dwLowDateTime;
	uli.HighPart = ft.dwHighDateTime;

	return (int64_t)((uli.QuadPart - epoch_diff_100ns) / 10ULL);
#else
	struct timeval tv;
	if (gettimeofday(&tv, NULL) != 0) {
		perror("gettimeofday");
		exit(1);
	}
	return (int64_t) (tv.tv_sec) * ((int64_t) 1000000) + (int64_t)(tv.tv_usec);
#endif
}

#ifndef WITH_MPI
int64_t t0 = 0;
#ifndef _WIN32
void __attribute__((constructor)) init_time(void) {
	t0 = ftimings_microseconds_since_epoch();
}
#endif

double seconds(void) {
	if (t0 == 0) {
		t0 = ftimings_microseconds_since_epoch();
	}
    return (ftimings_microseconds_since_epoch() - t0) / 1e6;
}
#endif
