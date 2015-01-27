/* Copyright 2014 Andreas Marek, Lorenz Hüdepohl
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
#include <sys/types.h>
#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

long ftimings_highwater_mark() {
	long hwm = 0L;
	char line[1024];
	FILE* fp = NULL;

	if ((fp = fopen( "/proc/self/status", "r" )) == NULL ) {
		return 0L;
	}

	/* Read memory size data from /proc/pid/status */
	while(fgets(line, sizeof line, fp)) {
		if (sscanf(line, "VmHWM: %ld kB", &hwm) == 1) {
			break;
		}
	}
	fclose(fp);

	return hwm * 1024L;
}
