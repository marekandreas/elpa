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
#include <unistd.h>

long ftimings_resident_set_size() {
	long rss = 0L;
	FILE* fp = NULL;
	if ((fp = fopen( "/proc/self/statm", "r" )) == NULL ) {
		return 0L;
	}
	if (fscanf(fp, "%*s%ld", &rss) != 1) {
		fclose(fp);
		return (size_t)0L;	  /* Can't read? */
	}
	fclose(fp);
	return rss * sysconf( _SC_PAGESIZE);
}
