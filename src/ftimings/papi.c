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
#include <stdlib.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

static int event_set;

static int tried_papi_init = 0;
static int papi_available = 0;
static int flops_available = 0;
static int ldst_available = 0;

#ifdef HAVE_LIBPAPI
#include <papi.h>

int ftimings_papi_init(void) {
	int ret;

	if (tried_papi_init) {
		return papi_available;
	}

#pragma omp critical
	{
		/* Think about it :) */
		if (tried_papi_init) {
			goto end;
		}

		tried_papi_init = 1;

		event_set = PAPI_NULL;

		if ((ret = PAPI_library_init(PAPI_VER_CURRENT)) < 0) {
			fprintf(stderr, "ftimings: %s:%d: PAPI_library_init(%d): %s\n",
					__FILE__, __LINE__, PAPI_VER_CURRENT, PAPI_strerror(ret));
			goto error;
		}

		if ((ret = PAPI_create_eventset(&event_set)) < 0) {
			fprintf(stderr, "ftimings: %s:%d PAPI_create_eventset(): %s\n",
					__FILE__, __LINE__, PAPI_strerror(ret));
			goto error;
		}

		/* Check FLOP counter availability */
		if ((ret = PAPI_query_event(PAPI_DP_OPS)) < 0) {
			fprintf(stderr, "ftimings: %s:%d: PAPI_query_event(PAPI_DP_OPS): %s\n",
					__FILE__, __LINE__, PAPI_strerror(ret));
			flops_available = 0;
		} else if ((ret = PAPI_add_event(event_set, PAPI_DP_OPS)) < 0) {
			fprintf(stderr, "ftimings: %s:%d PAPI_add_event(): %s\n",
					__FILE__, __LINE__, PAPI_strerror(ret));
			flops_available = 0;
		} else {
			flops_available = 1;
		}

		/* Loads + Stores */
		if ((ret = PAPI_query_event(PAPI_LD_INS)) < 0) {
			fprintf(stderr, "ftimings: %s:%d: PAPI_query_event(PAPI_LD_INS): %s\n",
					__FILE__, __LINE__, PAPI_strerror(ret));
			ldst_available = 0;
		} else if ((ret = PAPI_query_event(PAPI_SR_INS)) < 0) {
			fprintf(stderr, "ftimings: %s:%d: PAPI_query_event(PAPI_SR_INS): %s\n",
					__FILE__, __LINE__, PAPI_strerror(ret));
			ldst_available = 0;
		} else if ((ret = PAPI_add_event(event_set, PAPI_LD_INS)) < 0) {
			fprintf(stderr, "ftimings: %s:%d PAPI_add_event(event_set, PAPI_LD_INS): %s\n",
					__FILE__, __LINE__, PAPI_strerror(ret));
			ldst_available = 0;
		} else if ((ret = PAPI_add_event(event_set, PAPI_SR_INS)) < 0) {
			fprintf(stderr, "ftimings: %s:%d PAPI_add_event(event_set, PAPI_SR_INS): %s\n",
					__FILE__, __LINE__, PAPI_strerror(ret));
			ldst_available = 0;
		} else {
			ldst_available = 1;
		}

		/* Start */
		if ((ret = PAPI_start(event_set)) < 0) {
			fprintf(stderr, "ftimings: %s:%d PAPI_start(): %s\n",
					__FILE__, __LINE__, PAPI_strerror(ret));
			goto error;
		}

		goto end;

error:
		/* PAPI works */
		papi_available = 0;

end:
		/* PAPI works */
		papi_available = 1;

	} /* End of critical region */

	return papi_available;
}

int ftimings_flop_init(void) {
	int ret;

	if (!tried_papi_init) {
		ftimings_papi_init();
	}

	return flops_available;
}

int ftimings_loads_stores_init(void) {
	int ret;

	if (!tried_papi_init) {
		ftimings_papi_init();
	}

	return ldst_available;
}

void ftimings_papi_counters(long long *flops, long long *ldst) {
	long long res[3];
	int i, ret;

	if ((ret = PAPI_read(event_set, &res[0])) < 0) {
		fprintf(stderr, "PAPI_read: %s\n", PAPI_strerror(ret));
		exit(1);
	}

	i = 0;
	if (flops_available) {
		*flops = res[i++];
	} else {
		*flops = 0LL;
	}
	if (ldst_available) {
		*ldst = res[i++];
		*ldst += res[i++];
	} else {
		*ldst = 0LL;
	}
}
#endif
