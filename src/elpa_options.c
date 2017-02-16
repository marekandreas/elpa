#include <stdlib.h>
#include <string.h>
#include <search.h>

#define nelements(x) (sizeof(x)/sizeof(x[0]))

/* Incomplete forward declaration of configuration structure */
typedef struct elpa_config_struct elpa_config_t;

/* Function pointer type for the cardinality */
typedef int (*cardinality_t)();

/* Function pointer type to enumerate all possible options */
typedef const int (*enumerate_int_option_t)(unsigned int n);

/* Function pointer type check validity of option */
typedef int (*valid_int_option_t)(int value);

typedef struct {
	const char *name;
	cardinality_t cardinality;
	enumerate_int_option_t enumerate_option;
	valid_int_option_t valid_int_option;
} elpa_int_option_t;

/** OPTIONS **/

/* QR */
int qr_cardinality() {
	return 2;
}

const int qr_enumerate_option(unsigned int n) {
	return n;
}

int qr_valid_option(int value) {
	return value >= 0 && value < qr_cardinality();
}

/* Solver */
enum solver_type {
	ELPA_SOLVER_ELPA1,
	ELPA_SOLVER_ELPA2,
	NUM_ELPA_SOLVERS,
};

int solver_cardinality() {
	return NUM_ELPA_SOLVERS;
}

const int solver_enumerate_option(unsigned int n) {
	return n;
}

int solver_valid_option(int value) {
	return value >= 0 && value < solver_cardinality();
}


/** END OF OPTIONS **/


elpa_int_option_t elpa_int_options[] = {
	{"qr", qr_cardinality, qr_enumerate_option, qr_valid_option},
	{"solver", solver_cardinality, solver_enumerate_option, solver_valid_option},
};

struct elpa_config_struct {
	int integer_options[nelements(elpa_int_options)];
	int integer_options[nelements(elpa_int_options)];
};

int compar(const void *key, const void *member) {
	const char *name = (const char *) key;
	elpa_int_option_t *option = (elpa_int_option_t *) member;

	int l1 = strlen(option->name);
	int l2 = strlen(name);
	if (l1 != l2) {
		return 1;
	}
	if (strncmp(name, option->name, l1) == 0) {
		return 0;
	} else {
		return 1;
	}
}

int find_int_option(const char *name) {
	elpa_int_option_t *option;
	size_t nmembers = nelements(elpa_int_options);
	option = lfind((const void*) name, (const void *) &elpa_int_options, &nmembers, sizeof(elpa_int_option_t), compar);
	if (option) {
		return (option - &elpa_int_options[0]) / sizeof(elpa_int_option_t);
	} else {
		return -1;
	}
}

int* get_int_option(elpa_config_t *config, const char *name) {
	int n = find_int_option(name);
	if (n > 0) {
		return &(config->integer_options[n]);
	} else {
		return NULL;
	}
}

int set_int_option(elpa_config_t *config, const char *name, int value) {
	int n = find_int_option(name);
	if (n > 0) {
		config->integer_options[n] = value;
		return 1;
	} else {
		return 0;
	}
}
