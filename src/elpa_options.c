typedef struct {
	int set;
	char *name;
	char **options;
} elpa_option_t;

elpa_option_t elpa_options[] = {
	{"useQR", {"yes", "no"}},
	{"useGPU", {"yes", "no"}},
	{"solver", {"elpa1", "elpa2"}},
}
