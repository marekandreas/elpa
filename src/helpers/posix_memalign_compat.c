#include <stdint.h>
#include <stdlib.h>
#include <errno.h>

#ifdef _WIN32
int posix_memalign(void **memptr, size_t alignment, size_t size) {
    void *base;
    uintptr_t aligned;
    uintptr_t addr;

    if (memptr == NULL) {
        return EINVAL;
    }
    if ((alignment & (alignment - 1)) != 0 || alignment < sizeof(void *)) {
        return EINVAL;
    }

    base = malloc(size + alignment - 1 + sizeof(void *));
    if (base == NULL) {
        *memptr = NULL;
        return ENOMEM;
    }

    addr = (uintptr_t)base + sizeof(void *);
    aligned = (addr + alignment - 1) & ~(uintptr_t)(alignment - 1);
    ((void **)aligned)[-1] = base;
    *memptr = (void *)aligned;

    return 0;
}

void elpa_aligned_free(void *ptr) {
    if (ptr != NULL) {
        free(((void **)ptr)[-1]);
    }
}
#endif