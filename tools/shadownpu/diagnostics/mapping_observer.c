#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <time.h>
#include <unistd.h>

enum mapping_role { CPU_WEIGHTS, NPU_WEIGHTS, QNN_CONTEXT, UNNAMED, OTHER, ROLE_COUNT };

static const char *const role_names[ROLE_COUNT] = {
    "cpu_weights", "npu_weights", "qnn_context", "unnamed", "other"
};
static const char *const file_options[] = {
    "SHADOWNPU_MAP_CPU", "SHADOWNPU_MAP_NPU", "SHADOWNPU_MAP_CONTEXT"
};

struct mapping_stats {
    unsigned long long virtual_kib, rss_kib, anonymous_kib, swap_kib, resident_kib;
    unsigned vmas, mincore_errors;
};

static long long monotonic_us(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (long long)now.tv_sec * 1000000 + now.tv_nsec / 1000;
}

static void snapshot(unsigned boundary) {
    long long start = monotonic_us();
    struct stat files[UNNAMED];
    struct mapping_stats stats[ROLE_COUNT] = {0};
    for (unsigned i = 0; i < UNNAMED; i++) {
        const char *path = getenv(file_options[i]);
        if (!path || stat(path, &files[i]) != 0 || !S_ISREG(files[i].st_mode)) {
            printf("SHADOWNPU_MAPPING_ERROR boundary=%u reason=file_identity role=%s\n", boundary, role_names[i]);
            return;
        }
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    FILE *maps = fopen("/proc/self/smaps", "r");
    if (!maps || page_size <= 0) {
        if (maps) fclose(maps);
        printf("SHADOWNPU_MAPPING_ERROR boundary=%u reason=smaps_open\n", boundary);
        return;
    }

    unsigned char *pages = NULL;
    size_t page_capacity = 0;
    struct mapping_stats *current = NULL;
    char *line = NULL;
    size_t line_capacity = 0;
    while (getline(&line, &line_capacity, maps) >= 0) {
        unsigned long long begin, end, inode, quantity;
        unsigned dev_major, dev_minor;
        if (sscanf(line, "%llx-%llx %*4s %*llx %x:%x %llu",
                   &begin, &end, &dev_major, &dev_minor, &inode) == 5 && end > begin) {
            enum mapping_role role = inode ? OTHER : UNNAMED;
            for (unsigned i = 0; i < UNNAMED && inode; i++) {
                if (inode == files[i].st_ino && dev_major == major(files[i].st_dev) &&
                    dev_minor == minor(files[i].st_dev)) {
                    role = (enum mapping_role)i;
                    break;
                }
            }
            current = &stats[role];
            current->vmas++;
            const size_t length = end - begin;
            current->virtual_kib += length / 1024;
            if (role >= UNNAMED) continue;

            const size_t count = (length + page_size - 1) / page_size;
            if (count > page_capacity) {
                void *replacement = realloc(pages, count);
                if (!replacement) {
                    current->mincore_errors++;
                    continue;
                }
                pages = replacement;
                page_capacity = count;
            }
            if (mincore((void *)(uintptr_t)begin, length, pages) != 0) {
                current->mincore_errors++;
                continue;
            }
            size_t resident = 0;
            for (size_t i = 0; i < count; i++) resident += pages[i] & 1;
            current->resident_kib += resident * (page_size / 1024);
        } else if (current && sscanf(line, "Rss: %llu kB", &quantity) == 1) {
            current->rss_kib += quantity;
        } else if (current && sscanf(line, "Anonymous: %llu kB", &quantity) == 1) {
            current->anonymous_kib += quantity;
        } else if (current && sscanf(line, "Swap: %llu kB", &quantity) == 1) {
            current->swap_kib += quantity;
        }
    }
    const int read_error = ferror(maps);
    fclose(maps);
    free(line);
    free(pages);
    if (read_error) {
        printf("SHADOWNPU_MAPPING_ERROR boundary=%u reason=smaps_read\n", boundary);
        return;
    }
    const long long elapsed = monotonic_us() - start;
    for (unsigned i = 0; i < ROLE_COUNT; i++) {
        const struct mapping_stats *s = &stats[i];
        printf("SHADOWNPU_MAPPING boundary=%u role=%s vmas=%u virtual_kib=%llu rss_kib=%llu "
               "anonymous_kib=%llu swap_kib=%llu resident_kib=%llu mincore_errors=%u elapsed_us=%lld\n",
               boundary, role_names[i], s->vmas, s->virtual_kib, s->rss_kib, s->anonymous_kib,
               s->swap_kib, s->resident_kib, s->mincore_errors, elapsed);
    }
}

int getrusage(int who, struct rusage *usage) {
    const int status = (int)syscall(SYS_getrusage, who, usage);
    if (status != 0 || who != RUSAGE_SELF || getpid() != gettid()) return status;
    const char *runner = getenv("SHADOWNPU_MAP_RUNNER");
    Dl_info caller;
    if (!runner || !dladdr(__builtin_return_address(0), &caller)) return status;
    const int saved_errno = errno;
    if (strcmp(caller.dli_fname, runner) != 0) {
        struct stat actual, expected;
        const int same = stat(caller.dli_fname, &actual) == 0 && stat(runner, &expected) == 0 &&
                         actual.st_dev == expected.st_dev && actual.st_ino == expected.st_ino;
        errno = saved_errno;
        if (!same) return status;
    }

    static unsigned sequence;
    const unsigned boundary = sequence++;
    if (boundary == 1 || boundary == 3 || boundary == 4 || boundary == 5 ||
        boundary == 7 || boundary == 8 || boundary == 20) {
        snapshot(boundary);
    }
    errno = saved_errno;
    return status;
}
