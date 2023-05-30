#include "alloca.h"
#include "inttypes.h"
#include "math.h"
#include "memory.h"
#include "pthread.h"
#include "stdarg.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "unistd.h"

#include "c_common.h"

Uint64 now() {
    struct timespec spec;
    clock_gettime(CLOCK_MONOTONIC_RAW, &spec);
    return (spec.tv_sec) * 1000000 + spec.tv_nsec / 1000;
}

void ensure(int expr, const char *fmt, ...) {
    // Replacement for assert with var-args and local control of compilation.
    // See ensure_only_in_debug below.
    char buffer[4096];
    va_list args;
    va_start(args, fmt);
    if(!expr) {
        vfprintf(stderr, fmt, args);
        fprintf(stderr, "\n");
        fflush(stderr);

        vsprintf(buffer, fmt, args);
        trace("ensure failed: %s\n", buffer);
        exit(1);
    }
    va_end(args);
}

FILE *_log = (FILE *)NULL;
void _trace(char *file, int line, const char *fmt, ...) {
    // Replacement for printf that also flushes
    if(!_log) {
        _log = fopen("/erisyon/plaster/_c_common.log", "wt");
    }

    va_list args;
    va_start(args, fmt);
    fprintf(_log, "@%s:%d ", file, line);
    vfprintf(_log, fmt, args);
    fflush(_log);
    va_end(args);

    va_start(args, fmt);
    fprintf(stderr, "@%s:%d ", file, line);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

Bool is_pow_2(Uint64 x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

Uint64 pow_2_to_shift(Uint64 x) {
    Uint64 shifts = 0;
    while(x) {
        if(x == 1) {
            return shifts;
        }
        x >>= 1;
        shifts++;
    }
    return 0;
}

// Hash
//=========================================================================================

Hash hash_init(HashRec *buffer, Size n_max_recs) {
    Hash hash;
    hash.recs = (HashRec *)buffer;
    memset(buffer, 0, n_max_recs * sizeof(HashRec));
    hash.n_max_recs = n_max_recs;
    hash.n_active_recs = 0;
    return hash;
}

HashRec *hash_get(Hash hash, HashKey key) {
    /*
    Assumes the hash table is large enough to never over-flow.

    Usage:
        Hash hash = hash_init(buffer, n_recs);
        HashRec *rec = hash_get(hash, key);
        if(rec == (HashRec*)0) {
            // hash full!
        }
        else if(rec->key == 0) {
            // New record
        }
        else {
            // Existing record
        }
    */
    ensure(key != 0, "Invalid hashkey");
    Index i = key % hash.n_max_recs;
    Index start_i = i;
    HashKey key_at_i = hash.recs[i].key;
    while(key_at_i != key) {
        if(key_at_i == 0) {
            // Empty slot. The caller is responsible for filling in the key
            // by checking that key is 0
            return &hash.recs[i];
        }
        i = (i + 1) % hash.n_max_recs;
        key_at_i = hash.recs[i].key;
        if(i == start_i) {
            // Overflow
            return (HashRec *)0;
        }
    }
    // Found existing
    return &hash.recs[i];
}

void hash_dump(Hash hash) {
    // Debugging
    for(Index i = 0; i < hash.n_max_recs; i++) {
        printf("%08ld: %016lX %p\n", i, hash.recs[i].key, hash.recs[i].val);
    }
}

// Tab
//----------------------------------------------------------------------------------------

Tab tab_by_size(void *base, Size n_bytes, Size n_bytes_per_row, Uint64 flags) {
    Tab tab;
    tab.base = base;
    tab.n_bytes_per_row = n_bytes_per_row;
    tab.n_max_rows = n_bytes / n_bytes_per_row;
    tab.flags = flags;
    if(flags & TAB_GROWABLE) {
        memset(tab.base, 0, n_bytes);
        tab.n_rows = 0;
    } else {
        tab.n_rows = tab.n_max_rows;
    }
    return tab;
}

Tab tab_malloc_by_size(Size n_bytes, Size n_bytes_per_row, Uint64 flags) {
    Tab tab;
    tab.base = malloc(n_bytes);
    ensure(tab.base != NULL, "malloc failed");
    memset(tab.base, 0, n_bytes);
    tab.n_bytes_per_row = n_bytes_per_row;
    tab.n_max_rows = n_bytes / n_bytes_per_row;
    tab.flags = flags;
    if(flags & TAB_GROWABLE) {
        tab.n_rows = 0;
    } else {
        tab.n_rows = tab.n_max_rows;
    }
    return tab;
}

Tab tab_by_n_rows(void *base, Size n_rows, Size n_bytes_per_row, Uint64 flags) {
    return tab_by_size(base, n_rows * n_bytes_per_row, n_bytes_per_row, flags);
}

Tab tab_malloc_by_n_rows(Size n_rows, Size n_bytes_per_row, Uint64 flags) {
    return tab_malloc_by_size(n_rows * n_bytes_per_row, n_bytes_per_row, flags);
}

Tab tab_by_arr(
    void *base, Uint64 n_rows, Uint64 n_cols, Uint64 n_bytes_per_elem,
    Uint64 flags) {
    Size n_bytes_per_row = n_cols * n_bytes_per_elem;
    return tab_by_size(
        base, n_rows * n_bytes_per_row, n_bytes_per_row,
        flags | TAB_FLAGS_HAS_ELEMS);
}

Tab tab_malloc_by_arr(
    Uint64 n_rows, Uint64 n_cols, Uint64 n_bytes_per_elem, Uint64 flags) {
    Size n_bytes_per_row = n_cols * n_bytes_per_elem;
    return tab_malloc_by_size(
        n_rows * n_bytes_per_row, n_bytes_per_row, flags | TAB_FLAGS_HAS_ELEMS);
}

void tab_free(Tab *tab) {
    free(tab->base);
    tab->base = NULL;
}

Tab _tab_subset(Tab *src, Index row_i, Size n_rows, char *file, int line) {
    ensure_only_in_debug(
        n_rows >= 0, "tab_subset @%s:%d has illegal n_rows %lu", file, line,
        n_rows);
    ensure_only_in_debug(
        0 <= row_i && row_i < src->n_rows,
        "tab_subset @%s:%d has illegal row_i %lu", file, line, row_i);
    ensure_only_in_debug(
        0 <= row_i + n_rows && row_i + n_rows <= src->n_rows,
        "tab_subset @%s:%d has illegal row_i %lu beyond end", file, line,
        row_i);

    Index last_row = row_i + n_rows;
    last_row = last_row < src->n_max_rows ? last_row : src->n_max_rows;
    n_rows = last_row - row_i;

    Tab tab;
    tab.base = src->base + src->n_bytes_per_row * row_i;
    tab.n_bytes_per_row = src->n_bytes_per_row;
    tab.n_max_rows = n_rows;
    tab.n_rows = n_rows;
    tab.flags = src->flags & (~TAB_NOT_GROWABLE);
    return tab;
}

void *_tab_get(Tab *tab, Index row_i, Uint64 flags, char *file, int line) {
    ensure_only_in_debug(
        0 <= row_i && row_i < tab->n_rows,
        "tab_get outside bounds @%s:%d requested=%lu n_rows=%lu "
        "n_bytes_per_row=%lu",
        file, line, row_i, tab->n_rows, tab->n_bytes_per_row);
    ensure_only_in_debug(
        !(flags & TAB_FLAGS_HAS_ELEMS) || ((tab->flags & TAB_FLAGS_HAS_ELEMS) &&
                                           (flags & TAB_FLAGS_HAS_ELEMS)),
        "requesting elems on a non-array tab @%s:%d", file, line);
    return (void *)(tab->base + tab->n_bytes_per_row * row_i);
}

void _tab_set(Tab *tab, Index row_i, void *src, char *file, int line) {
    ensure_only_in_debug(
        0 <= row_i && row_i < tab->n_rows,
        "tab_set outside bounds @%s:%d row_i=%lu n_rows=%lu "
        "n_bytes_per_row=%lu",
        file, line, row_i, tab->n_rows, tab->n_bytes_per_row);
    if(src != (void *)0) {
        memcpy(
            tab->base + tab->n_bytes_per_row * row_i, src,
            tab->n_bytes_per_row);
    }
}

void _tab_set_col(
    Tab *tab, Index row_i, Index col_i, void *src, char *file, int line) {
    ensure_only_in_debug(
        0 <= row_i && row_i < tab->n_rows,
        "tab_set outside rouw bounds @%s:%d row_i=%lu "
        "n_rows=%lu n_bytes_per_row=%lu",
        file, line, row_i, tab->n_rows, tab->n_bytes_per_row);
    ensure_only_in_debug(
        0 <= col_i && col_i < tab->n_cols,
        "tab_set outside col bounds @%s:%d col_i=%lu n_cols=%lu", file, line,
        col_i, tab->n_cols);
    ensure_only_in_debug(
        tab->n_bytes_per_elem > 0,
        "tab_set_col outside on non-column tab @%s:%d", file, line);
    ensure_only_in_debug(
        tab->n_cols > 0, "tab_set_col outside on non-column tab @%s:%d", file,
        line);
    if(src != (void *)0) {
        memcpy(
            tab->base + tab->n_bytes_per_row * row_i +
                col_i * tab->n_bytes_per_elem,
            src, tab->n_bytes_per_elem);
    }
}

Index _tab_add(
    Tab *tab, void *src, pthread_mutex_t *lock, char *file, int line) {
    ensure_only_in_debug(
        tab->flags & TAB_GROWABLE,
        "Attempting to grow to a un-growable table @%s:%d", file, line);
    if(lock)
        pthread_mutex_lock(lock);
    Index row_i = tab->n_rows;
    tab->n_rows++;
    if(lock)
        pthread_mutex_unlock(lock);
    ensure_only_in_debug(
        0 <= row_i && row_i < tab->n_max_rows,
        "Table overflow @%s:%d. n_max_rows=%lu", file, line, tab->n_max_rows);
    if(src != (void *)0) {
        memcpy(
            tab->base + tab->n_bytes_per_row * row_i, src,
            tab->n_bytes_per_row);
    }
    return row_i;
}

void _tab_validate(Tab *tab, void *ptr, char *file, int line) {
    // Check that a ptr is valid on a table
    ensure_only_in_debug(
        tab->base <= ptr &&
            ptr < tab->base + tab->n_bytes_per_row * tab->n_rows,
        "Tab ptr invalid @%s:%d. n_max_rows=%lu", file, line, tab->n_max_rows);
}

void tab_dump(Tab *tab, char *msg) {
    printf("table %s:\n", msg);
    printf("  base=%p\n", tab->base);
    printf("  n_bytes_per_row=%ld\n", tab->n_bytes_per_row);
    printf("  n_max_rows=%ld\n", tab->n_max_rows);
    printf("  n_rows=%ld\n", tab->n_rows);
    printf("  flags=%lx\n", tab->flags);
}

void tab_tests() {
#define N_ROWS (5)
#define N_COLS (3)

    int buf[N_ROWS * N_COLS];
    for(int r = 0; r < N_ROWS; r++) {
        for(int c = 0; c < N_COLS; c++) {
            buf[r * N_COLS + c] = r << 16 | c;
        }
    }

    Tab tab_a = tab_by_n_rows(
        buf, N_ROWS * N_COLS, N_COLS * sizeof(int), TAB_NOT_GROWABLE);
    ensure(*(int *)tab_row(&tab_a, 0) == (0), "tab_row[0,0] wrong");
    ensure(*(int *)tab_row(&tab_a, 1) == (1 << 16 | 0), "tab_row[1,1] wrong");
    ensure(
        ((int *)tab_row(&tab_a, 1))[1] == (1 << 16 | 1), "tab_row[1,1] wrong");

    tab_var(int, b, &tab_a, 1);
    ensure(b[0] == (1 << 16 | 0), "tab_var[1,0] wrong");
    ensure(b[1] == (1 << 16 | 1), "tab_var[1,1] wrong");

    tab_var(int, row_1, &tab_a, 1);
    tab_set(&tab_a, 4, row_1);
    tab_var(int, row_4, &tab_a, 1);
    ensure(row_4[0] == (1 << 16 | 0), "tab_var[4,0] wrong after copy");
    ensure(row_4[2] == (1 << 16 | 2), "tab_var[4,2] wrong after copy");

    // RESET to zeros
    memset(buf, 0, N_ROWS * N_COLS * sizeof(int));

    Tab tab_b = tab_by_n_rows(buf, N_ROWS, N_COLS * sizeof(int), TAB_GROWABLE);
    ensure(tab_b.n_rows == 0, "n_rows wrong after reset");
    ensure(tab_b.n_max_rows == N_ROWS, "n_max_rows wrong after reset");

    int row[N_COLS] = {0, 1, 2};
    tab_add(&tab_b, row, TAB_NO_LOCK);
    tab_var(int, c, &tab_b, 0);
    ensure(c[0] == 0, "c[0] wrong");
    ensure(c[1] == 1, "c[1] wrong");
    ensure(c[2] == 2, "c[2] wrong");
    ensure(tab_b.n_rows == 1, "tab_b.n_rows wrong");

    // RESET to original
    for(int r = 0; r < N_ROWS; r++) {
        for(int c = 0; c < N_COLS; c++) {
            buf[r * N_COLS + c] = r << 16 | c;
        }
    }
    Tab tab_s = tab_subset(&tab_a, 2, 2);
    tab_var(int, s, &tab_s, 0);
    ensure(s[0] == (2 << 16 | 0), "s[0] wrong");
    ensure(s[1] == (2 << 16 | 1), "s[0] wrong");
    ensure(s[2] == (2 << 16 | 2), "s[0] wrong");
    s = tab_ptr(int, &tab_s, 1);
    ensure(s[0] == (3 << 16 | 0), "s[1] wrong");
    ensure(s[1] == (3 << 16 | 1), "s[1] wrong");
    ensure(s[2] == (3 << 16 | 2), "s[1] wrong");
}

int sanity_check() {
    ensure(
        UINT64_MAX == 0xFFFFFFFFFFFFFFFFULL, "Failed sanity check: UINT64_MAX");

    // This is particularly annoying. See csim.pxd for explanation
    ensure(N_MAX_CYCLES == 64, "Failed sanity check: N_MAX_CYCLES");

    RNG rng = make_rng_with_seed(0x2000000000000000ULL);
    ensure(
        rng.state0 == 0x2000000000000000ULL,
        "RNG lower int64 did not initialize correctly");
    ensure(rng.state1 == 0, "RNG upper int64 did not initialize correctly");

    Uint64 r0 = rng_uint64(&rng);
    Uint64 r1 = rng_uint64(&rng);
    Uint64 r2 = rng_uint64(&rng);
    ensure(r0 == 0x1b5284085c9bab16, "RNG0 returned wrong value");
    ensure(r1 == 0x9f46405715e7ddff, "RNG0 returned wrong value");
    ensure(r2 == 0x17bf77c24efe4861, "RNG0 returned wrong value");

    // Size-based types (See c_common_tools.py TYPEDEFS)
    ensure(sizeof(Uint8) == 1, "Wrong size: Uint8");
    ensure(sizeof(Uint16) == 2, "Wrong size: Uint16");
    ensure(sizeof(Uint32) == 4, "Wrong size: Uint32");
    ensure(sizeof(Uint64) == 8, "Wrong size: Uint64");
    ensure(sizeof(Uint128) == 16, "Wrong size: Uint128");
    ensure(sizeof(Sint8) == 1, "Wrong size: Sint8");
    ensure(sizeof(Sint16) == 2, "Wrong size: Sint16");
    ensure(sizeof(Sint32) == 4, "Wrong size: Sint32");
    ensure(sizeof(Sint64) == 8, "Wrong size: Sint64");
    ensure(sizeof(Sint128) == 16, "Wrong size: Sint128");
    ensure(sizeof(Bool) == 8, "Wrong size: Bool");
    ensure(sizeof(Float32) == 4, "Wrong size: Float32");
    ensure(sizeof(Float64) == 8, "Wrong size: Float64");

    // Generic types (See c_common_tools.py TYPEDEFS)
    ensure(sizeof(Size) == 8, "Wrong size: Size");
    ensure(sizeof(Index) == 8, "Wrong size: Index");
    ensure(sizeof(Size32) == 4, "Wrong size: Size32");
    ensure(sizeof(Index32) == 4, "Wrong size: Index32");

    // Application-specific types (See c_common_tools.py TYPEDEFS)
    ensure(sizeof(HashKey) == 8, "Wrong size: HashKey");
    ensure(sizeof(DytType) == 1, "Wrong size: DytType");
    ensure(sizeof(PCBType) == 8, "Wrong size: PCBType");
    ensure(sizeof(CycleKindType) == 1, "Wrong size: CycleKindType");
    ensure(sizeof(DytWeightType) == 8, "Wrong size: DytWeightType");
    ensure(sizeof(RadType) == 4, "Wrong size: RadType");
    ensure(sizeof(RecallType) == 4, "Wrong size: RecallType");
    ensure(sizeof(ScoreType) == 8, "Wrong size: ScoreType");
    ensure(sizeof(DytPepType) == 4, "Wrong size: DytPepType");
    ensure(sizeof(RowKType) == 4, "Wrong size: RowKType");
    ensure(sizeof(PIType) == 8, "Wrong size: PIType");
    ensure(sizeof(IsolationType) == 4, "Wrong size: IsolationType");
    ensure(sizeof(PriorParameterType) == 8, "Wrong size: PriorParameterType");

    // Test rng_uint64 and seeding
    {
        Float64 means[5];
        for(int trial_i = 0; trial_i < 5; trial_i++) {
            RNG rng = make_rng();
            const int n_samples = 100000;
            Uint64 valsU64[n_samples];
            Uint128 sum = 0;
            for(int i = 0; i < n_samples; i++) {
                valsU64[i] = rng_uint64(&rng);
                sum += valsU64[i];
            }
            Float64 mean = (Float64)sum / (Float64)n_samples;
            Float64 expected = (Float64)UINT64_MAX / (Float64)2.0;
            Float64 ratio = expected / mean;
            ensure(
                0.99 <= ratio && ratio <= 1.01,
                "rng_uint64 not returning correct values");
            means[trial_i] = mean;
        }
        for(int trial_i = 1; trial_i < 5; trial_i++) {
            ensure(means[trial_i] != means[0], "RNG is not re-seeding");
        }
    }

    // Test rng_float64 and seeding
    {
        Float64 means[5];
        for(int trial_i = 0; trial_i < 5; trial_i++) {
            RNG rng = make_rng();
            const int n_samples = 100000;
            Float64 valsF64[n_samples];
            Float64 sum = 0.0;
            for(int i = 0; i < n_samples; i++) {
                valsF64[i] = rng_float64(&rng);
                sum += valsF64[i];
            }
            Float64 mean = (Float64)sum / (Float64)n_samples;
            ensure(
                0.49 <= mean && mean <= 0.51,
                "rng_float64 not returning correct values");
            means[trial_i] = mean;
        }
        for(int trial_i = 1; trial_i < 5; trial_i++) {
            ensure(means[trial_i] != means[0], "RNG is not re-seeding");
        }
    }

    // Test rng_normal
    {
        RNG rng = make_rng();
        // The rng_normal mean ensure fails 0.1% of the time with a
        // +/-0.07 range and 3*10^5 samples. Increased sample size
        // to reduce likelihood of failure.

        const int n_samples = 1000000;
        Float64 *valsF64 = malloc(sizeof(valsF64) * n_samples);
        Float64 sum = 0.0;
        Float64 var = 0.0;
        for(int i = 0; i < n_samples; i++) {
            valsF64[i] = rng_normal(&rng, 1.0, 10.0);
            sum += valsF64[i];
            var += (valsF64[i] - 1.0) * (valsF64[i] - 1.0);
        }
        free(valsF64);
        valsF64 = NULL;
        Float64 mean = (Float64)sum / (Float64)n_samples;
        Float64 std = sqrt(var / (Float64)n_samples);
        ensure(
            1.0 - 0.07 <= mean && mean <= 1.0 + 0.07,
            "rng_normal mean not returning correct mean values");
        ensure(
            10.0 - 0.1 <= std && std <= 10.0 + 0.1,
            "rng_normal std not returning correct std values");
    }

    return 0;
}

// U8Arr
//----------------------------------------------------------------------------------------

void u8arr_set_shape(U8Arr *arr, Size n_dims, Size *shape) {
    arr->n_dims = n_dims;
    memset(arr->shape, 0, sizeof(Size) * MAX_ARRAY_DIMS);
    memset(arr->pitch, 0, sizeof(Size) * MAX_ARRAY_DIMS);
    Size size = 1;
    ensure(0 < n_dims && n_dims <= MAX_ARRAY_DIMS, "Illegal n_dims for U8Arr");
    for(Sint64 i = n_dims - 1; i >= 0; i--) {
        arr->shape[i] = shape[i];
        arr->pitch[i] = size;
        size *= arr->shape[i];
    }
}

U8Arr u8arr(void *base, Size n_dims, Size *shape) {
    U8Arr arr;
    arr.base = base;
    u8arr_set_shape(&arr, n_dims, shape);
    return arr;
}

U8Arr u8arr_subset(U8Arr *src, Index i) {
    U8Arr arr;
    arr.base = &src->base[i * src->pitch[0]];
    u8arr_set_shape(&arr, src->n_dims - 1, &src->shape[1]);
    return arr;
}

U8Arr u8arr_malloc(Size n_dims, Size *shape) {
    Size size = 1;
    for(Index i = 0; i < n_dims; i++) {
        size *= shape[i];
    }
    Uint8 *buffer = (Uint8 *)malloc(sizeof(Uint8) * size);
    ensure(buffer != NULL, "u8arr_malloc failed");
    memset(buffer, 0, size);
    U8Arr arr;
    arr.base = (Uint8 *)buffer;
    u8arr_set_shape(&arr, n_dims, shape);
    return arr;
}

void u8arr_free(U8Arr *arr) {
    free(arr->base);
}

Uint8 *u8arr_ptr1(U8Arr *arr, Index i) {
    return &arr->base[i * arr->pitch[0]];
}

Uint8 *u8arr_ptr2(U8Arr *arr, Index i, Index j) {
    return &arr->base[i * arr->pitch[0] + j * arr->pitch[1]];
}

Uint8 *u8arr_ptr3(U8Arr *arr, Index i, Index j, Index k) {
    return &arr->base
                [i * arr->pitch[0] + j * arr->pitch[1] + k * arr->pitch[2]];
}

Uint8 *u8arr_ptr4(U8Arr *arr, Index i, Index j, Index k, Index l) {
    return &arr->base
                [i * arr->pitch[0] + j * arr->pitch[1] + k * arr->pitch[2] +
                 l * arr->pitch[3]];
}

// U16Arr
//----------------------------------------------------------------------------------------

void u16arr_set_shape(U16Arr *arr, Size n_dims, Size *shape) {
    arr->n_dims = n_dims;
    memset(arr->shape, 0, sizeof(Size) * MAX_ARRAY_DIMS);
    memset(arr->pitch, 0, sizeof(Size) * MAX_ARRAY_DIMS);
    Size size = 1;
    ensure(0 < n_dims && n_dims <= MAX_ARRAY_DIMS, "Illegal n_dims for U16Arr");
    for(Sint64 i = n_dims - 1; i >= 0; i--) {
        arr->shape[i] = shape[i];
        arr->pitch[i] = size;
        size *= arr->shape[i];
    }
}

U16Arr u16arr(void *base, Size n_dims, Size *shape) {
    U16Arr arr;
    arr.base = base;
    u16arr_set_shape(&arr, n_dims, shape);
    return arr;
}

U16Arr u16arr_subset(U16Arr *src, Index i) {
    U16Arr arr;
    arr.base = &src->base[i * src->pitch[0]];
    u16arr_set_shape(&arr, src->n_dims - 1, &src->shape[1]);
    return arr;
}

U16Arr u16arr_malloc(Size n_dims, Size *shape) {
    Size size = 1;
    for(Index i = 0; i < n_dims; i++) {
        size *= shape[i];
    }
    Uint16 *buffer = (Uint16 *)malloc(sizeof(Uint16) * size);
    ensure(buffer != NULL, "u16arr_malloc failed");
    memset(buffer, 0, size);
    U16Arr arr;
    arr.base = (Uint16 *)buffer;
    u16arr_set_shape(&arr, n_dims, shape);
    return arr;
}

void u16arr_free(U16Arr *arr) {
    free(arr->base);
}

Uint16 *u16arr_ptr1(U16Arr *arr, Index i) {
    return &arr->base[i * arr->pitch[0]];
}

Uint16 *u16arr_ptr2(U16Arr *arr, Index i, Index j) {
    return &arr->base[i * arr->pitch[0] + j * arr->pitch[1]];
}

Uint16 *u16arr_ptr3(U16Arr *arr, Index i, Index j, Index k) {
    return &arr->base
                [i * arr->pitch[0] + j * arr->pitch[1] + k * arr->pitch[2]];
}

Uint16 *u16arr_ptr4(U16Arr *arr, Index i, Index j, Index k, Index l) {
    return &arr->base
                [i * arr->pitch[0] + j * arr->pitch[1] + k * arr->pitch[2] +
                 l * arr->pitch[3]];
}

// U32Arr
//----------------------------------------------------------------------------------------

void u32arr_set_shape(U32Arr *arr, Size n_dims, Size *shape) {
    arr->n_dims = n_dims;
    memset(arr->shape, 0, sizeof(Size) * MAX_ARRAY_DIMS);
    memset(arr->pitch, 0, sizeof(Size) * MAX_ARRAY_DIMS);
    Size size = 1;
    ensure(0 < n_dims && n_dims <= MAX_ARRAY_DIMS, "Illegal n_dims for U32Arr");
    for(Sint64 i = n_dims - 1; i >= 0; i--) {
        arr->shape[i] = shape[i];
        arr->pitch[i] = size;
        size *= arr->shape[i];
    }
}

U32Arr u32arr(void *base, Size n_dims, Size *shape) {
    U32Arr arr;
    arr.base = base;
    u32arr_set_shape(&arr, n_dims, shape);
    return arr;
}

U32Arr u32arr_subset(U32Arr *src, Index i) {
    U32Arr arr;
    arr.base = &src->base[i * src->pitch[0]];
    u32arr_set_shape(&arr, src->n_dims - 1, &src->shape[1]);
    return arr;
}

U32Arr u32arr_malloc(Size n_dims, Size *shape) {
    Size size = 1;
    for(Index i = 0; i < n_dims; i++) {
        size *= shape[i];
    }
    Uint32 *buffer = (Uint32 *)malloc(sizeof(Uint32) * size);
    ensure(buffer != NULL, "u32arr_malloc failed");
    memset(buffer, 0, size);
    U32Arr arr;
    arr.base = (Uint32 *)buffer;
    u32arr_set_shape(&arr, n_dims, shape);
    return arr;
}

void u32arr_free(U32Arr *arr) {
    free(arr->base);
}

Uint32 *u32arr_ptr1(U32Arr *arr, Index i) {
    return &arr->base[i * arr->pitch[0]];
}

Uint32 *u32arr_ptr2(U32Arr *arr, Index i, Index j) {
    return &arr->base[i * arr->pitch[0] + j * arr->pitch[1]];
}

Uint32 *u32arr_ptr3(U32Arr *arr, Index i, Index j, Index k) {
    return &arr->base
                [i * arr->pitch[0] + j * arr->pitch[1] + k * arr->pitch[2]];
}

Uint32 *u32arr_ptr4(U32Arr *arr, Index i, Index j, Index k, Index l) {
    return &arr->base
                [i * arr->pitch[0] + j * arr->pitch[1] + k * arr->pitch[2] +
                 l * arr->pitch[3]];
}

// U64Arr
//----------------------------------------------------------------------------------------

void u64arr_set_shape(U64Arr *arr, Size n_dims, Size *shape) {
    arr->n_dims = n_dims;
    memset(arr->shape, 0, sizeof(Size) * MAX_ARRAY_DIMS);
    memset(arr->pitch, 0, sizeof(Size) * MAX_ARRAY_DIMS);
    Size size = 1;
    ensure(0 < n_dims && n_dims <= MAX_ARRAY_DIMS, "Illegal n_dims for U64Arr");
    for(Sint64 i = n_dims - 1; i >= 0; i--) {
        arr->shape[i] = shape[i];
        arr->pitch[i] = size;
        size *= arr->shape[i];
    }
}

U64Arr u64arr(void *base, Size n_dims, Size *shape) {
    U64Arr arr;
    arr.base = base;
    u64arr_set_shape(&arr, n_dims, shape);
    return arr;
}

U64Arr u64arr_subset(U64Arr *src, Index i) {
    U64Arr arr;
    arr.base = &src->base[i * src->pitch[0]];
    u64arr_set_shape(&arr, src->n_dims - 1, &src->shape[1]);
    return arr;
}

U64Arr u64arr_malloc(Size n_dims, Size *shape) {
    Size size = 1;
    for(Index i = 0; i < n_dims; i++) {
        size *= shape[i];
    }
    Uint64 *buffer = (Uint64 *)malloc(sizeof(Uint64) * size);
    ensure(buffer != NULL, "u64arr_malloc failed");
    memset(buffer, 0, size);
    U64Arr arr;
    arr.base = (Uint64 *)buffer;
    u64arr_set_shape(&arr, n_dims, shape);
    return arr;
}

void u64arr_free(U64Arr *arr) {
    free(arr->base);
}

Uint64 *u64arr_ptr1(U64Arr *arr, Index i) {
    return &arr->base[i * arr->pitch[0]];
}

Uint64 *u64arr_ptr2(U64Arr *arr, Index i, Index j) {
    return &arr->base[i * arr->pitch[0] + j * arr->pitch[1]];
}

Uint64 *u64arr_ptr3(U64Arr *arr, Index i, Index j, Index k) {
    return &arr->base
                [i * arr->pitch[0] + j * arr->pitch[1] + k * arr->pitch[2]];
}

Uint64 *u64arr_ptr4(U64Arr *arr, Index i, Index j, Index k, Index l) {
    return &arr->base
                [i * arr->pitch[0] + j * arr->pitch[1] + k * arr->pitch[2] +
                 l * arr->pitch[3]];
}

// F32Arr
//----------------------------------------------------------------------------------------

void f32arr_set_shape(F32Arr *arr, Size n_dims, Size *shape) {
    arr->n_dims = n_dims;
    memset(arr->shape, 0, sizeof(Size) * MAX_ARRAY_DIMS);
    memset(arr->pitch, 0, sizeof(Size) * MAX_ARRAY_DIMS);
    Size size = 1;
    ensure(0 < n_dims && n_dims <= MAX_ARRAY_DIMS, "Illegal n_dims for F32Arr");
    for(Sint32 i = n_dims - 1; i >= 0; i--) {
        arr->shape[i] = shape[i];
        arr->pitch[i] = size;
        size *= arr->shape[i];
    }
}

F32Arr f32arr(void *base, Size n_dims, Size *shape) {
    F32Arr arr;
    arr.base = base;
    f32arr_set_shape(&arr, n_dims, shape);
    return arr;
}

F32Arr f32arr_subset(F32Arr *src, Index i) {
    F32Arr arr;
    arr.base = &src->base[i * src->pitch[0]];
    f32arr_set_shape(&arr, src->n_dims - 1, &src->shape[1]);
    return arr;
}

F32Arr f32arr_malloc(Size n_dims, Size *shape) {
    Size size = 1;
    for(Index i = 0; i < n_dims; i++) {
        size *= shape[i];
    }
    Float32 *buffer = (Float32 *)malloc(sizeof(Float32) * size);
    ensure(buffer != NULL, "f32arr_malloc failed");
    memset(buffer, 0, size);
    F32Arr arr;
    arr.base = (Float32 *)buffer;
    f32arr_set_shape(&arr, n_dims, shape);
    return arr;
}

void f32arr_free(F32Arr *arr) {
    free(arr->base);
}

Float32 *f32arr_ptr1(F32Arr *arr, Index i) {
    return &arr->base[i * arr->pitch[0]];
}

Float32 *f32arr_ptr2(F32Arr *arr, Index i, Index j) {
    return &arr->base[i * arr->pitch[0] + j * arr->pitch[1]];
}

Float32 *f32arr_ptr3(F32Arr *arr, Index i, Index j, Index k) {
    return &arr->base
                [i * arr->pitch[0] + j * arr->pitch[1] + k * arr->pitch[2]];
}

Float32 *f32arr_ptr4(F32Arr *arr, Index i, Index j, Index k, Index l) {
    return &arr->base
                [i * arr->pitch[0] + j * arr->pitch[1] + k * arr->pitch[2] +
                 l * arr->pitch[3]];
}

// F64Arr
//----------------------------------------------------------------------------------------

void f64arr_set_shape(F64Arr *arr, Size n_dims, Size *shape) {
    arr->n_dims = n_dims;
    memset(arr->shape, 0, sizeof(Size) * MAX_ARRAY_DIMS);
    memset(arr->pitch, 0, sizeof(Size) * MAX_ARRAY_DIMS);
    Size size = 1;
    ensure(0 < n_dims && n_dims <= MAX_ARRAY_DIMS, "Illegal n_dims for F64Arr");
    for(Sint64 i = n_dims - 1; i >= 0; i--) {
        arr->shape[i] = shape[i];
        arr->pitch[i] = size;
        size *= arr->shape[i];
    }
}

F64Arr f64arr(void *base, Size n_dims, Size *shape) {
    F64Arr arr;
    arr.base = base;
    f64arr_set_shape(&arr, n_dims, shape);
    return arr;
}

F64Arr f64arr_subset(F64Arr *src, Index i) {
    F64Arr arr;
    arr.base = &src->base[i * src->pitch[0]];
    f64arr_set_shape(&arr, src->n_dims - 1, &src->shape[1]);
    return arr;
}

F64Arr f64arr_malloc(Size n_dims, Size *shape) {
    Size size = 1;
    for(Index i = 0; i < n_dims; i++) {
        size *= shape[i];
    }
    Float64 *buffer = (Float64 *)malloc(sizeof(Float64) * size);
    ensure(buffer != NULL, "f64arr_malloc failed");
    memset(buffer, 0, size);
    F64Arr arr;
    arr.base = (Float64 *)buffer;
    f64arr_set_shape(&arr, n_dims, shape);
    return arr;
}

void f64arr_free(F64Arr *arr) {
    free(arr->base);
}

Float64 *f64arr_ptr1(F64Arr *arr, Index i) {
    return &arr->base[i * arr->pitch[0]];
}

Float64 *f64arr_ptr2(F64Arr *arr, Index i, Index j) {
    return &arr->base[i * arr->pitch[0] + j * arr->pitch[1]];
}

Float64 *f64arr_ptr3(F64Arr *arr, Index i, Index j, Index k) {
    return &arr->base
                [i * arr->pitch[0] + j * arr->pitch[1] + k * arr->pitch[2]];
}

Float64 *f64arr_ptr4(F64Arr *arr, Index i, Index j, Index k, Index l) {
    return &arr->base
                [i * arr->pitch[0] + j * arr->pitch[1] + k * arr->pitch[2] +
                 l * arr->pitch[3]];
}

// xoshiro256** RNG -- see https://prng.di.unimi.it/xoshiro256starstar.c
//----------------------------------------------------------------------------------------

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t xnext(xRNG *xrng) {
    const uint64_t result = rotl(xrng->s[1] * 5, 7) * 9;

    const uint64_t t = xrng->s[1] << 17;

    xrng->s[2] ^= xrng->s[0];
    xrng->s[3] ^= xrng->s[1];
    xrng->s[1] ^= xrng->s[2];
    xrng->s[0] ^= xrng->s[3];

    xrng->s[2] ^= t;

    xrng->s[3] = rotl(xrng->s[3], 45);

    return result;
}
