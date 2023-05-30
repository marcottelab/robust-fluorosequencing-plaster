#ifndef C_COMMON_H
#define C_COMMON_H

/* WARNING
 * Structures in this file may in part have been generated using
 *    plaster.tools.c_common.c_common_tools.struct_emit_header
 * Editing these structures may require careful updating of
 * corresponding *.c file to maintain consistency in the
 * C-Python interface.
 */

#include "pthread.h"
#include "stdio.h"
#include "math.h"
#include "sys/time.h"

#define N_MAX_CHANNELS_COUNT (8)

// TYPEDEFS: Must be kept in sync with c_common.c and c_common_tools.py
// See  c_common_tools.py:TYPEDEFS for more

// Size-based types (See c_common_tools.py TYPEDEFS)
typedef __uint8_t Uint8;
typedef __uint16_t Uint16;
typedef __uint32_t Uint32;
typedef __uint64_t Uint64;
typedef __uint128_t Uint128;
typedef __int8_t Sint8;
typedef __int16_t Sint16;
typedef __int32_t Sint32;
typedef __int64_t Sint64;
typedef __int128_t Sint128;
typedef __uint64_t Bool;
typedef float Float32;
typedef double Float64;

// Generic types (See c_common_tools.py TYPEDEFS)
typedef Uint64 Size;
typedef Uint64 Index;
typedef Uint32 Size32;
typedef Uint32 Index32;

// Application-specific types (See c_common_tools.py TYPEDEFS)
typedef Uint64 HashKey;
typedef Uint8 DytType;
typedef Float64 PCBType;
typedef Float64 CBBType;
typedef Uint8 CycleKindType;
typedef Uint64 DytWeightType;
typedef Float32 RadType;
typedef Float32 RecallType;
typedef Float64 ScoreType;
typedef Uint32 DytPepType;
typedef Float32 RowKType;
typedef Uint64 PIType;
typedef Float32 IsolationType;
typedef Float64 PriorParameterType;
typedef Index ChannelLocId[N_MAX_CHANNELS_COUNT];

extern Bool is_pow_2(Uint64 x);
extern Uint64 pow_2_to_shift(Uint64 x);

typedef struct {
    Float64 real;
    Float64 imag;
} F64Complex;

typedef struct {
    Size count;
    Index dyt_i;
    DytType chcy_dyt_counts[];
    // Note, this is a variable sized record
    // See dyt_* functions in sim_v2 for manipulating it
} DytRec; // Dye-track record

typedef struct {
    DytPepType dyt_i;
    DytPepType pep_i;
    DytPepType n_reads;
    DytPepType padding;
} DytPepRec;

#define DYTPEP_DYT_COL (0)
#define DYTPEP_PEP_COL (1)
#define DYTPEP_CNT_COL (2)

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

// Used for returning exception-like values from calls
#define check_and_return(expr, static_fail_string) \
    if(!(expr))                                    \
        return static_fail_string;

// Ensure
void ensure(int expr, const char *fmt, ...);
#ifdef DEBUG
    #define ensure_only_in_debug ensure
#else
    #define ensure_only_in_debug(...) ((void)0)
#endif

// Trace
extern FILE *_log;
void _trace(char *file, int line, const char *fmt, ...);
#ifdef DEBUG
    #define trace(...) _trace(__FILE__, __LINE__, __VA_ARGS__)
#else
    #define trace(...) ((void)0)
#endif

typedef void (*ProgressFn)(int complete, int total, int retry);
typedef int (*KeyboardInterruptFn)();

#define N_MAX_CHANNELS ((DytType)(N_MAX_CHANNELS_COUNT))
#define NO_LABEL ((DytType)(N_MAX_CHANNELS - 1))
#define N_MAX_CYCLES ((DytType)64)
#define CYCLE_TYPE_PRE ((CycleKindType)(0))
#define CYCLE_TYPE_MOCK ((CycleKindType)(1))
#define CYCLE_TYPE_EDMAN ((CycleKindType)(2))
#define N_MAX_NEIGHBORS (8)

Uint64 now();

#define in_bounds(x, a, b) (((a) <= (x)) && ((x) < (b)))

// Hash
//----------------------------------------------------------------------------------------

typedef Uint64 HashKey;

typedef struct {
    HashKey key;
    union {
        void *val;
        float contention_val;
    };
} HashRec;

typedef struct {
    HashRec *recs;
    Uint64 n_max_recs;
    Uint64 n_active_recs;
} Hash;

Hash hash_init(HashRec *buffer, Uint64 n_max_recs);
HashRec *hash_get(Hash hash, HashKey key);
void hash_dump(Hash hash);

// Tab
//----------------------------------------------------------------------------------------

#define TAB_NO_LOCK (void *)0

// See c_common_tools.py for duplicate defines
#define TAB_NOT_GROWABLE (0)
#define TAB_GROWABLE (1 << 0)
#define TAB_FLAGS_INT (1 << 1)
#define TAB_FLAGS_FLOAT (1 << 2)
#define TAB_FLAGS_UNSIGNED (1 << 3)
#define TAB_FLAGS_HAS_ELEMS (1 << 4)

typedef struct {
    void *base;
    Uint64 n_bytes_per_row;
    Uint64 n_max_rows;
    Uint64 n_rows;
    Uint64 n_cols; // Only applies if all columns are the same size
    Uint64 n_bytes_per_elem;
    Uint64 flags;
} Tab;

void tab_tests();
void tab_dump(Tab *tab, char *msg);
Tab _tab_subset(Tab *src, Uint64 row_i, Uint64 n_rows, char *file, int line);
Tab tab_by_n_rows(
    void *base, Uint64 n_rows, Uint64 n_bytes_per_row, Uint64 flags);
Tab tab_by_size(
    void *base, Uint64 n_bytes, Uint64 n_bytes_per_row, Uint64 flags);
Tab tab_by_arr(
    void *base, Uint64 n_rows, Uint64 n_cols, Uint64 n_bytes_per_elem,
    Uint64 flags);
Tab tab_malloc_by_n_rows(Uint64 n_rows, Uint64 n_bytes_per_row, Uint64 flags);
Tab tab_malloc_by_size(Uint64 n_bytes, Uint64 n_bytes_per_row, Uint64 flags);
void tab_free(Tab *tab);
void *_tab_get(Tab *tab, Uint64 row_i, Uint64 flags, char *file, int line);
void _tab_set(Tab *tab, Uint64 row_i, void *src, char *file, int line);
void _tab_set_col(
    Tab *tab, Uint64 row_i, Uint64 col_i, void *src, char *file, int line);
Uint64
_tab_add(Tab *tab, void *src, pthread_mutex_t *lock, char *file, int line);
void _tab_validate(Tab *tab, void *ptr, char *file, int line);

#define tab_n_rows_remain(tab) ((tab)->n_max_rows - (tab)->n_rows)
#define tab_row(tab, row_i) _tab_get(tab, row_i, 0, __FILE__, __LINE__)
#define tab_var(typ, var, tab, row_i) \
    typ *var = (typ *)_tab_get(tab, row_i, 0, __FILE__, __LINE__)
#define tab_ptr(typ, tab, row_i) \
    (typ *)_tab_get(tab, row_i, 0, __FILE__, __LINE__)
#define tab_get(typ, tab, row_i) \
    *(typ *)_tab_get(tab, row_i, 0, __FILE__, __LINE__)
#define tab_col(typ, tab, row_i, col_i) \
    ((typ *)_tab_get(                   \
        tab, row_i, TAB_FLAGS_HAS_ELEMS, __FILE__, __LINE__))[col_i]
#define tab_set(tab, row_i, src_ptr) \
    _tab_set(tab, row_i, src_ptr, __FILE__, __LINE__)
#define tab_set_col(tab, row_i, col_i, src_ptr) \
    _tab_set_col(tab, row_i, col_i, src_ptr, __FILE__, __LINE__)
#define tab_add(tab, src, lock) _tab_add(tab, src, lock, __FILE__, __LINE__)
#define tab_validate(tab, ptr) _tab_validate(tab, ptr, __FILE__, __LINE__)
#define tab_subset(src, row_i, n_rows) \
    _tab_subset(src, row_i, n_rows, __FILE__, __LINE__)

#ifdef DEBUG
    #define tab_validate_only_in_debug(tab, ptr) \
        _tab_validate(tab, ptr, __FILE__, __LINE__)
#else
    #define tab_validate_only_in_debug(...) ((void)0)
#endif

#define tab_alloca(table_name, n_rows, n_bytes_per_row)             \
    void *buf##__LINE__ = (void *)alloca(n_rows * n_bytes_per_row); \
    memset(buf##__LINE__, 0, n_rows *n_bytes_per_row);              \
    Tab table_name = tab_by_n_rows(                                 \
        buf##__LINE__, n_rows, n_bytes_per_row, TAB_NOT_GROWABLE)

// U8Arr
//----------------------------------------------------------------------------------------

#define MAX_ARRAY_DIMS (4)

typedef struct {
    Uint8 *base;
    Size n_dims;

    // Shape is the number of elements in each dimensions
    // or zero if none.
    Size shape[MAX_ARRAY_DIMS];

    // pitch is the product of all subordinate shapes
    // (ie, the amount you need to add to an index of that
    // dimension to get to the next element).
    Size pitch[MAX_ARRAY_DIMS];
} U8Arr;

void u8arr_set_shape(U8Arr *arr, Size n_dims, Size *shape);
U8Arr u8arr(void *base, Size n_dims, Size *shape);
U8Arr u8arr_subset(U8Arr *src, Index i);
U8Arr u8arr_malloc(Size n_dims, Size *shape);
void u8arr_free(U8Arr *arr);

Uint8 *u8arr_ptr1(U8Arr *arr, Index i);
Uint8 *u8arr_ptr2(U8Arr *arr, Index i, Index j);
Uint8 *u8arr_ptr3(U8Arr *arr, Index i, Index j, Index k);
Uint8 *u8arr_ptr4(U8Arr *arr, Index i, Index j, Index k, Index l);

// U16Arr
//----------------------------------------------------------------------------------------

#define MAX_ARRAY_DIMS (4)

typedef struct {
    Uint16 *base;
    Size n_dims;

    // Shape is the number of elements in each dimensions
    // or zero if none.
    Size shape[MAX_ARRAY_DIMS];

    // pitch is the product of all subordinate shapes
    // (ie, the amount you need to add to an index of that
    // dimension to get to the next element).
    Size pitch[MAX_ARRAY_DIMS];
} U16Arr;

void u16arr_set_shape(U16Arr *arr, Size n_dims, Size *shape);
U16Arr u16arr(void *base, Size n_dims, Size *shape);
U16Arr u16arr_subset(U16Arr *src, Index i);
U16Arr u16arr_malloc(Size n_dims, Size *shape);
void u16arr_free(U16Arr *arr);

Uint16 *u16arr_ptr1(U16Arr *arr, Index i);
Uint16 *u16arr_ptr2(U16Arr *arr, Index i, Index j);
Uint16 *u16arr_ptr3(U16Arr *arr, Index i, Index j, Index k);
Uint16 *u16arr_ptr4(U16Arr *arr, Index i, Index j, Index k, Index l);

// U32Arr
//----------------------------------------------------------------------------------------

#define MAX_ARRAY_DIMS (4)

typedef struct {
    Uint32 *base;
    Size n_dims;

    // Shape is the number of elements in each dimensions
    // or zero if none.
    Size shape[MAX_ARRAY_DIMS];

    // pitch is the product of all subordinate shapes
    // (ie, the amount you need to add to an index of that
    // dimension to get to the next element).
    Size pitch[MAX_ARRAY_DIMS];
} U32Arr;

void u32arr_set_shape(U32Arr *arr, Size n_dims, Size *shape);
U32Arr u32arr(void *base, Size n_dims, Size *shape);
U32Arr u32arr_subset(U32Arr *src, Index i);
U32Arr u32arr_malloc(Size n_dims, Size *shape);
void u32arr_free(U32Arr *arr);

Uint32 *u32arr_ptr1(U32Arr *arr, Index i);
Uint32 *u32arr_ptr2(U32Arr *arr, Index i, Index j);
Uint32 *u32arr_ptr3(U32Arr *arr, Index i, Index j, Index k);
Uint32 *u32arr_ptr4(U32Arr *arr, Index i, Index j, Index k, Index l);

// U64Arr
//----------------------------------------------------------------------------------------

typedef struct {
    Uint64 *base;
    Size n_dims;

    // Shape is the number of elements in each dimensions
    // or zero if none.
    Size shape[MAX_ARRAY_DIMS];

    // pitch is the product of all subordinate shapes
    // (ie, the amount you need to add to an index of that
    // dimension to get to the next element).
    Size pitch[MAX_ARRAY_DIMS];
} U64Arr;

void u64arr_set_shape(U64Arr *arr, Size n_dims, Size *shape);
U64Arr u64arr(void *base, Size n_dims, Size *shape);
U64Arr u64arr_subset(U64Arr *src, Index i);
U64Arr u64arr_malloc(Size n_dims, Size *shape);
void u64arr_free(U64Arr *arr);

Uint64 *u64arr_ptr1(U64Arr *arr, Index i);
Uint64 *u64arr_ptr2(U64Arr *arr, Index i, Index j);
Uint64 *u64arr_ptr3(U64Arr *arr, Index i, Index j, Index k);
Uint64 *u64arr_ptr4(U64Arr *arr, Index i, Index j, Index k, Index l);

// F32Arr
//----------------------------------------------------------------------------------------

typedef struct {
    Float32 *base;
    Size n_dims;

    // Shape is the number of elements in each dimensions
    // or zero if none.
    Size shape[MAX_ARRAY_DIMS];

    // pitch is the product of all subordinate shapes
    // (ie, the amount you need to add to an index of that
    // dimension to get to the next element).
    Size pitch[MAX_ARRAY_DIMS];
} F32Arr;

void f32arr_set_shape(F32Arr *arr, Size n_dims, Size *shape);
F32Arr f32arr(void *base, Size n_dims, Size *shape);
F32Arr f32arr_subset(F32Arr *src, Index i);
F32Arr f32arr_malloc(Size n_dims, Size *shape);
void f32arr_free(F32Arr *arr);

Float32 *f32arr_ptr1(F32Arr *arr, Index i);
Float32 *f32arr_ptr2(F32Arr *arr, Index i, Index j);
Float32 *f32arr_ptr3(F32Arr *arr, Index i, Index j, Index k);
Float32 *f32arr_ptr4(F32Arr *arr, Index i, Index j, Index k, Index l);

// F64Arr
//----------------------------------------------------------------------------------------

#define MAX_ARRAY_DIMS (4)

typedef struct {
    Float64 *base;
    Size n_dims;

    // Shape is the number of elements in each dimensions
    // or zero if none.
    Size shape[MAX_ARRAY_DIMS];

    // pitch is the product of all subordinate shapes
    // (ie, the amount you need to add to an index of that
    // dimension to get to the next element).
    Size pitch[MAX_ARRAY_DIMS];
} F64Arr;

void f64arr_set_shape(F64Arr *arr, Size n_dims, Size *shape);
F64Arr f64arr(void *base, Size n_dims, Size *shape);
F64Arr f64arr_subset(F64Arr *src, Index i);
F64Arr f64arr_malloc(Size n_dims, Size *shape);
void f64arr_free(F64Arr *arr);

Float64 *f64arr_ptr1(F64Arr *arr, Index i);
Float64 *f64arr_ptr2(F64Arr *arr, Index i, Index j);
Float64 *f64arr_ptr3(F64Arr *arr, Index i, Index j, Index k);
Float64 *f64arr_ptr4(F64Arr *arr, Index i, Index j, Index k, Index l);

// RNG
//----------------------------------------------------------------------------------------

typedef struct {
    Uint64 state0;
    Uint64 state1;
    Bool has_cached_normal;
    Float64 cached_normal;
} RNG;

inline RNG __attribute__((always_inline)) make_rng_with_seed(Uint128 seed) {
    RNG rng;
    rng.state0 = seed & 0xFFFFFFFFFFFFFFFF;
    rng.state1 = (seed >> 64) & 0xFFFFFFFFFFFFFFFF;
    rng.has_cached_normal = 0;
    rng.cached_normal = NAN;
    return rng;
}

inline RNG __attribute__((always_inline)) make_rng() {
    struct timeval time;
    gettimeofday(&time, NULL);
    Uint128 seed =
        (Uint128)time.tv_sec * (Uint128)100000 + (Uint128)time.tv_usec;
    return make_rng_with_seed(seed);
}

inline Uint64 __attribute__((always_inline)) rng_uint64(RNG *rng) {
    Uint128 *ptr = (Uint128 *)rng;
    *ptr *= (Uint128)0xda942042e4dd58b5;
    return (Uint64)(*ptr >> 64);
}

inline int __attribute__((always_inline)) rng_p_i(RNG *rng, Uint64 p_i) {
    // Return 1 if a random val is < p_i else 0
    // p_i is a unsigned 64-bit probability,
    // i.e. when p_i is small this function is likely to return 0
    Uint64 r = rng_uint64(rng);
    return r < p_i ? 1 : 0;
}

inline int __attribute__((always_inline))
rng_p_iz(RNG *rng, Uint64 *p_iz, Uint64 count) {
    // Return index i into p_iz which satisfies rng < p_iz[i], or count if none
    // do. p_iz is an array of cumulative unsigned 64-bit "probabilities"
    // Typically used to choose an outcome from a list whose cumulative prob
    // == 1.0 The last value may be omitted which consumes the remaining
    // probability. e.g. if p_iz=[10,50,100,200] if r=5  => return 0 if r=15 =>
    // return 1 if r=70 => return 2 if r=1000 => return 4
    Uint64 r = rng_uint64(rng);
    for(Index i = 0; i < count; i++) {
        if(r < p_iz[i])
            return i;
    }
    return count;
}

inline Float64 __attribute__((always_inline)) rng_float64(RNG *rng) {
    // Return a float64 between 0 and 1
    return (Float64)(rng_uint64(rng) >> 1) / (Float64)((Uint64)1 << (Uint64)63);
}

inline Float64 __attribute__((always_inline))
rng_normal(RNG *rng, Float64 mu, Float64 sigma) {
    // Marsaglia polar method of the Box-Muller transform
    // https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution
    // Generate pairs of independent numbers so one is cached
    // and returned on the next call.

    Float64 ret_val;

    if(rng->has_cached_normal) {
        ret_val = rng->cached_normal;
        rng->cached_normal = NAN;
        rng->has_cached_normal = 0;
    } else {
        // Generate two independent normals. Use one, cache one
        Float64 x0, x1, r2;
        do {
            x0 = 2.0 * rng_float64(rng) - 1.0;
            x1 = 2.0 * rng_float64(rng) - 1.0;
            r2 = x0 * x0 + x1 * x1;
        } while(r2 == 0.0 || r2 >= 1.0);

        ret_val = sqrt(-2.0 * log(r2) / r2);

        // Cache second value
        rng->cached_normal = ret_val * x0;
        rng->has_cached_normal = 1;

        ret_val = ret_val * x1;
    }
    return sigma * ret_val + mu;
}

inline Float64 __attribute__((always_inline))
rng_lognormal(RNG *rng, Float64 mu, Float64 sigma) {
    // lognormal is normal on the log
    return exp(rng_normal(rng, log(mu), sigma));
}

// xoshiro RNG
//----------------------------------------------------------------------------------------

typedef struct {
    Uint64 s[4];
} xRNG;

Uint64 xnext(xRNG *xrng);

#endif
