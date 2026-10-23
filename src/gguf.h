#ifndef GGUF_H
#define GGUF_H
/* ---- minimal GGUF v2/v3 reader (llama.cpp file format) ---- */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGUF_MAGIC 0x46554747u /* "GGUF" LE */

/* ggml tensor (block-quantization) type codes */
#define GGML_F32   0
#define GGML_F16   1
#define GGML_Q4_0  2
#define GGML_Q4_1  3
#define GGML_Q5_0  6
#define GGML_Q5_1  7
#define GGML_Q8_0  8
#define GGML_Q2_K 10
#define GGML_Q3_K 11
#define GGML_Q4_K 12
#define GGML_Q5_K 13
#define GGML_Q6_K 14
#define GGML_Q8_K 15

const char *ggml_type_name(int t);

/* one key/value entry from the metadata store */
typedef struct {
    char     *key;   /* NULL for skipped per-tensor kvs */
    int       type;  /* 0..12  (see GGUF spec)          */
    uint64_t  u;     /* u8/u16/u32/u64/bool             */
    int64_t   i;     /* i8/i16/i32/i64                  */
    float     f;     /* f32 (f64 truncated)             */
    char     *s;     /* string value (owned)            */
    /* arrays (type 9): exactly one of the below is filled */
    uint32_t  arr_count;
    char    **arr_str;   /* elem type 8  */
    float    *arr_f32;   /* elem type 6  */
    uint32_t *arr_u32;   /* elem type 4  */
    uint8_t  *arr_raw;   /* any other elem type */
} gguf_kv;

typedef struct {
    char     *name;
    uint32_t  n_dims;
    uint64_t *dims;   /* dims[0] = first dim of row-major layout */
    uint32_t  type;   /* ggml type */
    uint64_t  offset; /* absolute file offset of payload */
} gguf_tensor;

typedef struct {
    char          magic[4];
    uint32_t      version;
    uint64_t      n_tensors, n_kv;
    gguf_kv      *kvs;      /* n_kv entries (heap)   */
    gguf_tensor  *tensors;  /* n_tensors entries (heap) */
    uint64_t      data_start; /* absolute file offset of data section start */
} gguf_meta;

int  gguf_load (FILE *f, gguf_meta *m);      /* parse header+kv+tensor table */
void gguf_free (gguf_meta *m);
gguf_kv *gguf_get (gguf_meta *m, const char *key);        /* find kv by key  */
gguf_tensor *gguf_find (gguf_meta *m, const char *name);  /* find tensor     */
/* read a F32/F16 tensor payload into caller buffer (host little-endian floats) */
int gguf_read_float (FILE *f, gguf_meta *m, gguf_tensor *t, float *out);

#ifdef __cplusplus
}
#endif
#endif