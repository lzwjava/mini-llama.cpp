#ifndef QUANT_H
#define QUANT_H
/* ---- GGML block-quantization dequant (reference: ggml-quants.c/ggml-common.h) ---- */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QK5_0 32
#define QK8_0 32
#define QK_K  256
#define K_SCALE_SIZE 12

/* fp16 stored as uint16 bit pattern (host LE) */
static inline float q_fp16_to_fp32(uint16_t h){
    int sign = (h & 0x8000) ? -1 : 1;
    int exp  = (h >> 10) & 0x1F;
    int man  = h & 0x3FF;
    float v;
    if (exp == 0){                  /* zero or subnormal fp16 */
        if (man == 0) v = 0.0f;
        else          v = (float)sign * (float)man * 5.9604644775390625e-08f; /* man*2^-24 */
    }
    else if (exp == 31) v = (man == 0) ? (sign > 0 ? 1e30f : -1e30f) : 0.0f;
    else { union { int bi; float bf; } u;
           u.bi = (sign < 0 ? 0x80000000 : 0) | ((exp - 15 + 127) << 23) | (man << 13); v = u.bf; }
    return v;
}

typedef struct { uint16_t d; uint8_t qh[4]; uint8_t qs[QK5_0/2]; } block_q5_0;
typedef struct { uint16_t d; int8_t  qs[QK8_0];              } block_q8_0;
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; } block_q4_K;
typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; uint16_t d; } block_q6_K;

/* (elements per block, bytes per block) for the types used by this model */
typedef struct { int elems; int bytes; } quant_spec;
quant_spec quant_spec_of(int ggml_type);

/* dequantize `raw` (blocks, GGUF-FILE layout) into `out` (ne floats).
   Supports F32(0),F16(1),Q5_0(6),Q8_0(8),Q4_K(12),Q6_K(14). Returns 0 on success. */
int dequantize_tensor(const uint8_t *raw, int ggml_type, uint64_t ne, float *out);

void dequantize_row_q5_0(const block_q5_0 *x, float *y, int64_t k);
void dequantize_row_q8_0(const block_q8_0 *x, float *y, int64_t k);
void dequantize_row_q4_K(const block_q4_K *x, float *y, int64_t k);
void dequantize_row_q6_K(const block_q6_K *x, float *y, int64_t k);

#ifdef __cplusplus
}
#endif
#endif