#include <string.h>
#include <stdio.h>
#include "quant.h"

quant_spec quant_spec_of(int ggml_type){
    switch (ggml_type){
        case 0:  return (quant_spec){ 1,   4 };   /* F32   */
        case 1:  return (quant_spec){ 1,   2 };   /* F16   */
        case 6:  return (quant_spec){ QK5_0, sizeof(block_q5_0) };   /* Q5_0 */
        case 8:  return (quant_spec){ QK8_0, sizeof(block_q8_0) };   /* Q8_0 */
        case 12: return (quant_spec){ QK_K,  sizeof(block_q4_K) };   /* Q4_K */
        case 14: return (quant_spec){ QK_K,  sizeof(block_q6_K) };   /* Q6_K */
        default: return (quant_spec){ 0, 0 };
    }
}

/* ---- dequant implementations, ported from ggml-quants.c (exact arithmetic) ---- */

void dequantize_row_q5_0(const block_q5_0 *x, float *y, int64_t k){
    static const int qk = QK5_0;
    const int nb = (int)(k / qk);
    for (int i = 0; i < nb; i++){
        const float d = q_fp16_to_fp32(x[i].d);
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));
        for (int j = 0; j < qk/2; ++j){
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;
            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;
            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

void dequantize_row_q8_0(const block_q8_0 *x, float *y, int64_t k){
    static const int qk = QK8_0;
    const int nb = (int)(k / qk);
    for (int i = 0; i < nb; i++){
        const float d = q_fp16_to_fp32(x[i].d);
        for (int j = 0; j < qk; ++j) y[i*qk + j] = x[i].qs[j]*d;
    }
}

static inline void q4k_get_scale_min(int j, const uint8_t *q, uint8_t *d, uint8_t *m){
    if (j < 4){ *d = q[j] & 63; *m = q[j + 4] & 63; }
    else      { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
                *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4); }
}

void dequantize_row_q4_K(const block_q4_K *x, float *y, int64_t k){
    const int nb = (int)(k / QK_K);
    for (int i = 0; i < nb; i++){
        const uint8_t *q = x[i].qs;
        const float d   = q_fp16_to_fp32(x[i].d);
        const float min = q_fp16_to_fp32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64){
            q4k_get_scale_min(is + 0, x[i].scales, &sc, &m);
            const float d1 = d*sc; const float m1 = min*m;
            q4k_get_scale_min(is + 1, x[i].scales, &sc, &m);
            const float d2 = d*sc; const float m2 = min*m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

void dequantize_row_q6_K(const block_q6_K *x, float *y, int64_t k){
    const int nb = (int)(k / QK_K);
    for (int i = 0; i < nb; i++){
        const float d = q_fp16_to_fp32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128){
            for (int l = 0; l < 32; ++l){
                const int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

int dequantize_tensor(const uint8_t *raw, int ggml_type, uint64_t ne, float *out){
    switch (ggml_type){
        case 0:  memcpy(out, raw, (size_t)ne*4); return 0;                 /* F32  */
        case 1: { const uint16_t *h=(const uint16_t*)raw;                  /* F16  */
                  for (uint64_t i=0;i<ne;i++) out[i]=q_fp16_to_fp32(h[i]); return 0; }
        case 6:  dequantize_row_q5_0((const block_q5_0*)raw, out, (int64_t)ne); return 0;
        case 8:  dequantize_row_q8_0((const block_q8_0*)raw, out, (int64_t)ne); return 0;
        case 12: dequantize_row_q4_K((const block_q4_K*)raw, out, (int64_t)ne); return 0;
        case 14: dequantize_row_q6_K((const block_q6_K*)raw, out, (int64_t)ne); return 0;
        default: return -1;
    }
}