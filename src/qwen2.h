#ifndef QWEN2_H
#define QWEN2_H
#include <stdint.h>
#include "gguf.h"

/* ---- architecture cfg (from GGUF qwen2.* kvs) ---- */
typedef struct {
    int n_layer, n_embd, n_head, n_kv, head_dim, n_vocab, ffn, n_ctx;
    float eps, rope_base;
} qwen2_cfg;

typedef struct {
    float *wq, *wk, *wv, *wo;      /* row-major (rows=out_dim, cols=n_embd) */
    float *a_norm, *f_norm;        /* [n_embd] */
    float *wg, *wu, *wd;           /* gate/up: [ffn,n_embd], down: [n_embd,ffn] */
} qwen2_layer;

typedef struct {
    qwen2_cfg cfg;
    float *embd, *output;          /* [n_vocab, n_embd] row-major */
    float *out_norm;               /* final RMSNorm weights [n_embd] */
    qwen2_layer *layers;
} qwen2_model;

/* load model: dequantize ALL tensors from gguf into fp32 (CPU) */
int qwen2_load(const char *path, gguf_meta *m, qwen2_model *mdl);
int qwen2_cfg_from_gguf(gguf_meta *m, qwen2_cfg *cfg);

/* CPU greedy generation with KV cache.
   ids_in: prompt token ids (S), out_ids: caller buffer S+n_gen.
   ids_in[0..S-1] are consumed with cache, then n_gen tokens are generated.
   min(.,) of tokens returned. */
int qwen2_generate_cpu(qwen2_model *mdl, const int *ids_in, int S,
                       int n_gen, int *out_ids, int ctx_cap, float *out_logits_last);

/* one-token forward (CPU) with cache filled up to position pos */
void qwen2_forward_token(qwen2_model *mdl, int token, int pos,
                         float **k_cache, float **v_cache, float *logits);
#endif