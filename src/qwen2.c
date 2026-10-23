#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "qwen2.h"
#include "gguf.h"
#include "quant.h"

/* ================= config ================= */
int qwen2_cfg_from_gguf(gguf_meta *m, qwen2_cfg *cfg){
    memset(cfg, 0, sizeof(*cfg));
    gguf_kv *v;
    v = gguf_get(m,"qwen2.block_count");                  cfg->n_layer = v? (int)v->u : 24;
    v = gguf_get(m,"qwen2.embedding_length");             cfg->n_embd  = v? (int)v->u : 896;
    v = gguf_get(m,"qwen2.feed_forward_length");          cfg->ffn     = v? (int)v->u : 4864;
    v = gguf_get(m,"qwen2.attention.head_count");         cfg->n_head  = v? (int)v->u : 14;
    v = gguf_get(m,"qwen2.attention.head_count_kv");     cfg->n_kv    = v? (int)v->u : 2;
    v = gguf_get(m,"qwen2.context_length");               cfg->n_ctx   = v? (int)v->u : 32768;
    v = gguf_get(m,"qwen2.attention.layer_norm_rms_epsilon"); cfg->eps  = v? v->f : 1e-6f;
    v = gguf_get(m,"qwen2.rope.freq_base");               cfg->rope_base = v? v->f : 1e6f;
    if (cfg->n_head<=0 || cfg->n_embd % cfg->n_head) cfg->n_head = 14;
    cfg->head_dim = cfg->n_embd / cfg->n_head;
    return 0;
}

/* ================= model load ================= */
static float *dequant_tensor_file(const char *path, gguf_meta *m, const char *name){
    gguf_tensor *t = gguf_find(m, name);
    if (!t) return NULL;
    uint64_t ne = 1; for (uint32_t d=0;d<t->n_dims;d++) ne *= t->dims[d];
    FILE *f = fopen(path,"rb"); if (!f) return NULL;
    quant_spec qs = quant_spec_of(t->type);
    size_t nb = (size_t)(ne/qs.elems)*qs.bytes;
    uint8_t *raw = (uint8_t*)malloc(nb);
    if (fseek(f,(long)(m->data_start + t->offset),SEEK_SET)!=0 || fread(raw,1,nb,f)!=nb){
        free(raw); fclose(f); return NULL;
    }
    fclose(f);
    float *out = (float*)malloc((size_t)ne*4);
    if (dequantize_tensor(raw, t->type, ne, out) != 0){ free(out); free(raw); return NULL; }
    free(raw);
    return out;
}

/* rows = dims[last] (vocab / output features), cols = dims[0] (input features) */
static float *load_w(char *buf, size_t buflen, const char *path, gguf_meta *m,
                     const char *name, int rows, int cols, const char *tag){
    buflen = buflen; (void)buflen; (void)tag;
    float *w = dequant_tensor_file(path, m, name);
    if (!w){ fprintf(stderr, "  load_w FAILED: %s\n", name); return NULL; }
    gguf_tensor *t = gguf_find(m, name);
    uint64_t ne = 1; for (uint32_t d=0;d<t->n_dims;d++) ne*=t->dims[d];
    if ((int64_t)ne != (int64_t)rows*cols){
        fprintf(stderr, "  shape %s: got %llu want %d*%d\n", name, (unsigned long long)ne, rows, cols);
        free(w); return NULL;
    }
    return w;
}

int qwen2_cfg_discover_vocab(gguf_meta *m, qwen2_cfg *cfg){
    gguf_tensor *te = gguf_find(m, "token_embd.weight");
    if (!te) return -1;
    /* dims are reversed: contiguous dim first; rows (vocab) = last dim */
    cfg->n_vocab = (int)te->dims[te->n_dims-1];
    return 0;
}

int qwen2_load(const char *path, gguf_meta *m, qwen2_model *mdl){
    memset(mdl, 0, sizeof(*mdl));
    qwen2_cfg_from_gguf(m, &mdl->cfg);
    qwen2_cfg *c = &mdl->cfg;
    qwen2_cfg_discover_vocab(m, c);

    int E = c->n_embd, F = c->ffn, KV = c->n_kv*c->head_dim, V = c->n_vocab;
    mdl->embd   = load_w(NULL,0,path,m,"token_embd.weight", V, E, "embd");
    mdl->output = load_w(NULL,0,path,m,"output.weight",     V, E, "output");
    mdl->out_norm = load_w(NULL,0,path,m,"output_norm.weight", 1, E, "out_norm");
    if (!mdl->embd || !mdl->output || !mdl->out_norm){ fprintf(stderr,"failed to load embd/output/out_norm\n"); return -1; }
    mdl->layers = (qwen2_layer*)calloc((size_t)c->n_layer, sizeof(qwen2_layer));
    char nm[128];
    for (int i=0;i<c->n_layer;i++){
        qwen2_layer *L = &mdl->layers[i];
        snprintf(nm,sizeof(nm),"blk.%d.attn_norm.weight",   i); L->a_norm = load_w(NULL,0,path,m,nm, 1, E, "a_norm");
        snprintf(nm,sizeof(nm),"blk.%d.attn_q.weight",      i); L->wq     = load_w(NULL,0,path,m,nm, E, E, "wq");
        snprintf(nm,sizeof(nm),"blk.%d.attn_k.weight",      i); L->wk     = load_w(NULL,0,path,m,nm, KV, E, "wk");
        snprintf(nm,sizeof(nm),"blk.%d.attn_v.weight",      i); L->wv     = load_w(NULL,0,path,m,nm, KV, E, "wv");
        snprintf(nm,sizeof(nm),"blk.%d.attn_output.weight", i); L->wo     = load_w(NULL,0,path,m,nm, E, E, "wo");
        snprintf(nm,sizeof(nm),"blk.%d.ffn_norm.weight",    i); L->f_norm = load_w(NULL,0,path,m,nm, 1, E, "f_norm");
        snprintf(nm,sizeof(nm),"blk.%d.ffn_gate.weight",    i); L->wg     = load_w(NULL,0,path,m,nm, F, E, "wg");
        snprintf(nm,sizeof(nm),"blk.%d.ffn_up.weight",      i); L->wu     = load_w(NULL,0,path,m,nm, F, E, "wu");
        snprintf(nm,sizeof(nm),"blk.%d.ffn_down.weight",    i); L->wd     = load_w(NULL,0,path,m,nm, E, F, "wd");
        if (!L->a_norm||!L->wq||!L->wk||!L->wv||!L->wo||!L->f_norm||!L->wg||!L->wu||!L->wd){
            fprintf(stderr,"layer %d load failed\n", i); return -1;
        }
    }
    return 0;
}

/* ================= primitives (CPU) ================= */
static void matmul_cpu(const float *W, int rows, int cols, const float *x, float *y){
    for (int r=0;r<rows;r++){
        float s=0; const float *wp = W + (size_t)r*cols;
        for (int k=0;k<cols;k++) s += wp[k]*x[k];
        y[r]=s;
    }
}
static void rmsnorm_cpu(const float *g, const float *w, float *out, int n, float eps){
    double ss=0; for (int i=0;i<n;i++) ss += (double)g[i]*g[i];
    float inv = 1.0f/sqrtf((float)(ss/n)+eps);
    for (int i=0;i<n;i++) out[i] = w[i]*g[i]*inv;
}
static float silu(float x){ return x/(1.0f+expf(-x)); }

/* interleaved-halves RoPE (HF/Llama convention), applied to first dim of each head */
static void rope_cpu(float *vec, int pos, int head_dim, float base, int n_groups){
    for (int g=0; g<n_groups; g++){
        float *v = vec + (size_t)g*head_dim;
        int half = head_dim/2;
        for (int i=0;i<half;i++){
            double theta = (double)pos * pow((double)base, -(2.0*(double)i/(double)head_dim));
            float c = cosf((float)theta), s = sinf((float)theta);
            float x0 = v[2*i], x1 = v[2*i+1];
            v[2*i]   = x0*c - x1*s;
            v[2*i+1] = x0*s + x1*c;
        }
    }
}

/* batched matmul helper for attention: out[c] = <a, b[c*B + ]> */
/* ================= one token forward (with KV cache) ================= */
/* k_cache/v_cache: per layer, [ctx, KV] contiguous in ctx*KV layout */
void qwen2_forward_token(qwen2_model *mdl, int token, int pos,
                         float **k_cache, float **v_cache, float *logits){
    qwen2_cfg c = mdl->cfg;
    const int E=c.n_embd, F=c.ffn, H=c.n_head, KV=c.n_kv*c.head_dim, V=c.n_vocab;
    float *x = (float*)malloc((size_t)E*4);
    float *h = (float*)malloc((size_t)E*4);   /* residual stream */
    memcpy(h, mdl->embd + (size_t)token*E, (size_t)E*4);

    float *q  = (float*)malloc((size_t)E*4);
    float *k  = (float*)malloc((size_t)KV*4);
    float *v  = (float*)malloc((size_t)KV*4);
    float *oq = (float*)malloc((size_t)E*4);
    float *g  = (float*)malloc((size_t)F*4);
    float *u  = (float*)malloc((size_t)F*4);
    float *y  = (float*)malloc((size_t)F*4);
    float *score = (float*)malloc((size_t)(pos?pos+1:1)*4);
    float *ctx  = (float*)malloc((size_t)E*4);
    int maxctx = c.n_ctx; (void)maxctx;

    for (int L=0; L<c.n_layer; L++){
        qwen2_layer *W = &mdl->layers[L];
        /* attn */
        rmsnorm_cpu(h, W->a_norm, x, E, c.eps);
        matmul_cpu(W->wq, E, E, x, q);
        matmul_cpu(W->wk, KV, E, x, k);
        matmul_cpu(W->wv, KV, E, x, v);
        rope_cpu(q, pos, c.head_dim, c.rope_base, H);
        rope_cpu(k, pos, c.head_dim, c.rope_base, c.n_kv);
        memcpy(k_cache[L] + (size_t)pos*KV, k, (size_t)KV*4);
        memcpy(v_cache[L] + (size_t)pos*KV, v, (size_t)KV*4);
        /* attention per q head */
        memset(oq, 0, (size_t)E*4);
        for (int h=0; h<H; h++){
            int kvh = h * c.n_kv / H;
            const float *qh = q + (size_t)h*c.head_dim;
            /* scores over cached positions 0..pos */
            float mx = -1e30f;
            for (int ps=0; ps<=pos; ps++){
                const float *kc = k_cache[L] + (size_t)ps*KV + (size_t)kvh*c.head_dim;
                float s = 0; for (int d=0; d<c.head_dim; d++) s += qh[d]*kc[d];
                s /= sqrtf((float)c.head_dim);
                score[ps] = s; if (s>mx) mx=s;
            }
            float Z=0;
            for (int ps=0; ps<=pos; ps++){ score[ps]=expf(score[ps]-mx); Z+=score[ps]; }
            for (int ps=0; ps<=pos; ps++) score[ps]/=Z;
            for (int ps=0; ps<=pos; ps++){
                const float *vc = v_cache[L] + (size_t)ps*KV + (size_t)kvh*c.head_dim;
                float wgt = score[ps];
                float *dst = oq + (size_t)h*c.head_dim;
                for (int d=0; d<c.head_dim; d++) dst[d] += wgt*vc[d];
            }
        }
        /* output projection + residual */
        matmul_cpu(W->wo, E, E, oq, x);
        for (int i=0;i<E;i++) h[i] += x[i];
        /* mlp */
        rmsnorm_cpu(h, W->f_norm, x, E, c.eps);
        matmul_cpu(W->wg, F, E, x, g);
        matmul_cpu(W->wu, F, E, x, u);
        for (int i=0;i<F;i++) y[i] = silu(g[i])*u[i];
        matmul_cpu(W->wd, E, F, y, x);
        for (int i=0;i<E;i++) h[i] += x[i];
    }
    /* final norm + output head */
    rmsnorm_cpu(h, mdl->out_norm, x, E, c.eps);
    matmul_cpu(mdl->output, V, E, x, logits);
    free(x); free(h); free(q); free(k); free(v); free(oq);
    free(g); free(u); free(y); free(score); free(ctx);
}

/* ================= generation ================= */
int qwen2_generate_cpu(qwen2_model *mdl, const int *ids_in, int S,
                       int n_gen, int *out_ids, int ctx_cap, float *out_logits_last){
    qwen2_cfg c = mdl->cfg;
    int cap = ctx_cap > 0 && ctx_cap < c.n_ctx ? ctx_cap : c.n_ctx;
    if (cap > 4096) cap = 4096;
    int KV = c.n_kv*c.head_dim;
    float **kc = (float**)malloc((size_t)c.n_layer*sizeof(float*));
    float **vc = (float**)malloc((size_t)c.n_layer*sizeof(float*));
    for (int L=0;L<c.n_layer;L++){
        kc[L] = (float*)calloc((size_t)cap*KV,4);
        vc[L] = (float*)calloc((size_t)cap*KV,4);
    }
    float *logits = (float*)malloc((size_t)c.n_vocab*4);
    int total = 0;
    for (int p=0;p<S;p++) out_ids[total++] = ids_in[p];
    for (int p=0;p<S+n_gen;p++){
        int token = p<S ? ids_in[p] : -1;
        if (token < 0){
            /* argmax */
            int best=0; float bv=logits[0];
            for (int i=1;i<c.n_vocab;i++) if (logits[i]>bv){ bv=logits[i]; best=i; }
            token = best;
            out_ids[total++] = token;
        }
        qwen2_forward_token(mdl, token, p, kc, vc, logits);
        if (p == S-1 && out_logits_last) memcpy(out_logits_last, logits, (size_t)c.n_vocab*4);
    }
    free(logits);
    for (int L=0;L<c.n_layer;L++){ free(kc[L]); free(vc[L]); }
    free(kc); free(vc);
    return total;
}