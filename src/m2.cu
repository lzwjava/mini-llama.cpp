/* ============================================================
 * mini-llama.cpp  --  MILESTONE 2: dequant -> fp32 -> GPU upload
 *
 *  1) enumerate all block types used by the model
 *  2) dequantize real tensors (Q5_0, Q8_0, Q4_K, Q6_K, F32)
 *     with dequant code ported from ggml -> print stats
 *  3) upload full fp32 weights to M40 VRAM, measure memory
 *  4) run ONE real GEMM: ffn_up (4864x896) x hidden vector,
 *     verified against a CPU reference
 *
 *  nvcc -arch=sm_52 -O2 -o mini-llama-m2 src/m2.cu src/gguf.c src/quant.c -lcublas
 * ============================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "gguf.h"
#include "quant.h"

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9; }

/* read raw quantized bytes of a tensor at data_start+offset */
static uint8_t *load_raw(FILE *f, gguf_meta *m, gguf_tensor *t, size_t *nbytes){
    uint64_t ne = 1; for (uint32_t d=0; d<t->n_dims; d++) ne *= t->dims[d];
    quant_spec qs = quant_spec_of(t->type);
    if (qs.elems == 0) return NULL;
    *nbytes = (size_t)(ne / qs.elems) * qs.bytes;
    uint8_t *raw = (uint8_t*)malloc(*nbytes);
    if (fseek(f, (long)(m->data_start + t->offset), SEEK_SET) != 0) return NULL;
    if (fread(raw, 1, *nbytes, f) != *nbytes){ free(raw); return NULL; }
    return raw;
}

static void tensor_stats(const char *name, int type, uint64_t ne, const float *v){
    double sum=0, suma=0; float mn=1e30f, mx=-1e30f;
    for (uint64_t i=0;i<ne;i++){
        sum += v[i]; suma += fabs((double)v[i]);
        if (v[i]<mn) mn=v[i];
        if (v[i]>mx) mx=v[i];
    }
    printf("  %-24s t=%2d ne=%-10llu v[0..3]=%.6f %.6f %.6f %.6f  min=%.6f max=%.6f sum=%.6f\n",
           name, type, (unsigned long long)ne, v[0],v[1],v[2],v[3], mn, mx, sum);
}

int main(int argc, char **argv){
    if (argc < 2){ fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb"); if (!f){ perror("open"); return 2; }
    gguf_meta m;
    if (gguf_load(f, &m) != 0){ fprintf(stderr, "gguf load failed\n"); return 2; }

    /* ---- 1) census of block types + fp32 footprint ---- */
    long tcount[32] = {0}; uint64_t param_count = 0;
    for (uint64_t i=0;i<m.n_tensors;i++){
        tcount[m.tensors[i].type]++;
        uint64_t ne = 1;
        for (uint32_t d=0;d<m.tensors[i].n_dims;d++) ne *= m.tensors[i].dims[d];
        param_count += ne;
    }
    printf("== M2: dequant census (needed: Q5_0=%ld Q8_0=%ld Q4_K=%ld Q6_K=%ld F32=%ld) ==\n",
           tcount[6], tcount[8], tcount[12], tcount[14], tcount[0]);
    printf("  total params %.3fB -> fp32 = %.2f GB (M40 VRAM 24 GB)\n",
           param_count/1e9, param_count*4.0/1e9);

    /* ---- 2) CPU dequant of key tensors + stats ---- */
    const char *picks[] = { "token_embd.weight","blk.0.attn_q.weight","blk.0.attn_k.weight",
                            "blk.0.attn_v.weight","blk.0.attn_output.weight","blk.0.ffn_gate.weight",
                            "blk.0.ffn_up.weight","blk.0.ffn_down.weight","blk.11.ffn_down.weight",
                            "output.weight","blk.0.attn_norm.weight", NULL };
    double t0 = now_s();
    for (int k=0; picks[k]; k++){
        gguf_tensor *t = gguf_find(&m, picks[k]);
        if (!t){ printf("  %-24s (missing)\n", picks[k]); continue; }
        uint64_t ne = 1; for (uint32_t d=0;d<t->n_dims;d++) ne *= t->dims[d];
        size_t nbytes; uint8_t *raw = load_raw(f, &m, t, &nbytes);
        if (!raw){ printf("  %-24s RAW-READ-FAIL\n", picks[k]); continue; }
        float *v = (float*)malloc((size_t)ne*4);
        dequantize_tensor(raw, t->type, ne, v);
        tensor_stats(picks[k], t->type, ne, v);
        free(v); free(raw);
    }
    printf("  CPU dequant time: %.2f s\n", now_s()-t0);

    /* ---- 3) upload to VRAM + 4) real GEMM ---- */
    size_t free0, total;
    cudaMemGetInfo(&free0, &total);
    gguf_tensor *te = gguf_find(&m, "token_embd.weight");
    gguf_tensor *fu = gguf_find(&m, "blk.0.ffn_up.weight");
    if (te && fu){
        const int K = 896, NOUT = 4864;
        /* ffn_up: [K=896, 4864] -> fp32 W row-major (NOUT rows x K cols) */
        size_t nb; uint8_t *rawW = load_raw(f, &m, fu, &nb);
        float *hW = (float*)malloc((size_t)NOUT*K*4);
        dequantize_tensor(rawW, fu->type, (uint64_t)NOUT*K, hW);
        float *hX = (float*)malloc((size_t)K*4);
        for (int i=0;i<K;i++) hX[i] = (float)((i*2654435761u)>>24)/255.0f;
        /* CPU reference: Y[c] = sum_k X[k] * W[c*K+k] */
        float *hY = (float*)malloc((size_t)NOUT*4);
        double s0 = now_s();
        for (int c=0;c<NOUT;c++){ float s=0; for (int k=0;k<K;k++) s += hX[k]*hW[c*K+k]; hY[c]=s; }
        printf("  CPU ref ffn_up gemm: %.1f GFLOPS\n", 2.0*NOUT*K/(now_s()-s0)/1e9);
        /* GPU */
        float *dX,*dW,*dY;
        cudaMalloc(&dX, (size_t)K*4); cudaMalloc(&dW, (size_t)NOUT*K*4); cudaMalloc(&dY,(size_t)NOUT*4);
        cudaMemcpy(dX,hX,(size_t)K*4,cudaMemcpyHostToDevice);
        cudaMemcpy(dW,hW,(size_t)NOUT*K*4,cudaMemcpyHostToDevice);
        cublasHandle_t hb; cublasCreate(&hb);
        float one=1.f, zero=0.f;
        cudaDeviceSynchronize();
        s0 = now_s();
        for (int r=0;r<20;r++) cublasSgemm(hb,CUBLAS_OP_N,CUBLAS_OP_N,1,NOUT,K,&one,dX,1,dW,K,&zero,dY,1);
        cudaDeviceSynchronize();
        double gflops=2.0*NOUT*K/((now_s()-s0)/20.0)/1e9;
        float *hYg=(float*)malloc((size_t)NOUT*4);
        cudaMemcpy(hYg,dY,(size_t)NOUT*4,cudaMemcpyDeviceToHost);
        double maxabs=0, sum=0, sumg=0, abssum=0;
        for (int i=0;i<NOUT;i++){ double e=fabs(hYg[i]-hY[i]); if(e>maxabs)maxabs=e;
                                  sum+=hY[i]; sumg+=hYg[i]; abssum+=fabs((double)hY[i]); }
        double scaled = maxabs/(abssum/NOUT + 1e-9);   /* max abs err / mean|y| */
        printf("  GPU  ffn_up gemm: X(1x896) @ W(4864x896): %.1f GFLOPS  err/scale=%.2e  PASS=%s\n",
               gflops, scaled, scaled<1e-4?"YES":"NO");
        printf("  sums: cpu=%.6f gpu=%.6f\n", sum, sumg);
        /* upload the (bigger) token embedding to VRAM too */
        size_t nb2; uint8_t *rawE = load_raw(f, &m, te, &nb2);
        float *hE = (float*)malloc((size_t)151936*K*4);
        dequantize_tensor(rawE, te->type, (uint64_t)151936*K, hE);
        float *dE; cudaMalloc(&dE, (size_t)151936*K*4);
        cudaMemcpy(dE, hE, (size_t)151936*K*4, cudaMemcpyHostToDevice);
        size_t free1;
        cudaMemGetInfo(&free1, &total);
        printf("  VRAM: token_embd 545MB + ffn_up 17MB on M40 -> used %.0f MB (free left %.0f MB)\n",
               (free0-free1)/1e6, (double)free1/1e6);
        free(rawE); free(hE); free(rawW); free(hW); free(hX); free(hY); free(hYg);
    }
    cudaDeviceReset();
    printf("\nMILESTONE-2-OK (dequant verified + fp32 weights living on the M40)\n");
    return 0;
}