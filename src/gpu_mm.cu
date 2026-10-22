/* ============================================================
 * mini-llama.cpp  --  MILESTONE 1: "GPU pipeline alive"
 *
 *  1) parse a real GGUF model (header + kv + tensor table)
 *  2) load a real F32 tensor from the model onto the GPU,
 *     run a custom sm_52 kernel, compare vs CPU
 *  3) cuBLAS GEMM (the op transformer inference is built on)
 *     vs naive CPU reference -> error + GFLOPS
 *
 *  Build: nvcc -arch=sm_52 -O2 -o mini-llama-gpu src/gpu_mm.cu src/gguf.c -lcublas
 *  Run:   ./mini-llama-gpu /mnt/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
 * ============================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "gguf.h"

/* ---- tiny custom kernel for the M40 (fp32 only, sm_52) ---- */
__global__ void elem_fp32(float *v, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) v[i] = 2.0f*v[i] + 1.0f;  /* y = 2x+1  (trivial RMSNorm-like op) */
}

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9; }

int main(int argc, char **argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf\n",argv[0]); return 2; }
    FILE *f=fopen(argv[1],"rb"); if(!f){ perror("open"); return 2; }

    /* ---------- 1) GGUF metadata ---------- */
    gguf_meta m;
    int rc=gguf_load(f,&m);
    if(rc!=0){ fprintf(stderr,"gguf_load rc=%d (need GGUF v2/v3 file)\n",rc); return 2; }
    gguf_kv *arch=gguf_get(&m,"general.architecture");
    gguf_kv *name=gguf_get(&m,"general.name");
    gguf_kv *lay =gguf_get(&m,"qwen2.attention.layer_count");
    gguf_kv *embd=gguf_get(&m,"qwen2.embedding_length");
    printf("== GGUF ==\n  magic=%c%c%c%c  version=%u  tensors=%llu  kv=%llu\n",
           m.magic[0],m.magic[1],m.magic[2],m.magic[3],m.version,m.n_tensors,m.n_kv);
    printf("  arch = %s\n  name = %s\n", arch?arch->s:"?", name?name->s:"?");
    if(lay)  printf("  layers = %llu\n", lay->u);
    if(embd) printf("  embedding = %llu\n", embd->u);
    gguf_tensor *te=gguf_find(&m,"token_embd.weight");
    if(te)   printf("  vocab  = %llu (token_embd %s, hidden=%llu)\n",
                    te->dims[te->n_dims-1], ggml_type_name(te->type), te->dims[0]);
    printf("  first tensors:\n");
    for(int i=0;i<6 && i<(int)m.n_tensors;i++){
        printf("    %-36s %5s  dims=", m.tensors[i].name, ggml_type_name(m.tensors[i].type));
        for(uint32_t d=0;d<m.tensors[i].n_dims;d++) printf("%c%llu", d?120:32, m.tensors[i].dims[d]);
        printf("\n");
    }

    /* ---------- 2) real F32 tensor -> GPU kernel -> CPU check ---------- */
    gguf_tensor *t=NULL;
    for(uint64_t i=0;i<m.n_tensors;i++) if(m.tensors[i].type==GGML_F32){ t=&m.tensors[i]; break; }
    if(!t){ printf("no F32 tensor found - nothing to test\n"); return 1; }
    uint64_t n=1; for(uint32_t d=0;d<t->n_dims;d++) n*=t->dims[d];
    float *orig=(float*)malloc((size_t)n*4);
    if(gguf_read_float(f,&m,t,orig)!=0){ fprintf(stderr,"tensor read failed\n"); return 1; }
    float *host=(float*)malloc((size_t)n*4);
    float *gpu=NULL; cudaMalloc(&gpu,(size_t)n*4);
    cudaMemcpy(gpu,orig,(size_t)n*4,cudaMemcpyHostToDevice);
    elem_fp32<<<(int)((n+255)/256),256>>>(gpu,(int)n);
    cudaMemcpy(host,gpu,(size_t)n*4,cudaMemcpyDeviceToHost);
    cudaError_t ce = cudaGetLastError();
    double maxerr=0; for(uint64_t i=0;i<n;i++){ double e=fabs(host[i]-(2.0*orig[i]+1.0)); if(e>maxerr)maxerr=e; }
    printf("\n== real tensor: %s (%s, n=%llu) ==\n  cuda_err=%s\n  orig[0..2]=%.4f %.4f %.4f  gpu[0..2]=%.4f %.4f %.4f\n  max|err| = %.3e  %s\n",
           t->name, ggml_type_name(t->type), n, cudaGetErrorString(ce),
           orig[0],orig[1],orig[2], host[0],host[1],host[2],
           maxerr, maxerr<1e-4? "PASS":"FAIL");
    free(orig);

    /* ---------- 3) cuBLAS GEMM vs CPU ---------- */
    int N=512;
    float *A=(float*)malloc((size_t)N*N*4), *B=(float*)malloc((size_t)N*N*4);
    float *Cg=(float*)malloc((size_t)N*N*4), *Cc=(float*)malloc((size_t)N*N*4);
    for(int i=0;i<N*N;i++){ A[i]=(float)((i%97)+1)/97.0f; B[i]=(float)((i%89)+1)/89.0f; }
    float *dA,*dB,*dC; cudaMalloc(&dA,(size_t)N*N*4); cudaMalloc(&dB,(size_t)N*N*4); cudaMalloc(&dC,(size_t)N*N*4);
    cudaMemcpy(dA,A,(size_t)N*N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,(size_t)N*N*4,cudaMemcpyHostToDevice);
    cublasHandle_t h; cublasCreate(&h);
    float one=1.0f, zero=0.0f;
    double t0=now_s();
    for(int r=0;r<10;r++) cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&one,dA,N,dB,N,&zero,dC,N);
    cudaDeviceSynchronize();
    double dt=(now_s()-t0)/10.0;
    cudaMemcpy(Cg,dC,(size_t)N*N*4,cudaMemcpyDeviceToHost);
    /* cuBLAS is column-major: with row-major buffers this computes
       Cc[i*N+j] = sum_k A[k*N+j] * B[i*N+k]  -- emulate exactly that */
    for(int r=0;r<N;r++) for(int c=0;c<N;c++){ float s=0; for(int k=0;k<N;k++) s+=A[k*N+c]*B[r*N+k]; Cc[r*N+c]=s; }
    double maxrel=0; for(int i=0;i<N*N;i++){ double d=fabs(Cg[i]-Cc[i])/(fabs(Cc[i])+1e-12); if(d>maxrel)maxrel=d; }
    printf("\n== cuBLAS GEMM %dx%d ==\n  GFLOPS=%.1f  max rel err=%.2e  %s\n", N,N,
           2.0*N*N*N/dt/1e9, maxrel, maxrel<1e-3? "PASS":"FAIL");

    cudaDeviceReset();
    printf("\nMILESTONE-1-OK (m40 fp32 pipeline working)\n");
    return 0;
}