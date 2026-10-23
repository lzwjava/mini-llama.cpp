/* M3 CPU validation driver: tokenize -> greedy generation on CPU */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gguf.h"
#include "tokenizer.h"
#include "qwen2.h"

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9; }

int main(int argc, char **argv){
    setvbuf(stdout, NULL, _IONBF, 0);   /* don't lose output on crash */
    if (argc < 4){ fprintf(stderr,"usage: %s model.gguf \"prompt\" n_gen\n", argv[0]); return 2; }
    const char *path = argv[1];
    const char *prompt = argv[2];
    int n_gen = atoi(argv[3]);

    FILE *f = fopen(path,"rb");
    gguf_meta m; if (gguf_load(f,&m)!=0){ fprintf(stderr,"gguf load failed\n"); return 2; }
    fclose(f);

    tokenizer tk;
    if (tokenizer_init(&tk, &m)!=0){ fprintf(stderr,"tokenizer init failed\n"); return 2; }

    qwen2_model mdl;
    double t0 = now_s();
    if (qwen2_load(path,&m,&mdl)!=0){ fprintf(stderr,"model load failed\n"); return 2; }
    printf("model loaded: %d layers, %d embd, %d heads(hd=%d), %d kv, vocab %d, ffn %d  (%.1f s)\n",
           mdl.cfg.n_layer, mdl.cfg.n_embd, mdl.cfg.n_head, mdl.cfg.head_dim,
           mdl.cfg.n_kv, mdl.cfg.n_vocab, mdl.cfg.ffn, now_s()-t0);

    int ids[512];
    int S = tokenizer_encode(&tk, prompt, ids, 512);
    printf("prompt (%d chars) -> %d tokens:", (int)strlen(prompt), S);
    for (int i=0;i<S;i++) printf(" %d", ids[i]);
    printf("\n");

    int total = S + n_gen;
    int *out = (int*)malloc((size_t)total*sizeof(int));
    float *logits_last = (float*)malloc((size_t)mdl.cfg.n_vocab*4);

    t0 = now_s();
    int got = qwen2_generate_cpu(&mdl, ids, S, n_gen, out, 512, logits_last);
    printf("generated %d tokens in %.1f s (%.2f tok/s)\n", got-S, now_s()-t0, (got-S)/(now_s()-t0+1e-9));

    /* sanity: top-5 logits after prompt */
    int top[5] = {0,0,0,0,0}; float tv[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
    for (int i=0;i<mdl.cfg.n_vocab;i++){
        for (int j=0;j<5;j++) if (logits_last[i] > tv[j]){
            for (int k=4;k>j;k--){ top[k]=top[k-1]; tv[k]=tv[k-1]; }
            top[j]=i; tv[j]=logits_last[i]; break;
        }
    }
    printf("top-5 next-token: ");
    for (int j=0;j<5;j++){
        char *toktxt = tokenizer_decode(&tk, top+j, 1);
        printf("[%d '%.20s' %.2f] ", top[j], toktxt, tv[j]); free(toktxt);
    }
    printf("\n");
    char *txt = tokenizer_decode(&tk, out, got);
    printf("===== generating =====\n%s\n", txt);
    printf("===== end =====\n");
    printf("pmax=%.6f pmin=%.6f finite=%d\n", tv[0], tv[4],
           (int)(tv[0]==tv[0] && tv[0]!=(float)1.0/0.0));
    free(txt); free(out); free(logits_last);
    fprintf(stderr,"\n[cleanup]\n");
    tokenizer_free(&tk);
    fprintf(stderr,"[cleanup] tokenizer freed\n");
    gguf_free(&m);
    fprintf(stderr,"[cleanup] gguf freed\n");
    return 0;
}