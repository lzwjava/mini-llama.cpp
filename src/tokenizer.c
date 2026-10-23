#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "tokenizer.h"

static gguf_kv *kv(gguf_meta *m, const char *k){ return gguf_get(m, k); }

void tokenizer_free(tokenizer *t){
    /* tokens/scores/tok_type are owned by gguf_meta; free only our tables */
    free(t->sorted); free(t->m0); free(t->m1); free(t->mid);
    memset(t, 0, sizeof(*t));
}

/* ---------- token lookup: sort ids by token string ---------- */
static void sort_tokens(char **tok, int *idx, int lo, int hi){
    if (lo >= hi) return;
    int p = idx[(lo+hi)/2]; const char *pv = tok[p];
    int i = lo, j = hi;
    while (i <= j){
        while (strcmp(tok[idx[i]], pv) < 0) i++;
        while (strcmp(tok[idx[j]], pv) > 0) j--;
        if (i <= j){ int tg = idx[i]; idx[i]=idx[j]; idx[j]=tg; i++; j--; }
    }
    sort_tokens(tok, idx, lo, j);
    sort_tokens(tok, idx, i, hi);
}
static int token_find(tokenizer *t, const char *s){
    int lo=0, hi=t->n-1;
    while (lo <= hi){
        int mid = (lo+hi)/2, id = t->sorted[mid];
        int c = strcmp(t->tokens[id], s);
        if (c == 0) return id;
        if (c < 0) lo = mid+1; else hi = mid-1;
    }
    return -1;
}

int tokenizer_init(tokenizer *t, gguf_meta *m){
    memset(t, 0, sizeof(*t));
    gguf_kv *tok = kv(m, "tokenizer.ggml.tokens");
    gguf_kv *scr = kv(m, "tokenizer.ggml.scores");
    gguf_kv *typ = kv(m, "tokenizer.ggml.token_type");
    gguf_kv *mrg = kv(m, "tokenizer.ggml.merges");
    if (!tok || !tok->arr_str || tok->arr_count == 0) return -1;
    t->n = (int)tok->arr_count;
    t->tokens = tok->arr_str;   /* view only */
    t->scores   = scr && scr->arr_f32 ? scr->arr_f32 : NULL;
    t->tok_type = typ && typ->arr_u32 ? (uint8_t*)typ->arr_u32 : NULL;

    t->sorted = (int*)malloc((size_t)t->n*sizeof(int));
    for (int i=0;i<t->n;i++) t->sorted[i]=i;
    sort_tokens(t->tokens, t->sorted, 0, t->n-1);

    /* GPT2 bytes_to_unicode byte->token map (space -> U+0120 'Ġ' etc.) */
    for (int b=0;b<256;b++) t->byte_map[b] = -1;
    {
        unsigned cs_map[256]; int n2 = 0;
        for (int b=0;b<256;b++){
            int in = (b>=0x21 && b<=0x7E) || (b>=0xA1 && b<=0xAC) || (b>=0xAE && b<=0xFF);
            cs_map[b] = in ? (unsigned)b : (unsigned)(256 + (n2++));
        }
        for (int i=0;i<t->n;i++){
            const char *s = t->tokens[i];
            unsigned len = 0; unsigned cp = 0;
            unsigned b0 = (unsigned char)s[0];
            if (b0 < 0x80){ cp = b0; len = 1; }
            else if ((b0 & 0xE0) == 0xC0 && (unsigned char)s[1] != 0){
                cp = ((b0 & 0x1F) << 6) | ((unsigned char)s[1] & 0x3F); len = 2;
                if ((unsigned char)s[1] == 0) len=0;
            }
            else if ((b0 & 0xF0) == 0xE0){ cp = ((b0 & 0x0F)<<12)|(((unsigned char)s[1]&0x3F)<<6)|((unsigned char)s[2]&0x3F); len = 3; }
            if (len == 0 || strlen(s) != len) continue;
            for (int b=0;b<256;b++) if (cs_map[b] == cp){ t->byte_map[b] = i; break; }
        }
    }

    /* merge table: (m0[k],m1[k]) -> merged id, priority = k (lower = better) */
    if (mrg && mrg->arr_str){
        t->n_merges = (int)mrg->arr_count;
        t->m0 = (int*)malloc((size_t)t->n_merges*sizeof(int));
        t->m1 = (int*)malloc((size_t)t->n_merges*sizeof(int));
        t->mid= (int*)malloc((size_t)t->n_merges*sizeof(int));
        char *a = malloc(1024); char *b = malloc(1024);
        for (int k=0;k<t->n_merges;k++){
            const char *s = mrg->arr_str[k];
            const char *sp = strchr(s, ' ');
            if (!sp){ t->m0[k]=-1; continue; }
            size_t la = (size_t)(sp-s);
            if (la == 0 || la >= 1022){ t->m0[k]=-1; continue; }
            memcpy(a, s, la); a[la]=0;
            memcpy(b, sp+1, strlen(sp+1)+1);
            int id0 = token_find(t, a);
            int id1 = token_find(t, b);
            /* merged token = pair WITHOUT the space: "a"+"b" */
            snprintf(a+strlen(a), 1024-strlen(a), "%s", b);
            int mid = token_find(t, a);
            t->m0[k]=id0; t->m1[k]=id1; t->mid[k]=mid;
        }
        free(a); free(b);
    }
    gguf_kv *b = kv(m,"tokenizer.ggml.bos_token_id");
    gguf_kv *e = kv(m,"tokenizer.ggml.eos_token_id");
    t->bos = b? (int)b->u : 151643;
    t->eos = e? (int)e->u : 151643;
    return 0;
}

static int token_of_byte(tokenizer *t, unsigned char b){
    if (t->byte_map[b] >= 0) return t->byte_map[b];
    char s[8]; snprintf(s, sizeof(s), "<0x%02X>", b);
    return token_find(t, s);
}

int tokenizer_encode(tokenizer *t, const char *text, int *out_ids, int max_ids){
    int n = (int)strlen(text);
    int *ids = (int*)malloc((size_t)(n<64?64:n)*sizeof(int));
    int len = 0;
    for (int i=0;i<n;i++){
        int id = token_of_byte(t, (unsigned char)text[i]);
        ids[len++] = id < 0 ? 0 : id;   /* rare fallback: id 0 */
    }
    /* greedy merges: repeatedly apply the highest-priority merge present */
    for (;;){
        int best_k = -1, best_i = -1;
        int nb = t->n_merges > 0 ? t->n_merges : -1;
        if (nb <= 0) break;
        for (int k=0;k<nb;k++){
            if (t->m0[k] < 0 || best_k >= 0) continue;
            for (int i=0;i<len-1;i++){
                if (ids[i]==t->m0[k] && ids[i+1]==t->m1[k]){ best_k=k; best_i=i; break; }
            }
        }
        if (best_k < 0 || t->mid[best_k] < 0) break;
        ids[best_i] = t->mid[best_k];
        for (int j=best_i+1;j<len-1;j++) ids[j]=ids[j+1];
        len--;
    }
    if (len > max_ids) len = max_ids;
    memcpy(out_ids, ids, (size_t)len*sizeof(int));
    free(ids);
    return len;
}

char *tokenizer_decode(tokenizer *t, const int *ids, int n){
    size_t cap = 4096, len = 0;
    char *out = (char*)malloc(cap);
    for (int k=0;k<n;k++){
        int id = ids[k];
        if (id < 0 || id >= t->n) continue;
        const char *tok = t->tokens[id];
        size_t tl = strlen(tok);
        if (tl == 0) continue;
        if (tl == 6 && memcmp(tok,"<0x",3)==0 && tok[5]=='>'){
            unsigned int v; if (sscanf(tok+3,"%2x",&v)==1 && v<=0xFF){
                if (len+1 >= cap){ cap*=2; out=realloc(out,cap); }
                out[len++] = (char)v; continue;
            }
        }
        if (len+tl+2 >= cap){ cap = (len+tl+2)*2; out=realloc(out,cap); }
        if (tl >= 2 && (unsigned char)tok[0]==0xC4 && (unsigned char)tok[1]==0xA0){
            out[len++] = ' '; tok += 2; tl -= 2;
        }
        memcpy(out+len, tok, tl); len += tl;
    }
    out[len] = 0;
    return out;
}