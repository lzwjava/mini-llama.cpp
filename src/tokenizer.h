#ifndef TOKENIZER_H
#define TOKENIZER_H
#include <stdint.h>
#include "gguf.h"

typedef struct {
    int      n;             /* vocab size                  */
    char   **tokens;        /* VIEW over gguf vocab (not owned - gguf_meta frees) */
    float   *scores;
    uint8_t *tok_type;
    int     *sorted;        /* token ids sorted by string  */
    int     byte_map[256];  /* GPT2 bytes_to_unicode byte->token  */
    int      n_merges;
    int     *m0, *m1, *mid; /* merge lhs/rhs/result ids    */
    int      bos, eos;
} tokenizer;

int  tokenizer_init(tokenizer *t, gguf_meta *m);
void tokenizer_free(tokenizer *t);
int  tokenizer_encode(tokenizer *t, const char *text, int *out_ids, int max_ids);
char *tokenizer_decode(tokenizer *t, const int *ids, int n);
#endif