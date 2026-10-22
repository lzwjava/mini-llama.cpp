#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "gguf.h"

/* ---- little-endian readers (GGUF is LE) ---- */
static uint8_t  rd_u8 (FILE*f){ uint8_t  v; fread(&v,1,1,f); return v; }
static uint16_t rd_u16(FILE*f){ uint16_t v; fread(&v,2,1,f); return v; }
static uint32_t rd_u32(FILE*f){ uint32_t v; fread(&v,4,1,f); return v; }
static uint64_t rd_u64(FILE*f){ uint64_t v; fread(&v,8,1,f); return v; }
static int64_t  rd_i64(FILE*f){ int64_t  v; fread(&v,8,1,f); return v; }
static float    rd_f32(FILE*f){ float    v; fread(&v,4,1,f); return v; }
static double   rd_f64(FILE*f){ double   v; fread(&v,8,1,f); return v; }
static char*    rd_str(FILE*f){ uint64_t n=rd_u64(f); char*s=(char*)malloc((size_t)n+1);
                                if(!s){perror("malloc"); exit(1);} fread(s,1,(size_t)n,f); s[n]=0; return s; }

/* read one key-value; for type 9 (array) we only skip the payload */
static gguf_kv read_kv(FILE *f){
    gguf_kv kv; kv.key=NULL; kv.s=NULL; kv.u=0; kv.i=0; kv.f=0.f;
    kv.key  = rd_str(f);
    kv.type = (int)rd_u32(f);  /* value type field is uint32 in GGUF v3 */
    switch (kv.type){
        case 0: case 7: kv.u=rd_u8(f);  break;
        case 1:        kv.i=rd_u8(f);   break;
        case 2:        kv.u=rd_u16(f);  break;
        case 3:        kv.i=rd_u16(f);  break;
        case 4:        kv.u=rd_u32(f);  break;
        case 5:        kv.i=rd_u32(f);  break;
        case 6:        kv.f=rd_f32(f);  break;
        case 8:        kv.s=rd_str(f);  break;
        case 9: { /* [u32 elem_type, u64 n, n * elem] */
            uint32_t et=rd_u32(f); uint64_t n=rd_u64(f);
            for(uint64_t i=0;i<n;i++)
                switch(et){
                    case 0: case 1: case 7: rd_u8(f);  break;
                    case 2: case 3:        rd_u16(f);  break;
                    case 4: case 5:        rd_u32(f);  break;
                    case 6:                rd_f32(f);  break;
                    case 10: case 11:      rd_i64(f);  break;
                    case 12:               rd_f64(f);  break;
                    case 8:  { char *x=rd_str(f); free(x); } break;
                    default: fprintf(stderr,"gguf: unsupported array elem type %u\n",et); exit(1);
                }
        } break;
        case 10: kv.u=rd_u64(f);  break;
        case 11: kv.i=rd_i64(f);  break;
        case 12: kv.f=(float)rd_f64(f); break;
        default: fprintf(stderr,"gguf: unsupported kv type %d\n",kv.type); exit(1);
    }
    return kv;
}

int gguf_load(FILE *f, gguf_meta *m){
    if(fread(m->magic,4,1,f)!=1) return -1;
    if(memcmp(m->magic,"GGUF",4)!=0 && memcmp(m->magic,"GGFU",4)!=0) return -2; // accept LE (GGUF) and BE-marked (GGFU) files
    m->version   = rd_u32(f);
    m->n_tensors = rd_u64(f);
    m->n_kv      = rd_u64(f);
    m->kvs = (gguf_kv*)malloc((size_t)m->n_kv*sizeof(gguf_kv));
    for(uint64_t i=0;i<m->n_kv;i++) m->kvs[i]=read_kv(f);
    m->tensors = (gguf_tensor*)malloc((size_t)m->n_tensors*sizeof(gguf_tensor));
    for(uint64_t i=0;i<m->n_tensors;i++){
        gguf_tensor *t=&m->tensors[i];
        t->name=rd_str(f);
        uint32_t nd=rd_u32(f);
        t->n_dims=nd;
        t->dims=(uint64_t*)malloc((size_t)nd*8);
        for(uint32_t d=0;d<nd;d++) t->dims[d]=rd_u64(f);
        t->type=rd_u32(f);
        t->offset=rd_u64(f);
        /* GGUF v3 tensor info has no per-tensor key-value store */
    }
    /* data section start: end of tensor table, padded to alignment (default 32) */
    uint64_t align = 32;
    gguf_kv *ak = gguf_get(m,"general.alignment");
    if(ak && ak->type==4) align = ak->u;
    long pos = ftell(f);
    m->data_start = ((uint64_t)pos + align - 1) / align * align;
    return 0;
}

void gguf_free(gguf_meta *m){
    for(uint64_t i=0;i<m->n_kv;i++){ free(m->kvs[i].key); if(m->kvs[i].s) free(m->kvs[i].s); }
    free(m->kvs);
    for(uint64_t i=0;i<m->n_tensors;i++){ free(m->tensors[i].name); free(m->tensors[i].dims); }
    free(m->tensors);
}

gguf_kv *gguf_get(gguf_meta *m, const char *key){
    for(uint64_t i=0;i<m->n_kv;i++)
        if(m->kvs[i].key && strcmp(m->kvs[i].key,key)==0) return &m->kvs[i];
    return NULL;
}
gguf_tensor *gguf_find(gguf_meta *m, const char *name){
    for(uint64_t i=0;i<m->n_tensors;i++)
        if(strcmp(m->tensors[i].name,name)==0) return &m->tensors[i];
    return NULL;
}

int gguf_read_float(FILE *f, gguf_meta *m, gguf_tensor *t, float *out){
    uint64_t n=1; for(uint32_t d=0;d<t->n_dims;d++) n*=t->dims[d];
    if(fseek(f,(long)(m->data_start + t->offset),SEEK_SET)!=0) return -1; /* offset is relative to data section */
    if(t->type==GGML_F32){
        if(fread(out,4,(size_t)n,f)!=(size_t)n) return -2;
    } else if(t->type==GGML_F16){
        unsigned char *buf=(unsigned char*)malloc((size_t)n*2);
        if(!buf) return -3;
        if(fread(buf,2,(size_t)n,f)!=(size_t)n){ free(buf); return -4; }
        for(uint64_t i=0;i<n;i++){
            unsigned int h=buf[2*i]|((unsigned int)buf[2*i+1]<<8);
            int sign=(h&0x8000)?-1:1, exp=(h>>10)&0x1F, man=h&0x3FF;
            float v;
            if(exp==0) v=0.0f;
            else if(exp==31) v=(man==0)?((sign>0)?1e30f:-1e30f):0.0f;
            else { union{ int bi; float bf; } u;
                   u.bi=(sign<0?0x80000000:0)|((exp-15+127)<<23)|(man<<13); v=u.bf; }
            out[i]=v;
        }
        free(buf);
    } else return -5; /* quantized types: dequantize in milestone 2 */
    return 0;
}

const char *ggml_type_name(int t){
    switch(t){
        case GGML_F32: return "F32";  case GGML_F16: return "F16";
        case GGML_Q4_0: return "Q4_0"; case GGML_Q4_1: return "Q4_1";
        case GGML_Q5_0: return "Q5_0"; case GGML_Q5_1: return "Q5_1";
        case GGML_Q8_0: return "Q8_0"; case GGML_Q2_K: return "Q2_K";
        case GGML_Q3_K: return "Q3_K"; case GGML_Q4_K: return "Q4_K";
        case GGML_Q5_K: return "Q5_K"; case GGML_Q6_K: return "Q6_K";
        case GGML_Q8_K: return "Q8_K";
        default: return "?";
    }
}