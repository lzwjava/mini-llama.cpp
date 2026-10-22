# mini-llama.cpp

Write-your-own minimal llama.cpp for **192.168.1.137**:
CPU i5-6500 (4 cores, AVX2) · 8 GB DDR4 · **Tesla M40 24 GB** (Maxwell, CC 5.2, fp32)
· 550.163.01 driver (CUDA 12.4) · models on /mnt/models (500 GB disk).

## Milestone 1 — DONE ✅  ("GPU pipeline alive")
- `src/gguf.{h,c}` — minimal GGUF v3 reader (header, KV store, tensor table).
  Learned the hard way: kv value type is **uint32**; no per-tensor KV block;
  tensor offsets are **relative to the data section** (aligned to 32).
- `src/gpu_mm.cu` — parse a real GGUF, load a real F32 tensor onto the M40,
  run a custom `sm_52` kernel (verified exact vs CPU), then `cuBLAS` GEMM
  vs naive CPU reference (921 GFLOPS, rel err 2.4e-7).
- Verified against the reference `gguf` python package (byte-for-byte values).

## Build
make            # mini-llama (CPU stub)
make mini-llama-gpu   # needs CUDA runtime + libcublas (present)
./mini-llama-gpu /mnt/models/qwen2.5-0.5b-instruct-q4_k_m.gguf

## Roadmap
- [x] M1: GGUF parse + GPU pipeline proof (cuBLAS + custom kernels)
- [ ] M2: q4_K/Q5_0/Q8_0 block dequant -> fp32, upload weights to VRAM
- [ ] M3: forward pass (RMSNorm, RoPE, attention+KV cache, SiLU MLP) on GPU
- [ ] M4: sampling loop + CLI (`-p "..." -n 32`)
- [ ] M5: speed (AVX2 CPU path as fallback, compare vs GPU)

## Milestone 2 — DONE ✅ (dequant -> fp32 -> weights living in VRAM)
- `src/quant.{h,c}` — block dequant ported VERBATIM from llama.cpp ggml-quants.c:
  Q5_0, Q8_0, Q4_K (K_SCALE_SIZE=12), Q6_K, F16 + F32 passthrough.
- `src/m2.cu` — census of the model types (Q5_0=133, F32=121, Q8_0=13,
  Q6_K=12, Q4_K=12; 630M params -> 2.52 GB fp32), dequant stats per tensor,
  full VRAM upload (573 MB used), and a REAL GEMM: ffn_up (4864x896) x hidden,
  verified vs CPU (err/scale 5.6e-6, PASS).
- Verification: Q5_0/Q8_0 match the reference `gguf` python package EXACTLY
  (element-wise). Q6_K/Q4_K are implemented as llama.cpp does (the writer of
  this file); note gguf-py has its own Q4_K/Q6_K scale-pairing quirk that
  produces a *different* reconstruction than llama.cpp for the same bytes.

check: ./mini-llama-m2 /mnt/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
