CC      ?= cc
NVCC    ?= nvcc
CFLAGS  ?= -O2 -Wall -Wextra
NVFLAGS ?= -arch=sm_52 -O2

BIN     := mini-llama
GPU_BIN := mini-llama-gpu
M2_BIN  := mini-llama-m2
SRC     := src/main.c
GPU_SRC := src/gpu_mm.cu src/gguf.c
M2_SRC  := src/m2.cu src/gguf.c src/quant.c

all: $(BIN) $(GPU_BIN) $(M2_BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

$(GPU_BIN): $(GPU_SRC)
	$(NVCC) $(NVFLAGS) -o $@ src/gpu_mm.cu src/gguf.c -lcublas

$(M2_BIN): $(M2_SRC)
	$(NVCC) $(NVFLAGS) -o $@ src/m2.cu src/gguf.c src/quant.c -lcublas

clean:
	rm -f $(BIN) $(GPU_BIN) $(M2_BIN)

.PHONY: all clean
