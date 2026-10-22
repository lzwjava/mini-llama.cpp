CC      ?= cc
NVCC    ?= nvcc
CFLAGS  ?= -O2 -Wall -Wextra
NVFLAGS ?= -arch=sm_52 -O2

BIN     := mini-llama
GPU_BIN := mini-llama-gpu
SRC     := src/main.c
GPU_SRC := src/gpu_mm.cu src/gguf.c

all: $(BIN) $(GPU_BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

$(GPU_BIN): $(GPU_SRC)
	$(NVCC) $(NVFLAGS) -o $@ src/gpu_mm.cu src/gguf.c -lcublas

clean:
	rm -f $(BIN) $(GPU_BIN)

.PHONY: all clean
