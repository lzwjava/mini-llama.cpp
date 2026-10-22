CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra
SRC := $(wildcard src/*.c)
BIN := mini-llama
all: $(BIN)
$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^
clean:
	rm -f $(BIN)
.PHONY: all clean
