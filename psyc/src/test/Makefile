SHELL=/bin/bash
CC=gcc
CFLAGS=-std=gnu99 -g -ggdb
LDFLAGS=-lz -lm
OBJS=../psyc.o ../utils.o ../convolutional.o ../recurrent.o ../lstm.o ../mnist.o test.o

include ../avx.mk
ifeq ($(AVX),on)
	CFLAGS+=-DUSE_AVX -mavx2 -mfma
        OBJS+=../avx.o
endif

default: all

main_tests: $(OBJS) main_tests.o
	$(CC) -o main_tests $(OBJS) main_tests.o $(LDFLAGS)
all: main_tests
	./main_tests
        
