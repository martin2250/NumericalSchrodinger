CXX := nvcc

.PHONY: all

all: cuda2d

cuda2d: cuda2d.cu harmonicosc.h
	nvcc -o $@ $< --gpu-architecture=compute_61 --use_fast_math -ICImg -lX11 -lpthread -std=c++11 -I/usr/include/boost -L/usr/lib/ -lboost_math_c99
