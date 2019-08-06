all:
	nvcc -arch=sm_61 source/main.cpp source/kernel.cu -lcudart -Xcompiler -fopenmp -o gpu -I include