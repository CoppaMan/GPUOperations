all:
	nvcc -arch=sm_61 source/main.cpp source/kernel.cu -lcudart -o gpu -I include