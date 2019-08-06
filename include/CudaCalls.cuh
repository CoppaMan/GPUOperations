#pragma once
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>

#define Cubism_bs 4096 // 16 cubed elements per block
#define BpP 20 // Blocks per partition

#define ps (Cubism_bs*BpP) // Needs to be large enough to be buisier than the cpu
#define st 3 // 2 is optimal for this problem

#define Real double

namespace GPU {
	void AXPY(Real *res, Real *v1, Real *v2, Real scalar, size_t stride, size_t n_elements);
	void Dot(float *res, float *v, size_t n_elements);
	void Dot_Streams(Real *res, Real *v, size_t stride, size_t n_elements);
	void Dot_Streams2(Real *res, Real *v1, Real *v2, size_t stride, size_t n_elements);
}