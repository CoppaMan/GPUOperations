#include <iostream>
#include "CudaCalls.cuh"

struct Element {
    float x;
    float y;
    float z;
};

void AXPY (float * res, float * v1, float * v2, float scalar, size_t n_blocks) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ps*n_blocks; i++) {
        res[i] = v1[i] + scalar*v2[i];
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    std::cout << "CPU Elapsed: " << elapsed.count() << std::endl;
}

void Dot (float *res, float* v, size_t n_elements) {
    auto start = std::chrono::high_resolution_clock::now();
    res[0] = 0;
    for (size_t i = 0; i < n_elements; i++) {
        res[0] += v[i]*v[i];
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "CPU::Dot Elapsed: " << elapsed.count() << std::endl;
}

int main (int argc, char *argv[]) {
    // Use an array of pinned memory locations
    size_t n_elements = 100*Cubism_bs;
    float *v_blocks, *v1_blocks, *v2_blocks, *res_blocks;
    float *sum;

    // Similar to cubisms layout
    //cudaMallocHost((void **) &v_blocks, n_elements * sizeof(Element));

    cudaMallocHost((void **) &v1_blocks, n_elements * sizeof(float));
    cudaMallocHost((void **) &v2_blocks, n_elements * sizeof(float));
    cudaMallocHost((void **) &res_blocks, n_elements * sizeof(float));
    cudaMallocHost((void **) &sum, sizeof(float));

    for (int element = 0; element < n_elements; element++) {
        v1_blocks[element] = (float) 1;
    }

    //GPU::AXPY(res_blocks, v1_blocks, v2_blocks, 2.0f, n_blocks);
    //AXPY(res_blocks, v1_blocks, v2_blocks, -2.0f, n_blocks);
    GPU::Dot_Streams(sum, v1_blocks, n_elements);
    std::cout << "Reduction: " << sum[0] << std::endl;

    Dot(sum, v1_blocks, n_elements);
    std::cout << "Reduction: " << sum[0] << std::endl;

    /*
    for (size_t block = 0; block < 1; block++) {
        for (int element = 0; element < 32; element++) {
            std::cout << v1_blocks[element + ps*block] << " ";
        } std::cout << std::endl;
    }
    */
}