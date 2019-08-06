#include <iostream>
#include "CudaCalls.cuh"

struct Element {
    Real a;
    Real b;
    Real c;
    Real d;
    Real e;
};

void AXPY (float * res, float * v1, float * v2, float scalar, size_t n_elements) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_elements; i++) {
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
    Element *elements;
    Real res = 42.0;;
    int bs = 4096;
    int nb = 1000;
    int N = nb*bs;

    cudaMallocHost((void **) &elements, N * sizeof(Element));
    for (int i = 0; i < 5; i++) {
        elements[i].a = 1.0;
        elements[i].b = 2.0;
        elems[i].c = 3.0;
        elements[i].d = 4.0;
        elems[i].e = 5.0;
    }

    GPU::Dot_Streams(&res, elements, sizeof(Element), N);

    for (int i = 0; i < 5; i++) {
        std::cout << "[" << i << "] = " << xs[i] << std::endl;
    }
}