#include <iostream>
#include "CudaCalls.cuh"

struct Element {
    Real a;
    Real b;
    Real c;
    Real d;
    Real e;
};

void AXPY (Real * res, Real * v1, Real * v2, Real scalar, size_t n_elements) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_elements; i++) {
        res[5*i] = v1[5*i] + scalar*v2[5*i];
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    std::cout << "AXPY:CPU Elapsed: " << elapsed.count() << std::endl;
}

void Dot (Real *res, Real* v1, size_t n_elements) {
    Real sum = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    //#pragma omp parallel for reduction(+: sum)
    for(int i=0; i<n_elements; i++) {
        sum += v1[5*i]*v1[5*i];
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "CPU::Dot Elapsed: " << elapsed.count() << std::endl;
    *res = sum;
}

void Dot2 (Real *res, Real* v1, Real* v2, size_t n_elements) {
    Real sum = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    //#pragma omp parallel for reduction(+: sum)
    for(int i=0; i<n_elements; i++) {
        sum += v1[5*i]*v2[5*i];
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "CPU::Dot Elapsed: " << elapsed.count() << std::endl;
    *res = sum;
}

int main (int argc, char *argv[]) {
    Element *elements;
    Real res1 = 42.0;
    Real res2 = 42.0;
    Real res1s = 42.0;
    Real res2s = 42.0;
    int bs = 4096;
    int nb = 1000;
    int N = nb*bs;

    cudaMallocManaged((void **) &elements, N * sizeof(Element));
    //cudaMallocHost((void **) &elements, N * sizeof(Element));
    //elements = (Element *) malloc(N * sizeof(Element)); Slower
    for (int i = 0; i < N; i++) {
        elements[i].a = 0.01;
        elements[i].b = 0.05;
        elements[i].c = 0.03;
        elements[i].d = 0.04;
        elements[i].e = 0.05;
    }

    
    GPU::Dot_Streams(&res2, &elements->a, sizeof(Element), N);
    std::cout << "GPU Single Res: " << res2s << std::endl;

    Dot(&res1, &elements->a, N);
    std::cout << "CPU Single Res: " << res1s << std::endl;

    GPU::Dot_Streams2(&res2, &elements->a, &elements->b, sizeof(Element), N);
    std::cout << "GPU Res: " << res2 << std::endl;
    
    Dot2(&res1, &elements->a, &elements->b, N);
    std::cout << "CPU Res: " << res1 << std::endl;
    
    /*
    for (int i = 0; i < 100; i++) {
        std::cout << "[" << i << "] = " << elements[i].c << " ";
    }
    */
}