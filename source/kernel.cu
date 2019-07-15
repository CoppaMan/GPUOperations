#include "CudaCalls.cuh"

__global__ void axpy(float res[ps], float v1[ps], float v2[ps], float s)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < ps) res[i] = v1[i] + s*v2[i];
}

__global__ void dot(float *v, size_t n_elements) // v * v in place
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n_elements) v[i] *= v[i];
}

__inline__ __device__ float warpReduction(float val)
{
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__inline__ __device__ float blockReduction(float val)
{
  static __shared__ float shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduction(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduction(val); //Final reduce within first warp

  return val;
}

__global__ void reduction(float *in, float* out, int N)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= N) return;
  float res = blockReduction(in[i]);
  
  // atomic blockwise reduction
  if (threadIdx.x == 0) atomicAdd(out, res);
}

namespace GPU {
  void AXPY(float *res, float *v1, float *v2, float scalar, size_t elements) {
    cudaStream_t streams[st];
    for (int i = 0; i < st; i++) cudaStreamCreate(&streams[i]);

    float *d_v1, *d_v2, *d_res;
    size_t partitions = (ps-1)+elements / ps;
    size_t 

    cudaProfilerStart();

    cudaMalloc((void**) &d_v1, st*ps*sizeof(float));
    cudaMalloc((void**) &d_v2, st*ps*sizeof(float));
    cudaMalloc((void**) &d_res, st*ps*sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();
    for (int partition = 0; partition < partitions; partition++) {
      size_t occupancy = (partition == partitions-1) ? elements - (partition*ps) : ps; // Elements in this partition
      size_t stream = partition % st;
      size_t offset = stream * ps; // For selecting device range for this stream
      size_t offset2 = partition * ps; // For selecting host region belonging to this partition
      std::cout << "Process partition " << partition << " on stream " << stream << " with off1 " << offset << " and off2 " << offset2 << std::endl;
      cudaMemcpyAsync(&d_v1[offset], &v1[offset2], occupancy*sizeof(float), cudaMemcpyHostToDevice, streams[stream]);
      cudaMemcpyAsync(&d_v2[offset], &v2[offset2], occupancy*sizeof(float), cudaMemcpyHostToDevice, streams[stream]);
      axpy<<<(occupancy+255)/256, 256, 0, streams[stream]>>>(&d_res[offset], &d_v1[offset], &d_v2[offset], scalar);
      cudaMemcpyAsync(&res[offset2], &d_res[offset], occupancy*sizeof(float), cudaMemcpyDeviceToHost, streams[stream]);
    }

    cudaDeviceSynchronize();
    cudaProfilerStop();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    std::cout << "GPU Elapsed: " << elapsed.count() << std::endl;
  }

  void Dot(float *res, float *v, size_t n_elements) {
    float *d_v, *d_res;
    res[0] = 0.0f;
    cudaMalloc((void**) &d_v, n_elements*sizeof(float));
    cudaMalloc((void**) &d_res, sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_v, v, n_elements*sizeof(float), cudaMemcpyHostToDevice);
    dot<<<(n_elements+255)/256, 256>>>(d_v, n_elements);
    reduction<<<(n_elements+255)/256, 256>>>(d_v, d_res, n_elements);
    cudaMemcpy(res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaProfilerStop();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "GPU::Dot Elapsed: " << elapsed.count() << std::endl;
  }

  void Dot_Streams(float *res, float *v, size_t n_elements) {
    cudaStream_t streams[st];
    for (int i = 0; i < st; i++) cudaStreamCreate(&streams[i]);

    float *d_v, *d_res, *sums;
    int partitions = ((ps)+n_elements-1) / (ps);
    cudaMalloc((void**) &d_v, st*(ps)*sizeof(float));
    cudaMalloc((void**) &d_res, st*sizeof(float));

    cudaMallocHost((void **) &sums, st * sizeof(float));

    std::cout << partitions << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    for (int partition = 0; partition < partitions; partition++) {
      int occupancy = (partition == partitions-1) ? n_elements - ((ps)*partition) : ps; // Elements in this partition
      int stream = partition % st;
      int offset = stream * (ps); // For selecting device range for this stream
      int offset2 = partition * (ps); // For selecting host region belonging to this partition
      std::cout << "Process partition " << partition+1 << " with (" << occupancy << "/" << ps << ") elements on stream " << stream+1 << " with off1 " << offset << " and off2 " << offset2 << std::endl;
      cudaMemcpyAsync(&d_v[offset], &v[offset2], occupancy*sizeof(float), cudaMemcpyHostToDevice, streams[stream]);
      dot<<<(occupancy+255)/256, 256, 0, streams[stream]>>>(&d_v[offset], occupancy);
      reduction<<<(occupancy+255)/256, 256, 0, streams[stream]>>>(&d_v[offset], &d_res[stream], occupancy);
    }
    //reduction<<<1,st>>>(d_sum, d_res, st);
    //cudaMemcpy(res, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sums, d_res, st*sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t t = 0; t < st; t++) {
      std::cout << sums[t] << std::endl;
    }

    cudaProfilerStop();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "GPU::Dot Stream Elapsed: " << elapsed.count() << std::endl;
  }
}


