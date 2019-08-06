#include "CudaCalls.cuh"

// Computes v1 + s*v2 and stores it in v1
__global__ void axpy(Real v1[ps], Real v2[ps], Real s)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < ps) v1[i] += s*v2[i];
}

__global__ void dot(Real *v, size_t n_elements) // v * v in place
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n_elements) v[i] *= v[i];
}

__global__ void dot2(Real *u, Real *v, size_t n_elements) // u*v in u
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n_elements) u[i] *= v[i];
}

__global__ void summation(Real *d_sum, int sums){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < sums-1) atomicAdd(&d_sum[0], d_sum[i+1]);
}

__inline__ __device__ Real warpReduction(Real val)
{
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down_sync(0xffffffff,val, offset);
  return val;
}

__inline__ __device__ Real blockReduction(Real val)
{
  static __shared__ Real shared[32]; // Shared mem for 32 partial sums
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

__global__ void reduction(Real *in, Real* out, int N)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= N) return;
  Real res = blockReduction(in[i]);
  
  // atomic blockwise reduction
  if (threadIdx.x == 0) atomicAdd(out, res);
}

namespace GPU {
  void AXPY(Real *res, Real *v1, Real *v2, Real scalar, size_t stride, size_t n_elements) {
    cudaStream_t streams[st];
    for (int i = 0; i < st; i++) cudaStreamCreate(&streams[i]);

    Real *d_v1, *d_v2, *d_sums;
    int partitions = ((ps)+n_elements-1) / (ps);
    cudaMalloc((void**) &d_v1, st*(ps)*sizeof(Real));
    cudaMalloc((void**) &d_v2, st*(ps)*sizeof(Real));
    cudaMalloc((void**) &d_sums, st*sizeof(Real));

    auto start = std::chrono::high_resolution_clock::now();
    
    for (int partition = 0; partition < partitions; partition++) {
      int occupancy = (partition == partitions-1) ? n_elements - (ps*partition) : ps; // Elements in this partition
      int stream = partition % st;
      int offset = stream * ps; // For selecting device range for this stream
      int structElements = stride/sizeof(Real);
      int offsetWstrid = partition * ps * structElements;
      int offset2 = partition * ps; // For selecting host region belonging to this partition
      std::cout << "Process partition " << partition+1 << " with (" << occupancy << "/" << ps << ") elements on stream " <<
      stream+1 << " with off1 " << offset << " and off2 " << offset2 << " with offWS " << offsetWstrid << std::endl;
      if (stride > 0) {
        cudaMemcpy2DAsync(&d_v1[offset], sizeof(Real), v1+offsetWstrid, stride, sizeof(Real), occupancy, cudaMemcpyHostToDevice, streams[stream]);
        cudaMemcpy2DAsync(&d_v2[offset], sizeof(Real), v2+offsetWstrid, stride, sizeof(Real), occupancy, cudaMemcpyHostToDevice, streams[stream]);
      } else if (stride == 0) {
        cudaMemcpyAsync(&d_v1[offset], &v1[offset2], occupancy*sizeof(Real), cudaMemcpyHostToDevice, streams[stream]);
        cudaMemcpyAsync(&d_v2[offset], &v2[offset2], occupancy*sizeof(Real), cudaMemcpyHostToDevice, streams[stream]);
      } else {
        // Whaa!
      }
      axpy<<<(occupancy+255)/256, 256, 0, streams[stream]>>>(&d_v1[offset], &d_v2[offset], scalar);
      if (stride > 0) {
        cudaMemcpy2DAsync(v1+offsetWstrid, stride, &d_v1[offset], sizeof(Real), sizeof(Real), occupancy, cudaMemcpyDeviceToHost, streams[stream]);
      } else if (stride == 0) {
        cudaMemcpyAsync(&res[offset2], &d_v1[offset], occupancy*sizeof(Real), cudaMemcpyDeviceToHost, streams[stream]);
      } else {
        // Whaa again
      }
    }
    summation<<<1,st>>>(d_sums, st);
    cudaMemcpy(res, &d_sums[0], sizeof(Real), cudaMemcpyDeviceToHost);

    cudaProfilerStop();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "GPU::AXPY Stream Elapsed: " << elapsed.count() << std::endl;
  }

  void Dot_Streams(Real *res, Real *v, size_t stride, size_t n_elements) {
    cudaStream_t streams[st];
    for (int i = 0; i < st; i++) cudaStreamCreate(&streams[i]);

    Real *d_v, *d_sums;
    int partitions = ((ps)+n_elements-1) / (ps);
    cudaMalloc((void**) &d_v, st*(ps)*sizeof(Real));
    cudaMalloc((void**) &d_sums, st*sizeof(Real));

    auto start = std::chrono::high_resolution_clock::now();
    
    for (int partition = 0; partition < partitions; partition++) {
      int occupancy = (partition == partitions-1) ? n_elements - (ps*partition) : ps; // Elements in this partition
      int stream = partition % st;
      int offset = stream * ps; // For selecting device range for this stream
      int structElements = stride/sizeof(Real);
      int offsetWstrid = partition * ps * structElements;
      int offset2 = partition * ps; // For selecting host region belonging to this partition
      //std::cout << "Process partition " << partition+1 << " with (" << occupancy << "/" << ps << ") elements on stream " <<
      //stream+1 << " with off1 " << offset << " and off2 " << offset2 << " with offWS " << offsetWstrid << std::endl;
      if (stride > 0) {
        cudaMemcpy2DAsync(&d_v[offset], sizeof(Real), v+offsetWstrid, stride, sizeof(Real), occupancy, cudaMemcpyHostToDevice, streams[stream]);
      } else if (stride == 0) {
        cudaMemcpyAsync(&d_v[offset], &v[offset2], occupancy*sizeof(Real), cudaMemcpyHostToDevice, streams[stream]);
      } else {
        // Whaa!
      }
      dot<<<(occupancy+255)/256, 256, 0, streams[stream]>>>(&d_v[offset], occupancy);
      reduction<<<(occupancy+255)/256, 256, 0, streams[stream]>>>(&d_v[offset], &d_sums[stream], occupancy);
    }
    summation<<<1,st>>>(d_sums, st);
    cudaMemcpy(res, &d_sums[0], sizeof(Real), cudaMemcpyDeviceToHost);

    cudaProfilerStop();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "GPU::Dot Stream Elapsed: " << elapsed.count() << std::endl;
  }

  void Dot_Streams2(Real *res, Real *v1, Real *v2, size_t stride, size_t n_elements) {
    cudaStream_t streams[st];
    for (int i = 0; i < st; i++) cudaStreamCreate(&streams[i]);

    Real *d_v1, *d_v2, *d_sums;
    int partitions = ((ps)+n_elements-1) / (ps);
    cudaMalloc((void**) &d_v1, st*(ps)*sizeof(Real));
    cudaMalloc((void**) &d_v2, st*(ps)*sizeof(Real));
    cudaMalloc((void**) &d_sums, st*sizeof(Real));

    auto start = std::chrono::high_resolution_clock::now();
    
    for (int partition = 0; partition < partitions; partition++) {
      int occupancy = (partition == partitions-1) ? n_elements - (ps*partition) : ps; // Elements in this partition
      int stream = partition % st;
      int offset = stream * ps; // For selecting device range for this stream
      int structElements = stride/sizeof(Real);
      int offsetWstrid = partition * ps * structElements;
      int offset2 = partition * ps; // For selecting host region belonging to this partition
      //std::cout << "Process partition " << partition+1 << " with (" << occupancy << "/" << ps << ") elements on stream " <<
      //stream+1 << " with off1 " << offset << " and off2 " << offset2 << " with offWS " << offsetWstrid << std::endl;
      if (stride > 0) {
        cudaMemcpy2DAsync(&d_v1[offset], sizeof(Real), v1+offsetWstrid, stride, sizeof(Real), occupancy, cudaMemcpyHostToDevice, streams[stream]);
        cudaMemcpy2DAsync(&d_v2[offset], sizeof(Real), v2+offsetWstrid, stride, sizeof(Real), occupancy, cudaMemcpyHostToDevice, streams[stream]);
      } else if (stride == 0) {
        cudaMemcpyAsync(&d_v1[offset], &v1[offset2], occupancy*sizeof(Real), cudaMemcpyHostToDevice, streams[stream]);
        cudaMemcpyAsync(&d_v2[offset], &v2[offset2], occupancy*sizeof(Real), cudaMemcpyHostToDevice, streams[stream]);
      } else {
        // Whaa!
      }
      dot2<<<(occupancy+255)/256, 256, 0, streams[stream]>>>(&d_v1[offset], &d_v2[offset], occupancy);
      reduction<<<(occupancy+255)/256, 256, 0, streams[stream]>>>(&d_v1[offset], &d_sums[stream], occupancy);
    }
    summation<<<1,st>>>(d_sums, st);
    cudaMemcpy(res, &d_sums[0], sizeof(Real), cudaMemcpyDeviceToHost);

    cudaProfilerStop();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "GPU::Dot Stream Elapsed: " << elapsed.count() << std::endl;
  }
}