#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void sharedMemoryKernel(float*  d_out, 
                                    const float* d_in, 
                                    int size, 
                                    int sharedMemSize) {
    extern __shared__ float sharedMem[];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        sharedMem[threadIdx.x] = d_in[tid];
        __syncthreads();

        float temp = sharedMem[threadIdx.x] * 2.0f;
        d_out[tid] = temp;
    }
}

int main() {
    int device;
    cudaGetDevice(&device);

    int sharedMemPerBlock, l1CacheSize;
    cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device);
    cudaDeviceGetAttribute(&l1CacheSize, cudaDevAttrLocalL1CacheSupported, device);

    std::cout << "Shared Memory per Block: " <<  sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "L1 Cache Supported: " << (l1CacheSize ? "Yes" : "No") << std::endl;

    const int size = 1024 * 1024;
    float *h_in = new float[size];
    float *h_out= new float[size];
    for (int i = 0; i < size; i  ++) {
        h_in[i] = static_cast<float>(i);
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);

    for (int sharedMemSize = 0;  sharedMemSize <= sharedMemPerBlock; sharedMemSize += 1024) {
        auto start = std::chrono::high_resolution_clock::now();

        sharedMemoryKernel<<<gridSize, blockSize, sharedMemSize>>>(d_out, d_in, size, sharedMemSize);

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        std::cout << "Shared Memory Size:  " << sharedMemSize << 
                    " bytes, Time: " << duration.count() << " ms\n";
    }

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}        
