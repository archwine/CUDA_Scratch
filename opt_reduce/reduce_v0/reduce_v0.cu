#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define THREAD_PER_BLOCK 256

__global__ void reduce0(float *input, float *output) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    // eche thread loads data from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];

    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem [output]
    if (tid == 0) output[blockIdx.x] = sdata[0]; 
}


bool check(float* output, float* result, int n){
    for (int i=0; i < n ; i ++) {
        if (output[i]!=result[i])
           return false; 
    }
    return true;
}

int main() {
    const int N = 32 * 1024 * 1024;
    float *input = (float *) malloc(N * sizeof(float));

    float *d_input;
    cudaMalloc((void **)&d_input, N * sizeof(float));

    int block_num = N / THREAD_PER_BLOCK;
    float* output = (float *)malloc((N / THREAD_PER_BLOCK) * sizeof(float));

    float* d_output;
    cudaMalloc((void **)&d_output, (N / THREAD_PER_BLOCK) * sizeof(float));
    float *result = (float*)malloc((N / THREAD_PER_BLOCK) * sizeof(float));

    // init the input array with random numbers
    for (int i = 0; i < N; i++) {
        input[i] = 2 * (float)drand48() - 1.0;
    }

    // cpu reduce function
    for (int i = 0; i < block_num; i ++) {
        float cur = 0;
        for (int j = 0; j < THREAD_PER_BLOCK; j ++){
            cur += input[i * THREAD_PER_BLOCK + j];
        }
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(N / THREAD_PER_BLOCK, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);

    reduce0<<<Grid, Block>>>(d_input, d_output);

    cudaMemcpy(output, d_output, (N / THREAD_PER_BLOCK) * sizeof(float), cudaMemcpyDeviceToHost);

    if(check(output, result, N / THREAD_PER_BLOCK))printf("the ans is right\n");
    else {
        printf("the ans is wrong\n");
        for (int i = 0; i < N / THREAD_PER_BLOCK; i ++){
            printf("%f\n", output[i]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(input);

    printf("reduce_v0 \n");
    return 0;
}