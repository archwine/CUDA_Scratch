#include <iostream>
#include <cuda_runtime.h>


int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return 1;
    }

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;

        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
        std::cout << "  Max Threads Per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads Per Multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Register Per Block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Register Per Multiprocessor: " << deviceProp.regsPerMultiprocessor << std::endl;
        std::cout << "  Shared Memory per Block Optin: " << deviceProp.sharedMemPerBlockOptin << " bytes" << std::endl;
        std::cout << "  Shared Memory per Multiprocessor: " << deviceProp.sharedMemPerMultiprocessor << " bytes" << std::endl;

	std::cout << " ====== ====== ====== ====== ====== ====== ====== ====== ======" << std::endl;

	int blockLimitSM;
    	cudaDeviceGetAttribute(&blockLimitSM, cudaDevAttrMaxBlocksPerMultiprocessor, i);
	std::cout << "  Block Limit SM: " << blockLimitSM << std::endl;

	int theoreticalActiveWarpsPerSM = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
	std::cout << "  Theoretical Active Warps per SM: " << theoreticalActiveWarpsPerSM << std::endl;

    }
}
