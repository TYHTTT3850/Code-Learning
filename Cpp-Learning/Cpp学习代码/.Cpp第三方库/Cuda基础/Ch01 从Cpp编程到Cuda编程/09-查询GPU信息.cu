#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int count = 0;

    //获取cuda GPU 数量
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};//GPU属性存储
        err = cudaGetDeviceProperties(&prop, i);//获取对应GPU的属性
        if (err != cudaSuccess) {
            printf("GPU %d: %s\n", i, cudaGetErrorString(err));
            continue;
        }

        printf("GPU 名称 %d: %s\n", i, prop.name);
        printf("显存总量: %.2f GiB\n",prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("计算能力: %d.%d\n", prop.major, prop.minor);
        printf("SM 数量: %d\n", prop.multiProcessorCount);
        printf("每个 block 的最大 thread 数: %d\n", prop.maxThreadsPerBlock);
        printf("Warp 大小: %d\n", prop.warpSize);
        printf("block 各维度上限: (%d, %d, %d)\n", prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
        printf("grid 各维度上限: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("每个 SM 最大驻留 thread: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("每个 SM 最大驻留 block 数: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("每个 block 共享内存上限: %.2f KiB\n", prop.sharedMemPerBlock / 1024.0);
        printf("每个 SM 共享内存容量: %.2f KiB\n",prop.sharedMemPerMultiprocessor / 1024.0);
        printf("每个 block 寄存器数量上限: %d\n", prop.regsPerBlock);
        printf("每个 SM 寄存器数量: %d\n", prop.regsPerMultiprocessor);
        printf("L2 缓存容量: %.2f MiB\n",prop.l2CacheSize / (1024.0 * 1024.0));
        printf("显存位宽: %d bit\n", prop.memoryBusWidth);
        printf("支持 kernel 并发: %s\n", prop.concurrentKernels ? "yes" : "no");
    }

    return 0;
}