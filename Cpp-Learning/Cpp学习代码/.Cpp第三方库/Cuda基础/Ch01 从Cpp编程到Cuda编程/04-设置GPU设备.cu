#include <cstdio>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    //检测计算机GPU数量
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0){
        printf("No CUDA compatable GPU found\n");
        exit(-1);
    }
    else {
        printf("The count of GPUs is %d.\n", deviceCount);
    }

    //设置GPU
    int deviceIndex = 0;
    error = cudaSetDevice(deviceIndex);
    if (error != cudaSuccess){
        printf("Failed to set GPU0 for computing\n");
        exit(-1);
    }
    else {
        printf("set GPU0 for computing.");
    }

    return 0;
}