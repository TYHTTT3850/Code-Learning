#include <cuda_runtime_api.h>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;//计算线程的全局索引

    if (i < n)//全局索引小于总元素个数的线程执行对应索引数据的加法
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    
    const int N = 10;//总共 10 个元素
    const size_t size = N * sizeof(float);//10个浮点数需要的内存空间
    
    // CPU 内存：malloc
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    
    /*
    初始化 CPU 数据
    h_A=[0,1,2,3,4,5,6,7,8,9]
    h_B=[0,1,4,6,8,10,12,14,16,18]
    */
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }
    
    // GPU 内存：cudaMalloc
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // 数据从 CPU 复制到 GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // 核函数启动配置
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    //创建开始和结束事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //记录开始事件和结束事件的时刻
    cudaEventRecord(start);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    
    //等待结束事件完成
    cudaEventSynchronize(stop);
    
    //同步
    cudaDeviceSynchronize();
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);//计算开始事件和结束事件的时刻的差
    printf("耗时：%f ms\n", ms);
    
    //销毁两个事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 数据从 GPU 复制到 CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 输出结果
    for (int i = 0; i < N; ++i)
    {
        std::cout
            << h_A[i] << " + "
            << h_B[i] << " = "
            << h_C[i]
            << '\n';
    }

    // GPU 内存释放：cudaFree
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // CPU 内存释放：free
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}