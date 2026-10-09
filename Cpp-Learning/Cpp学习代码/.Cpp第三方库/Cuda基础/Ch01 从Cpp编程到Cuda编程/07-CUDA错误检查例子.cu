#include <stdio.h>
#include <cuda_runtime.h>

// 错误检查函数
cudaError_t ErrorCheck(cudaError_t error_code,
                      const char* filename,
                      int lineNumber)
{
    if (error_code != cudaSuccess) {
        printf("CUDA error: code=%d, name=%s, description=%s, file=%s, line=%d\n",
               static_cast<int>(error_code),
               cudaGetErrorName(error_code),
               cudaGetErrorString(error_code),
               filename,
               lineNumber);
    }

    return error_code;
}

// 核函数：故意访问空指针
__global__ void kernel(int* p)
{
    *p = 123;
}

int main()
{
    // 示例一：检查普通运行时 API 的错误
    printf("示例一：cudaMalloc 参数错误\n");

    cudaError_t error = ErrorCheck(cudaMalloc(nullptr, 1024), __FILE__, __LINE__);

    cudaGetLastError();//将错误记录重置为 cudaSuccess

    // 示例二：检查核函数的启动和执行错误
    printf("\n示例二：核函数访问空指针\n");

    kernel<<<1, 1>>>(nullptr);//执行核函数
    
    // 检查核函数启动时的启动错误，并清除最后错误记录
    error = ErrorCheck(cudaGetLastError(), __FILE__, __LINE__);//核函数返回值必须为空，所以需要使用 cudaGetLastError() 来获取错误

    // 本例中核函数启动时没有错误
    if (error != cudaSuccess) {
        return 1;
    }

    // 等待 GPU 工作结束(工作结束包括执行成功和执行失败两种情况)，检查核函数执行中的执行错误
    error = ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);

    // 本例中核函数执行时出现错误
    if (error != cudaSuccess) {
        return 1;
    }

    return 0;
}