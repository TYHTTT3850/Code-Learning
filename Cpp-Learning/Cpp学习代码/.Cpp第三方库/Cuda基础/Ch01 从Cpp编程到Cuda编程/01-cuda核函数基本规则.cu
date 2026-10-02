#include <stdio.h>
#include <cuda_runtime.h>

/*
1.核函数只能访问GPU显存
2.核函数不能使用变长参数
3.核函数不能使用静态变量
4.核函数不能使用函数指针
5.核函数具有异步性
*/

__global__ void helloFromGPU()//核函数在GPU上并行执行，限定词__global__，返回值必须是void
{
    printf("Hello World from GPU! I am thread: %d\n", threadIdx.x);//核函数不支持 iostream
}

int main()
{
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 5>>>();

    cudaDeviceSynchronize();//同步cpu和gpu

    return 0;
}