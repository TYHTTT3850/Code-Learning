#include <stdio.h>
#include <cuda_runtime.h>

// 这是一个 Kernel（核函数），它在 GPU 上运行
__global__ void helloFromGPU() {
    // 打印当前执行任务的线程 ID
    printf("Hello World from GPU! I am thread: %d\n", threadIdx.x);
}

int main() {
    printf("Hello World from CPU!\n\n");

    // <<<1, 5>>> 意思是启动 1 个 Block，里面包含 5 个 Thread 并发执行
    helloFromGPU<<<1, 5>>>();

    // 这一步必须有！
    // 因为 GPU 执行是异步的，CPU 触发指令后会立刻往下走。
    // 如果不强行让 CPU 等待，程序会直接结束，你什么输出都看不到。
    cudaDeviceSynchronize();

    return 0;
}