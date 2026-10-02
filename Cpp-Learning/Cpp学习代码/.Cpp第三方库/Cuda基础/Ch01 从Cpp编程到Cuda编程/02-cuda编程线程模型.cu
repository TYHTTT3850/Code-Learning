#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU_1D()
{
    printf("Hello World from GPU! I am thread %d of block %d\n", threadIdx.x,blockIdx.x);
}

__global__ void helloFromGPU_3D()
{
    // 当前线程在 Block 内的三维坐标
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // 当前 Block 在 Grid 内的三维坐标
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    // 当前线程在整个 Grid 中的全局三维坐标
    // 三维坐标得到后，可以计算全局id：global_id = z * width * height + y * width + x;
    // width = gridDim.x * blockDim.x , 代表 x 方向上有几个 thread
    // height = gridDim.y * blockDim.y , 代表 y 方向上有几个 thread
    // depth = gridDim.z * blockDim.z , 代表 z 方向上有几个 thread
    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;
    int z = bz * blockDim.z + tz;

    printf(
        "Block(%d, %d, %d), "
        "Thread(%d, %d, %d), "
        "Global(%d, %d, %d)\n",
        bx, by, bz,
        tx, ty, tz,
        x, y, z
    );
}

int main()
{
    /*
    CUDA 一维线程模型：
    1. Thread：最基本的执行单元，通过 threadIdx 获取线程在线程块中的编号。
    2. Block：多个线程组成一个线程块，通过 blockIdx 获取线程块编号。
    3. Grid：多个线程块组成一个网格。

    核函数调用格式：
        kernel_Function<<<gridDim, blockDim>>>();

    gridDim  ：Grid 中 Block 的数量
    blockDim ：每个 Block 中 Thread 的数量

    下面启动 2 个 Block，每个 Block 有 4 个 Thread，
    因此总共启动 2 × 4 = 8 个线程。
    */
    helloFromGPU_1D<<<2, 4>>>();

    cudaDeviceSynchronize();

    /*
    CUDA 三维线程模型：

    Grid:
        x 方向有 3 个 Block
        y 方向有 2 个 Block
        z 方向有 2 个 Block

        共 3 * 2 * 2 = 12 个 Block

    每个 Block:
        x 方向有 2 个 Thread
        y 方向有 2 个 Thread
        z 方向有 3 个 Thread

        每个 Block 共 2 * 2 * 3 = 12 个 Thread

    总线程数：
        12 * 12 = 144

    注意事项：
    Grid 每个维度最大：
        x : 2^31 - 1
        y : 2^16-1
        z : 2^16-1

    Block 每个维度最大：
        x : 1024
        y : 1024
        z : 64

    每个 Block 的线程总数最大为 1024，也就是说一个线程块中，blockDim.x*blockDim.y*blockDim.z不超过1024.
    */

    dim3 grid_size(3, 2, 2);//创建 dim3 变量
    dim3 block_size(2, 2, 3);

    helloFromGPU_3D<<<grid_size, block_size>>>();

    cudaDeviceSynchronize();

    return 0;
}