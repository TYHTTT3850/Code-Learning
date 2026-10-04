| 操作         | C                           | C++                            | CUDA                                  |
| ------------ | --------------------------- | ------------------------------ | ------------------------------------- |
| 分配数组     | `malloc(n * sizeof(float))` | `new float[n]`                 | `cudaMalloc(&d_p, n * sizeof(float))` |
| 释放数组     | `free(p)`                   | `delete[] p`                   | `cudaFree(d_p)`                       |
| 复制数据     | `memcpy(dst, src, bytes)`   | `std::memcpy(dst, src, bytes)` | `cudaMemcpy(dst, src, bytes, 方向)`   |
| 按字节初始化 | `memset(p, 0, bytes)`       | `std::memset(p, 0, bytes)`     | `cudaMemset(d_p, 0, bytes)`           |

示例：

```cpp
// C：返回分配的地址
float* p = (float*)malloc(n * sizeof(float));
free(p);

// C++：返回分配的地址
float* p = new float[n];
delete[] p;

// CUDA：通过 &d_p 写入地址，返回错误码
float* d_p = nullptr;
cudaMalloc(&d_p, n * sizeof(float));
cudaFree(d_p);
```

