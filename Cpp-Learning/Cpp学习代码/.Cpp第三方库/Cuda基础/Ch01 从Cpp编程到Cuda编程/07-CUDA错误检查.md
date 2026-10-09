CUDA 运行时 API 用 `cudaError_t` 枚举类型表示执行状态：`cudaSuccess` 表示成功，其他值表示错误或特定状态。常用函数如 `cudaMalloc`、`cudaMemcpy` 都通过返回值报告状态。

常见错误码：

| 数字错误码 | 枚举名称 | 含义 |
|---:|---|---|
| 0 | `cudaSuccess` | 成功 |
| 1 | `cudaErrorInvalidValue` | 参数值非法 |
| 2 | `cudaErrorMemoryAllocation` | 内存或资源分配失败 |
| 3 | `cudaErrorInitializationError` | 初始化失败 |
| 9 | `cudaErrorInvalidConfiguration` | 核函数启动配置超过设备限制 |
| 600 | `cudaErrorNotReady` | 异步操作仍在执行 |
| 700 | `cudaErrorIllegalAddress` | 非法内存访问 |
| 701 | `cudaErrorLaunchOutOfResources` | 核函数启动资源不足 |
| 702 | `cudaErrorLaunchTimeout` | 核函数执行超时 |

完整定义见 [CUDA 官方错误码文档](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html)。

CUDA 的错误检查通常涉及这几个函数：

| 函数 | 作用 |
|---|---|
| `cudaGetLastError()` | 获取当前线程的最后错误，并将错误记录重置为 `cudaSuccess` |
| `cudaPeekAtLastError()` | 获取当前线程的最后错误，保留错误记录 |
| `cudaGetErrorName(err)` | 将错误码转换为枚举名称 |
| `cudaGetErrorString(err)` | 将错误码转换为错误说明 |

典型的错误检查函数：

```cpp
// filename一般填宏：__FILE__，lineNumber一般填宏：__LINE__
cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
{
    if (error_code != cudaSuccess) {
        printf("CUDA error: code=%d, name=%s, description=%s, file=%s, line=%d\n", static_cast<int>(error_code), cudaGetErrorName(error_code), cudaGetErrorString(error_code), filename, lineNumber);
    }

    return error_code;
}
```

