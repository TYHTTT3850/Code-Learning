# VS Code 配置 CUDA + clangd 开发环境

需要的编程环境：

```
Cuda 环境
MSVC 编译器
Ninja
CMake
```

VS Code 要安装的插件：

```
1. C/C++
   Publisher: Microsoft
   Extension ID: ms-vscode.cpptools

2. CMake Tools
   Publisher: Microsoft
   Extension ID: ms-vscode.cmake-tools

3. clangd
   Publisher: LLVM
   Extension ID: llvm-vs-code-extensions.vscode-clangd
```

## 安装 Ninja 并配置

安装地址：https://github.com/ninja-build/ninja 。Windows 版本通常下载：

```
ninja-win.zip
```

下载好后，比如解压到：`D:\Ninja\` ，其中会有：`D:\Ninja\ninja.exe` 。

把 `D:\Ninja\` 加入到环境变量 `PATH` 中。

然后打开 CMD 验证：

```cmd
ninja --version #安装成功会显示版本号
```

在项目中创建 `CMakePresets.json` 并配置为：

```json
{
    "version": 6,
    "configurePresets": [
        {
            "name": "windows-msvc-cuda",
            "displayName": "Windows MSVC + CUDA + Ninja",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/build",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug",
                "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
                "CMAKE_CXX_COMPILER": "cl.exe",
                "CMAKE_CUDA_COMPILER": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe",
                "CMAKE_CUDA_HOST_COMPILER": "D:/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"
            }
        }
    ],
    "buildPresets": [
        {
            "name": "windows-msvc-cuda-debug",
            "configurePreset": "windows-msvc-cuda"
        }
    ]
}
```

`generator` 指定构建工具为 Ninja。

`CMAKE_CUDA_COMPILER` 和 `CMAKE_CUDA_HOST_COMPILER` 修改为实际路径。

关键的是 `"CMAKE_EXPORT_COMPILE_COMMANDS": "ON"` ，它会生成：`build/compile_commands.json`，clangd 会通过这个文件获取真实编译参数。

## 安装 clangd 并配置

安装 clangd，不需要安装完整 LLVM。

下载地址：https://github.com/clangd/clangd/releases 。Windows 版本的下载如：

```
clangd-windows-23.1.0.zip
```

下载好后，比如解压到 `D:\clangd_23.1.0` ，其中会有 `D:\clangd_23.1.0\bin\clangd.exe` 。

把 `D:\clangd_23.1.0\bin` 加入到环境变量 `PATH` 中。

然后打开 CMD 验证：

```cmd
clangd --version #安装成功会显示版本号
```

然后再在 VS Code 扩展市场安装 `clangd, Publisher: LLVM` ，然后禁用 Microsoft C/C++ 插件的代码检查和补全 IntelliSense ，否则会和 clangd 冲突，禁用方式：

在 VS Code 全局 `settings.json` 中加入：

```json
{
    "C_Cpp.intelliSenseEngine": "disabled",

    "clangd.arguments": [
        "--background-index",
        "--clang-tidy",
        "--completion-style=detailed"
    ],
}
```

这样做的话，clangd负责补全、诊断、跳转、索引；Microsoft C/C++ 可以继续保留，用于调试等功能。

因为这部分已经写在全局用户设置中，所以项目内不再需要项目设置文件 `.vscode/settings.json` 。

在项目中创建 `.clangd` 并配置为：

```yaml
CompileFlags:
  Add:
    - -xcuda
    - --cuda-path=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0 #修改为实际Cuda路径
```

## CMakeLists.txt 配置 CUDA

项目使用 CMake 管理 CUDA。

示例：

```cmake
cmake_minimum_required(VERSION 3.24)

project(CudaLearning LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(CudaLearning
    main.cu
)
```

## 测试 CUDA 代码

测试代码：

```c++
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU()
{
    printf("Hello World from GPU! I am thread: %d\n", threadIdx.x);
}

int main()
{
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 5>>>();

    cudaDeviceSynchronize();

    return 0;
}
```

配置成功后 clangd 应该正常识别：

```
__global__
threadIdx
blockIdx
blockDim
cudaDeviceSynchronize
<<< >>>
```

并且程序可以正常编译运行。

## 最终项目结构

学习 CUDA 初期，推荐保持简单：

```
Cuda-Learning/
├── .clangd
├── CMakeLists.txt
├── CMakePresets.json
└── main.cu
```

本地生成：

```
build/
.cache/
```

随着代码增多，可以扩展为：

```
Cuda-Learning/
├── .clangd
├── CMakeLists.txt
├── CMakePresets.json
├── include/
│   └── xxx.cuh
└── src/
    ├── main.cu
    └── xxx.cu
```