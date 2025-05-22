### Python 调用 C++代码(ctypes)

```cpp
extern "C" __declspec(dllexport) int add(int a, int b) {
    return a + b;
}
```

```python
import ctypes
import os

# 加载 DLL
dll_path = os.path.abspath("./cmake-build-debug/add.dll")
add_lib = ctypes.CDLL(dll_path)

# 定义函数签名
add_lib.add.argtypes = [ctypes.c_int, ctypes.c_int]
add_lib.add.restype = ctypes.c_int

# 调用
result = add_lib.add(5, 7)
print("结果是：", result)
```

```cmake
cmake_minimum_required(VERSION 3.30)
project(CppProject)

set(CMAKE_CXX_STANDARD 20)

add_library(add SHARED add.cpp)

if(MSVC)
    set_target_properties(add PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

```

### Python 调用 C++代码(pybind)

首先下载 pybind 源码，官网：[pybind/pybind11: Seamless operability between C++11 and Python](https://github.com/pybind/pybind11)

解压后进入 pybind 的源码目录，在该目录下打开终端，依次执行以下命令：

```cmd
mkdir build

cd build

cmake ..

cmake --build . --config Release

cmake --install . --prefix ../install
```

安装完成后，`pybind11` 的头文件将位于 `pybind11/install/include` 目录中。

新建一个 C++ 项目，示例代码如下

```cmake
cmake_minimum_required(VERSION 3.30)
project(example)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加 pybind11 的头文件路径
include_directories("D:/pybind11-2.13.6/install/include")

# 添加 pybind11 的 CMake 模块路径
set(pybind11_DIR "D:/pybind11-2.13.6/install/share/cmake/pybind11")

# 查找 pybind11 包
find_package(pybind11 REQUIRED)

# 添加模块
pybind11_add_module(example example.cpp)

```

```cpp
#include <pybind11/pybind11.h>

int add(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "一个简单的加法模块";
    m.def("add", &add, "加两个数的函数");
}
```

构建 `.pyd` 文件，构建完成后，将 `.pyd` 复制到 Python 项目的根目录下，即可调用

```python
import example

result = example.add(3, 4)
print(result)  # 输出: 7
```

