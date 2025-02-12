# Clion Qt6 开发环境的配置

基于 Clion 的 Qt 6 开发环境配置

# MinGW 编译器

在 Qt6 的官方安装工具中下载 MinGW 编译器，版本号目录下的和 Build Tools 目录下的都要下载。

## 系统变量的配置

进入 Qt6 安装目录，将版本号文件夹和 Tools 文件夹下的 `MinGW` 工具添加进环境变量，如下所示：

```
D:\Qt\6.8.0\mingw_64\bin
D:\Qt\Tools\mingw1310_64\bin
```

## 配置 MinGW 工具链

在 `设置->构建、执行、部署->工具链` 中创建新的基于 `MinGW` 的 Qt6 工具链。

`名称` 填入 `MinGW_Qt`

`工具集` 填入 Tools 目录下的 `MinGw` 工具。例如：

```
D:\Qt\Tools\mingw1310_64
```

`CMake` 中填入 Tools 目录下的 `CMake` 工具。例如：

```
D:\Qt\Tools\cmake\bin\cmake.exe
```

其他可以不用改。

## 配置外部工具

需要添加 `Qt Designer` 和 `UIC` 这两个外部工具。

`Qt Designer` 是一个可视化的用户界面设计工具，用于创建Qt应用程序的用户界面，允许开发人员通过拖放和布局来设计和创建GUI界面。

`uic` 工具读取 `****.ui` 文件，根据 `****.ui` 文件生成相对应的头文件，生成的格式为：`ui_****.h`

### 设置 Qt Designer

将 `Qt Designer` 添加到 `CLion` 的外部工具中，这样就可以在 `CLion` 中使用 `Qt Designer` 打开 `****.ui `文件了。

在 `设置->工具->外部工具` 中创建新的外部工具，名字填入 `Qt Designer`。

转到 `工具设置` ， `程序` 填入 `Qt Designer.exe` 的具体安装路径。例如：

```
D:\Qt\6.8.0\mingw_64\bin\designer.exe
```

`实参` 填入： `$FileName$` 。`工作目录` 填入 `$FileDir$` 。

### 设置 UIC

将 `UIC` 添加到 `CLion` 的外部工具中，每次在 `CLion` 中使用 `Qt Designer` 更改或新建 `****.ui `文件后，都要都要在外部工具(右键当前项目的目录即可找到)里面点击UIC进行编译。

在 `设置->工具->外部工具` 中创建新的外部工具，名字填入 `UIC`。

转到 `工具设置` ， `程序` 填入 `UIC.exe` 的具体安装路径。例如：

```
D:\Qt\6.8.0\mingw_64\bin\uic.exe
```

`实参` 填入： `$FileName$ -o ui_$FileNameWithoutExtension$.h` 。`工作目录` 填入 `$FileDir$` 。

## 创建Qt6项目

在 CLion 中创建新的项目，选择 `Qt微件可执行文件(Qt Widgets Executable)`，Qt版本选择6，C++标准选择17。

`Qt CMake 前缀路径`填入版本号文件夹下的 `MinGW` 工具。例如：

```
D:\Qt\6.8.0\mingw_64
```

在 `设置->构建、执行、部署-> CMake ` 中为新创建的 Qt6 项目选择工具链 `MinGW_Qt`。

综上所述，在 CLion 下的 Qt6 开发环境已配置完成，可以用于开发。

# MSVC 编译器

大致步骤与配置 MinGW 一样。不过 Qt6 的 Tools 目录不提供 MSVC 。所以相较于 MinGW 的配置，MSVC 会简单许多。

从官网下载 Visual Studio，安装后得到 Visual Studio Installer。

选择 `单个组件` ，找到 MSVC 编译器和 Windows SDK，下载最新的版本即可。

## 配置 MSVC 工具链

在 `设置->构建、执行、部署->工具链` 中创建新的基于 `Visual Studio` 的 Qt6 工具链。

`工具集` 填入 Visual Studio 的具体安装路径。例如：

```
D:\Microsoft Visual Studio\2022\Enterprise
```
CLion 会自动扫描并检测。

`架构` 根据电脑具体配置选择，这里选择 `amd64`。其他都可以不更改。

## 配置外部工具

步骤与配置 MinGW 相似。仅需改变 `Qt Designer` 和 `UIC` 两个工具的 exe 文件路径。例如：

```
D:\Qt\6.8.0\msvc2022_64\bin\designer.exe
```
```
D:\Qt\6.8.0\msvc2022_64\bin\uic.exe
```
其他设置都一样。

## 创建Qt6项目

步骤与配置 MinGW 相似。唯一需要改变的就是 Qt CMake 前缀路径。

`Qt CMake 前缀路径`填入版本号文件夹下的 `MSVC` 工具。例如：

```
D:/Qt/6.8.0/msvc2022_64
```

同样在 `设置->构建、执行、部署-> CMake ` 中为新创建的 Qt6 项目选择 `Visual Studio` 工具链。

# 命名规范

类名：首字母大写，单词和单词之间首字母大写。

函数名，变量名：首字母小写，单词和单词之间首字母大写
