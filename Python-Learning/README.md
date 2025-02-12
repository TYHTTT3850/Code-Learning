## pip工具常用操作指令

显示所有已安装的库。

```cmd
pip list
```

 查看指定库的完整安装路径。

```cmd
pip show <库名>
```

自动安装最新版的库

```cmd
pip install <库名>
```

安装指定版本的库。

```cmd
pip install <库名>==<版本号>
```

卸载指定的库

```cmd
pip uninstall <库名>   
```

查看可以升级的库

```cmd
pip list -o
```

升级指定库

```cmd
pip install -U <库名>
```



升级pip

```cmd
pip install -U pip
```

## Python模块、包、库、框架的区别和联系

模块Module：后缀为.py的文件(就是Pyhton代码文件)，里面定义了函数，类等等。

包Package：几个模块的集合

库Library：几个包的集合

框架Framework：Python库的集合

模块⊂包⊂库⊂框架

## 多版本 Python 管理

Python 自带 Python 启动器，在 Windows 操作系统上提供了一种方便的方式来运行和管理多个版本的 Python。它是 Python 安装的一部分，特别是在 Windows 上，它使得用户能够在同一台机器上轻松切换和使用不同版本的 Python。

例如，运行 `py -3.9` 会启动 Python 3.9，运行 `py -3.12` 会启动 Python 3.12。

`py -0` 显示所有已安装的 Python 版本。

如果没有明确指定版本，py 会使用系统默认的 Python 版本(环境变量 path 中处于最上层的 Python 版本)。

### 不同版本 Python 安装第三方库

从默认下载源安装最新版本

```cmd

py -<版本> -m pip install <库名>

```

例如：

```cmd

py -3.12 -m pip install numpy

```

指定版本和下载源

```cmd

py -<版本> -m pip install <库名>[==<版本号>] [-i <镜像源地址>]

```

卸载和更新也同理。
