# SciPy库简介

&emsp;SciPy库是在NumPy的基础上实现的，对NumPy数组和基本的数组运算进行拓展。

&emsp;SciPy支持的功能包括：文件处理、积分、方程求解、数值分析、优化方法、统计学、信号与图像处理、聚类分析和空间分析等。

## 常用模块

### 微积分模块(scipy.integrate)

&emsp;支持数值积分和微分方程的数值解。

#### (1)给定函数的数值积分

&emsp;quad()：一重数值积分。

&emsp;dblquad()：二重数值积分。

&emsp;tplquad()：三重数值积分。

&emsp;nquad()：通用N重数值积分。

&emsp;fixed_quad()：使用固定阶高斯求积公式求数值积分。

&emsp;quadrature()：使用固定误差限的高斯求积公式求数值积分。

&emsp;romberg()：求函数的Romberg积分。

#### (2)给定离散点的数值积分

&emsp;cumtrapz()：用梯形法求数值积分。

&emsp;simps()：用辛普森法求数值积分。

&emsp;romb()：用Romberg积分法求自变量均匀间隔的离散点的数值积分。

#### (3)微分方程的数值解

&emsp;odeint()：使用FORTRAN库中的方法求微分方程组的数值解。

&emsp;ode()：求一般微分方程组的数值解。

&emsp;complex_ode()：求复微分方程组的数值解。

### 线性代数模块(scipy.linalg)

&emsp;相比于numpy.linalg模块更高级。

### 优化模块(scipy.optimize)

&emsp;解决单变量和多变量的目标函数的最小值问题。通过大量算法解决最小化问题。

&emsp;支持线性回归、搜索函数的最小值与最大值、方程求根、线性规划、拟合等功能。

### 插值模块(scipy.interpolate)

&emsp;支持一维和多维插值，如泰勒(Taylor)多项式插值，一维和多维样条插值。

### 统计学模块(scipy.stats)

&emsp;提供了各种随机变量的分布、统计量的计算、分布拟合、参数检验等功能。

#### (1)常用方法

&emsp;rvs()：产生随机数，通过 size参数指定输出的数组大小。

&emsp;pdf()：计算连续型随机变量在某个特定值处的概率密度。

&emsp;pmf()：计算离散型随机变量在某个特定值处的概率质量。

&emsp;cdf()：计算随机变量的分布函数。$`F(x) = P(\xi ≤ x)`$。

&emsp;sf()：计算随机变量的生存函数。$`S(x) = 1-F(x) = P(\xi > x)`$。

&emsp;ppf()：计算分布函数的反函数(百分位数)。也就是说给定一个$`P(\xi ≤ x)`$计算使$`F(x) = P(\xi ≤ x)`$成立的x值。

&emsp;stats()：计算随机变量的期望和方差。

&emsp;fit()：对一组随机样本利用极大似然估计法，估计总体中的位置参数。

&emsp;mean()：计算随机变量的期望。

&emsp;var()：计算随机变量的方差。

&emsp;std()：计算随机变量的标准差。

### 傅里叶变换模块(scipy.fftpack)

&emsp;适用于一维或多维的离散傅里叶变换

&emsp;fft()：傅里叶变换。

&emsp;ifft()：傅里叶逆变换。

### 信号处理模块(scipy.signal)

&emsp;包含一系列滤波函数、滤波器设计函数，以及对一维和二维数据进行B-样条插值的函数。

&emsp;模块中的方法可以进行以下操作：

&emsp;&emsp;卷积、B-样条、滤波、滤波器设计、MatLab式的IIR滤波器设计、

&emsp;&emsp;连续时间的线性系统、离散时间的线性系统、线性时不变系统、

&emsp;&emsp;信号波形、窗函数、小波分析、光谱分析等。 

### 多维图像处理模块(scipy.ndimage)

&emsp;提供了图像处理的各种方法，例如图像几何变换、图像滤波等。

### 空间分析模块(scipy.spatial)

&emsp;空间数据：与地理空间或垂直空间相关的数据对象。

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;这种数据包括点、线、多边形、其他几何和地理特征信息。

&emsp;空间分析：一系列用于分析空间数据的算法

&emsp;支持Delaunay三角剖分、Voronoi图、N维凸包等功能。

&emsp;支持KD树(scipy.spatial.kdtree)实现快速近邻查找算法。

&emsp;支持对初始向量集合进行距离矩阵的计算。

### 聚类模块(scipy.cluster)

&emsp;聚类：将一个大的集合分成多个组的过程。

&emsp;包含两个子模块：

&emsp;&emsp;scipy.cluster.vq(向量量化)。支持K均值聚类和向量量化。

&emsp;&emsp;scipy.cluster.hierarchy(层次聚类)。支持分层聚类和聚合聚类。

### 文件输入输出模块(scipy.io)

&emsp;支持一系列的文件格式的读和写。

&emsp;支持的文件格式包括：

&emsp;&emsp;MATLAB文件、ALD文件、Matrix Market文件、无格式的FORTRAN文件、

&emsp;&emsp;WAV声音文件、ARFF文件、NetCDF文件。

## 其他模块

### 附件模块(scipy.misc)

&emsp;实现图形读写操作功能

### 稀疏矩阵及其相关算法模块(scipy.sparse)

### 特殊函数模块(scipy.special)
