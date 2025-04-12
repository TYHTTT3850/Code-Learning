"""
scipy.stats模块包含了多种概率分布的随机变量。
所有的连续型随机变量(90多种)都是rv_continuous的派生类。
所有的离散型随机变量(14种)都是rv_discrete的派生类。

随机变量常用方法：
1、rvs()方法：产生随机数，通过 size参数指定输出的数组大小。
2、pdf()方法：连续型随机变量的概率密度函数。
3、pmf()方法：离散型随机变量的概率质量函数。
4、cdf()方法：随机变量的分布函数。F(x) = P(X<=x)。
5、sf()方法：随机变量的生存函数。S(x) = 1-F(x) = P(X>x)。
6、ppf()方法：分布函数的反函数。
7、stats()方法：计算随机变量的期望和方差。
8、fit()方法：对一组随机样本利用极大似然估计法，估计总体中的位置参数。
9、mean()方法：计算随机变量的期望。
10、var()方法：计算随机变量的方差。
11、std()方法：计算随机变量的标准差。
"""

import numpy as np
from scipy import stats


# 创建一个正态分布随机变量
rv = stats.norm(loc=0, scale=1)

# 计算概率密度函数
x = 0
pdf_value = rv.pdf(x)
print(f"PDF at {x}: {pdf_value}")

# 计算累积分布函数
cdf_value = rv.cdf(x)
print(f"CDF at {x}: {cdf_value}")

# 计算生存函数
sf_value = rv.sf(x)
print(f"SF at {x}: {sf_value}")

# 计算分位数
q = 0.95
ppf_value = rv.ppf(q)
print(f"95th percentile: {ppf_value}")

# 生成随机样本
random_samples = rv.rvs(size=5)
print(f"Random samples: {random_samples}")

# 计算均值和方差
mean_value = rv.mean()
var_value = rv.var()
std_value = rv.std()
print(f"Mean: {mean_value}, Variance: {var_value}, Std Dev: {std_value}")

# 同时计算期望和方差
print(f"Mean and Variance are:{rv.stats()}")
