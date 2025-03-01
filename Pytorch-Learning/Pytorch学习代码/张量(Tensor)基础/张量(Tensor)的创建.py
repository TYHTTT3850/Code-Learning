import torch
import numpy as np
from sympy.stats.rv import probability

"""--------------------torch.tensor()方法--------------------"""
print("--------------------torch.tensor()方法--------------------")
# 使用 Python List 创建。
l1 = [[1,2,3],[4,5,6],[7,8,9]]
t_from_list = torch.tensor(l1)
print("由Python列表创建张量：",t_from_list,sep="\n",end="\n\n")

# 使用 Numpy nfarray 创建张量。
arr1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
t_from_ndarray = torch.tensor(arr1)
print("由Numpy数组创建张量：",t_from_ndarray,sep="\n",end="\n\n")

"""--------------------torch.from_numpy()方法--------------------"""
print("--------------------torch.from_numpy()方法--------------------")
# 使用此方法创建时的，array改变，tensor也会变，即共享同一块内存。
arr2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
t_from_numpy = torch.from_numpy(arr2)
print("torch.from_numpy()方法创建张量：",t_from_numpy,sep="\n",end="\n\n")

# 修改array后，tensor也跟着改变。
arr2[0,0] = 0
print("修改后的array",arr2,sep="\n",end="\n\n")
print("修改array后的tensor：",t_from_numpy,sep="\n",end="\n\n")

t_from_numpy[0,0] = -1
print("修改后的tensor：",t_from_numpy,sep="\n",end="\n\n")
print("修改tensor后的array",arr2,sep="\n",end="\n\n")

"""--------------------依数值创建--------------------"""
print("--------------------依数值创建--------------------")
# torch.zeros()，创建全0张量。
zero_t1 = torch.zeros([3,3])
print("全 0 tensor",zero_t1,sep="\n",end="\n\n")

# torch.zeros_like()，创建全0张量。
zero_t2 = torch.zeros_like(zero_t1) #根据输入的 tensor 的尺寸创建全 0 tensor。
print("根据输入tensor的尺寸创建全 0 tensor",zero_t1,sep="\n",end="\n\n")

# torch.ones()和torch.ones_like()同理，就是创建全 1 tensor。

# torch.full()，依给定的尺寸和填充值创建张量。
full_t = torch.full([3,3],5)
print("依给定的尺寸和填充值创建张量",full_t,sep="\n",end="\n\n")

# torch.full_like()就是根据输入的 tensor 的尺寸和指定填充值创建张量

# torch.arange()，创建等差的一维张量
"""
start：起始值。
end：结束值。
step：步长。
"""
arange_t1 = torch.arange(start=0,end=6,step=2) # 区间：[0, 6)
print("创建等差的一维张量",arange_t1,sep="\n",end="\n\n") # [0, 2, 4]，长度=(end-start)/step

# torch.linspace()，创建均匀分布的一维张量
"""
start：起始值。
end：结束值。
steps：一维张量长度(元素个数)。
"""
linspace_t1 = torch.linspace(start=0,end=3,steps=4) # 区间：[0, 3]
print("创建均匀分布的一维张量",linspace_t1,sep="\n",end="\n\n") # [0, 1, 2, 3]

# torch.logspace()，创建对数均分的一维张量
"""
start：起始值(base^start)。
end：结束值(base^end)。
steps：一维张量长度(元素个数)。
base：底数，默认为10。
"""
logspace_t1 = torch.logspace(start=0,end=3,steps=4,base=10) # 区间：[10^0, 10^3]
print("创建均匀分布的一维张量",logspace_t1,sep="\n",end="\n\n") # [1, 10, 100, 1000]

# torch.eye()，创建单位对角矩阵
print("创建单位对角阵")
print("默认创建方阵：",torch.eye(3),sep="\n",end="\n\n")
print("3*4矩阵：",torch.eye(3,4),sep="\n",end="\n\n")

# torch.empty()
empty_t1 = torch.empty(3)
empty_t2 = torch.empty([3,3])
print("创建空张量",empty_t1,empty_t2,sep="\n",end="\n\n") # "空"指的是不进行任何赋值操作

"""torch.empty_like() 之于 torch.empty() 类似于 torch.zeros_like() 之于 torch.zeros()"""

"""--------------------依概率分布创建--------------------"""
print("--------------------依概率分布创建--------------------")
# torch.normal()，采用正态分布随机数为每个元素赋值
"""
mean：均值，可以传入张量或标量。注意数据类型为浮点型。
std：标准差，可以传入张量或标量。注意数据类型为浮点型。
mean 和 std 的传入共有四种组合，每种组合效果不一样。

mean张量+std张量：每个元素服从不同的均值和标准差，均值和标准差分别来自于 mean 和 std 的对应行和列。
mean张量+std标量：每个元素服从不同的均值和相同的标准差，均值来自于 mean 对应行和列。标准差都是标量std。
mean标量+std张量：每个元素服从相同的均值和不同的标准差，均值都是标量mean，标准差值来自于 std 对应行和列。
mean标量+std标量：每个元素服从相同的均值和标准差，均值和标准差分别来自于标量 mean 和标量 std。
"""
mean_tensor = torch.arange(0,2,dtype=torch.float32)
std_tensor = torch.arange(1,0,-0.5,dtype=torch.float32)
print("正态分布张量：")
print(torch.normal(mean=mean_tensor,std=std_tensor),end="\n\n") #尺寸根据传入的张量决定。
print(torch.normal(mean=0,std=1,size=(3,3)),end="\n\n") # 标量+标量需要制定输出的尺寸。

# torch.rand()，采用[0,1)区间上均匀分布的随机数为每个元素赋值
print("均匀分布张量：")
print(torch.rand(size=(3,3)),end="\n\n")

"""torch.rand_like() 之于 torch.rand() 类似于 torch.zeros_like() 之于 torch.zeros()"""

# torch.randint()，采用[low,high)区间上均匀分布的随机整数为每个元素赋值
"""
low：下限。
high：上限。
size：尺寸。
"""
print("均匀分布整数张量：")
print(torch.randint(low=3,high=10,size=(3,3)),end="\n\n")

"""torch.randint_like() 之于 torch.randint() 类似于 torch.zeros_like() 之于 torch.zeros()"""

# torch.randn()，采用标准正态分布的随机数为每个元素赋值
print("标准正态分布张量：")
print(torch.randn(size=(3,3)),end="\n\n")

"""torch.randn_like() 之于 torch.randn() 类似于 torch.zeros_like() 之于 torch.zeros()"""

# torch.randperm()，生成随机排列。perm：premutation。
print("随机排列张量：")
print(torch.randperm(n=5),end="\n\n") # 范围为 0 到 n-1

# torch.bernoulli()，采用伯努利分布(0-1分布，两点分布)的随机数为每个元素赋值
"""
input：为 1 的概率。注意传入的是 tensor。
"""
print("伯努利分布张量：")
probability_tensor = torch.rand(size=(3,3))
print(torch.bernoulli(input=probability_tensor),end="\n\n")