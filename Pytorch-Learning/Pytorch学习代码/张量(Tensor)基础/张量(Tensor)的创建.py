import torch
import numpy as np

"""--------------------torch.tensor()方法--------------------"""
print("--------------------torch.tensor()方法--------------------")
# 使用 Python List 创建
l1 = [[1,2,3],[4,5,6],[7,8,9]]
t_from_list = torch.tensor(l1)
print("由Python列表创建张量：",t_from_list,sep="\n",end="\n\n")

# 使用 Numpy nfarray 创建张量
arr1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
t_from_ndarray = torch.tensor(arr1)
print("由Numpy数组创建张量：",t_from_ndarray,sep="\n",end="\n\n")

"""--------------------torch.from_numpy()方法--------------------"""
print("--------------------torch.from_numpy()方法--------------------")
# 使用此方法创建时的，array改变，tensor也会变，即共享同一块内存
arr2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
t_from_numpy = torch.from_numpy(arr2)
print("torch.from_numpy()方法创建张量：",t_from_numpy,sep="\n",end="\n\n")

# 修改array后，tensor也跟着改变
arr2[0,0] = 0
print("修改后的array",arr2,sep="\n",end="\n\n")
print("修改array后的tensor：",t_from_numpy,sep="\n",end="\n\n")

t_from_numpy[0,0] = -1
print("修改后的tensor：",t_from_numpy,sep="\n",end="\n\n")
print("修改tensor后的array",arr2,sep="\n",end="\n\n")

"""--------------------依数值创建--------------------"""
print("--------------------依数值创建--------------------")
# torch.zeros()，创建全0张量
zero_t1 = torch.zeros([3,3])
print("全 0 tensor",zero_t1,sep="\n",end="\n\n")

# torch.zeros_like()，创建全0张量
zero_t2 = torch.zeros_like(zero_t1) #根据输入的 tensor 的尺寸创建全 0 tensor
print("根据输入tensor的尺寸创建全 0 tensor",zero_t1,sep="\n",end="\n\n")

# torch.ones()和torch.ones_like()同理，就是创建全 1 tensor

# torch.full()，依给定的尺寸和填充值创建张量
full_t = torch.full([3,3],5)
print("依给定的尺寸和填充值创建张量",full_t,sep="\n",end="\n\n")

# torch.full_like()就是根据输入的 tensor 的尺寸和填充值创建
