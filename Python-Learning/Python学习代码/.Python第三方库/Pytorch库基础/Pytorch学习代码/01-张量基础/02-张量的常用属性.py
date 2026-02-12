import torch

# 示例张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# 1. 数据类型和形状相关属性
print("数据类型 (dtype):", x.dtype)  # torch.float32
print("维度 (ndim):", x.ndim)  # 2
print("形状 (shape):", x.shape)  # torch.Size([2, 3])
print("元素总数 (numel):", x.numel())  # 6

# 2. 存储位置相关属性
print("是否在GPU上:", x.is_cuda)  # False
print("设备信息:", x.device)  # cpu

# 3. 内存连续性
print("是否内存连续:", x.is_contiguous())  # True

# 4. 梯度相关属性
x_with_grad = torch.tensor([1, 2, 3],dtype=torch.float32,requires_grad=True)
print("是否需要梯度:", x_with_grad.requires_grad)  # True