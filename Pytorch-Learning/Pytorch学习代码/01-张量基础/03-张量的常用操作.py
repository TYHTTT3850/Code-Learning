import torch

# 创建示例张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print("原始张量x:", x,sep="\n",end="\n\n")

"""--------------------张量操作--------------------"""
print("--------------------张量操作--------------------")

# reshape/view - 重塑张量形状
reshaped = x.reshape(3, 2)  # 或 x.view(3, 2)
print("reshape后的张量:", reshaped,sep="\n",end="\n\n")

# squeeze - 移除单例维度(行维度只有一行，列维度只有只有一列，层维度只有一层，这样叫做单例维度)
y = torch.tensor([[[1], [2]]])  # 形状为 [1, 2, 1]
squeezed = y.squeeze()
print("原始y张量形状:", y.shape,end="\n\n")
print("原始y张量:", y,sep="\n",end="\n\n")
print("squeeze后形状:", squeezed.shape,end="\n\n")
print("squeeze后的张量:", squeezed,sep="\n",end="\n\n")

# unsqueeze - 添加维度
unsqueezed = x.unsqueeze(0)  # 在第0维添加一个维度
print("unsqueeze后的张量形状:", unsqueezed.shape,sep="\n",end="\n\n")

# transpose/permute - 转置或重排维度
transposed = x.transpose(0, 1)  # 交换第0维和第1维
print("转置后的张量:", transposed,sep="\n",end="\n\n")

z = torch.randn(2, 3, 4)
permuted = z.permute(2, 0, 1)  # 重排维度顺序为[2,0,1]
print("原始z张量形状:", z.shape)
print("permute后形状:", permuted.shape,sep="\n",end="\n\n")

"""--------------------数学运算--------------------"""
print("--------------------数学运算--------------------")

a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([4, 5, 6], dtype=torch.float32)

# 基本运算
print("加法:", a + b,sep="\n",end="\n\n")
print("减法:", a - b,sep="\n",end="\n\n")
print("乘法:", a * b,sep="\n",end="\n\n")  # 元素级乘法
print("除法:", a / b,sep="\n",end="\n\n")

# 矩阵乘法
c = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
d = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
print("矩阵乘法 (matmul):", torch.matmul(c, d),sep="\n",end="\n\n")
print("矩阵乘法 (@运算符):", c @ d,sep="\n",end="\n\n")

# 点积
dot_product = torch.dot(a, b)
print("向量点积:", dot_product,sep="\n",end="\n\n")

# 求和、平均值、最大值、最小值
print("求和:", a.sum(),sep="\n",end="\n\n")
print("平均值:", a.mean(),sep="\n",end="\n\n")
print("最大值:", a.max(),sep="\n",end="\n\n")
print("最小值:", a.min(),sep="\n",end="\n\n")

"""--------------------索引切片--------------------"""
print("--------------------索引切片--------------------")
e = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("原始张量e:", e,sep="\n",end="\n\n")

# 基本索引
print("取第一行:", e[0],sep="\n",end="\n\n")
print("取第二列:", e[:, 1],sep="\n",end="\n\n")
print("取子矩阵:", e[0:2, 1:3],sep="\n",end="\n\n")

# 布尔索引
mask = e > 5
print("大于5的布尔掩码:", mask,sep="\n",end="\n\n")
print("大于5的元素:", e[mask],sep="\n",end="\n\n")

# 花式索引
indices = torch.tensor([0, 2])
print("取第0行和第2行:", e[indices],sep="\n",end="\n\n")

# index_select
selected = torch.index_select(e, dim=0, index=indices)
print("index_select结果:", selected,sep="\n",end="\n\n")

"""--------------------拆分与拼接--------------------"""
print("--------------------拆分与拼接--------------------")
f = torch.ones((2,2))
g = torch.zeros((2,2))

# cat - 沿着已有维度拼接
cat_result1 = torch.cat((f, g), dim=0)  # 垂直拼接(行方向拼接)
print("cat 垂直拼接结果:", cat_result1,sep="\n",end="\n\n")

cat_result2 = torch.cat((f, g), dim=1)  # 水平拼接(列方向拼接)
print("cat 水平拼接结果:", cat_result2,sep="\n",end="\n\n")

# stack - 沿着新维度拼接
stack_result = torch.stack((f, g), dim=0)
print("stack 结果形状:", stack_result.shape,sep="\n",end="\n\n")
print("stack 结果:", stack_result,sep="\n",end="\n\n")

# chunk - 拆分张量
chunks = torch.chunk(e, dim=0, chunks=3) #张量e沿行的方向拆分为3块
print("chunk 拆分结果 - 第一块:", chunks[0],sep="\n",end="\n\n")

# split - 拆分张量
splits = torch.split(e, dim=1, split_size_or_sections=1) #张量e沿列的方向拆分，每1列拆分为一块
print("split 拆分结果 - 第一块:", splits[0],sep="\n",end="\n\n")

"""--------------------梯度相关--------------------"""
print("--------------------梯度相关--------------------")

h = torch.tensor([1., 2., 3.], requires_grad=True)
i = h * 2
j = i.sum()

# backward - 反向传播
j.backward()
print("梯度:", h.grad, sep="\n",end="\n\n")

# detach - 分离张量
detached = i.detach()
print("detach后是否需要梯度:", detached.requires_grad,sep="\n",end="\n\n")

# no_grad - 暂时关闭梯度计算
with torch.no_grad():
    k = h * 3
    print("no_grad下是否需要梯度:", k.requires_grad,sep="\n",end="\n\n")

"""--------------------设备间操作--------------------"""
print("--------------------设备间操作--------------------")

# 注意: 以下代码仅在有GPU的环境中运行
# 检查是否有可用的GPU
if torch.cuda.is_available():
    # 使用.cuda()方法
    l_gpu = e.cuda()
    print("张量已移至GPU:", l_gpu.device)

    # 使用.to()方法
    l_cpu = l_gpu.to("cpu")
    print("张量已移回CPU:", l_cpu.device)
else:
    print("\n没有可用的GPU，跳过设备间操作演示")