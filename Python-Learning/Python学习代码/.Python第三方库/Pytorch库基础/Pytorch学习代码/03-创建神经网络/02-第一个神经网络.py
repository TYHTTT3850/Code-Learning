import torch
import torch.nn as nn

# 定义输入层大小、隐藏层大小、输出层大小和批量大小(一次喂给模型的训练数据的个数)
n_in, n_h, n_out, batch_size = 10, 5, 1, 10

# 模拟训练数据

## 训练输入数据
X = torch.randn(size=(batch_size,n_in)) # 每一行为一组数据，每组数据的特征个数与输入层相等

## 训练目标值
Y = torch.tensor([[1.0], [0.0], [0.0],[1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]]) #每一组数据对应的输出

# 创建顺序模型，包含线性层、ReLU激活函数和Sigmoid激活函数
model = nn.Sequential(
    nn.Linear(in_features=n_in, out_features=n_h), # 输入层到隐藏层的线性变换
    nn.ReLU(), # 隐藏层的ReLU激活函数
    nn.Linear(in_features=n_h, out_features=n_out), # 隐藏层到输出层的线性变换
    nn.Sigmoid() # 输出层的Sigmoid激活函数
)

criterion = nn.MSELoss() # 均方误差损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # 随机梯度下降优化器

"""
训练过程（Training Process）
训练神经网络涉及以下步骤：
准备数据：通过 DataLoader 加载数据。
定义损失函数和优化器。
前向传播：计算模型的输出。
计算损失：与目标进行比较，得到损失值。
反向传播：通过 loss.backward() 计算梯度。
更新参数：通过 optimizer.step() 更新模型的参数。
重复上述步骤，直到达到预定的训练轮数。
"""
# 执行梯度下降算法进行模型训练
for epoch in range(50): #训练50轮
    model.train() # 设置模型为训练模式
    optimizer.zero_grad() # 梯度清零
    y_pred = model(X) # 向前传播，计算预测值
    loss = criterion(y_pred, Y) # 计算损失
    loss.backward() # 反向传播，计算梯度
    optimizer.step() # 更新模型参数
    print('epoch: ', epoch, 'loss: ', loss.item())  # 打印损失值

"""
测试与评估
训练完成后，需要对模型进行测试和评估。
常见的步骤包括：
1、计算测试集的损失：测试模型在未见过的数据上的表现。
2、计算准确率（Accuracy）：对于分类问题，计算正确预测的比例。
"""

# 假设你有测试集 X_test 和 Y_test
# model.eval()  # 设置模型为评估模式
# with torch.no_grad():  # 在评估过程中禁用梯度计算
#     output = model(X_test)
#     loss = criterion(output, Y_test)
#     print(f'Test Loss: {loss.item():.4f}')
