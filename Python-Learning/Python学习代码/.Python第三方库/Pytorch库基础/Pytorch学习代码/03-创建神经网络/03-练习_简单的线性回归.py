import torch
import numpy as np
import matplotlib.pyplot as plt

# 随机种子，确保每次运行结果一致
torch.manual_seed(42)

# 生成训练数据
X = torch.randn(100, 2)  # 100 个样本，每个样本 2 个特征
true_w = torch.tensor([2.0, 3.0])  # 假设真实权重
true_b = 4.0  # 偏置项
Y = X @ true_w + true_b + torch.randn(100) * 0.1  # 加入一些噪声，Y即为目标值

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=2, out_features=1) # 只需要一层

    def forward(self, x):
        return self.fc1(x) # 线性回归，不需要非线性的激活函数

model = LinearRegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练
for epoch in range(1000):
    y_pred = model(X)
    loss = torch.nn.functional.mse_loss(y_pred.squeeze(), Y) #采用均方误差损失函数
    loss.backward()
    optimizer.step()
    optimizer.zero_grad() # 清零梯度

    if epoch % 10 == 0:
        print(f"epoch: {epoch}, loss: {loss.item()}")

# 绘制
with torch.no_grad():
    y_pred = model(X)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y,color='r',label="True Values")
ax.scatter(X[:, 0], X[:, 1], y_pred.squeeze(),color='b',label="Predictions")
ax.legend()
plt.show()