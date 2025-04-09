import matplotlib.pyplot as plt
import torch

# 训练数据
torch.manual_seed(42)
X = 2*torch.pi*torch.rand(size=(1000,1))
Y = torch.sin(X) + torch.normal(mean=0.0,std=0.1,size=(1000,1))

# 设计神经网络
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 使用三层神经网络，输入层1个输入，隐藏层10个特征，输出层1个结果
        self.fc1 = torch.nn.Linear(in_features=1,out_features=10)
        self.fc2 = torch.nn.Linear(in_features=10,out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.silu(x) # 激活函数
        x = self.fc2(x)
        return x

model = SimpleNN()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练
for epoch in range(10000):
    y_pred = model(X)
    loss = torch.nn.functional.mse_loss(y_pred, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, loss: {loss.item()}")

# 评估
with torch.no_grad():
    y_pred = model(X)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X, Y,color='blue',alpha=0.5,label='Training data')
X_sorted = torch.sort(X.squeeze())[0]
ax.plot(X_sorted,torch.sin(X_sorted),color='red',linestyle='--',label='Exact Value')
ax.scatter(X,y_pred,color='green',alpha=0.5,label='Prediction')
ax.legend()
plt.show()
