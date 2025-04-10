"""
以一维热传导方程：u_t-u_xx = 0为例
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

class FCN(torch.nn.Module):
    def __init__(self,layers):
        """
        参数:
        layers: 一个包含每一层神经元个数的列表，例如 [2, 20, 20, 1]表示输入层有2个神经元(x, t)，两个隐藏层各20个神经元，输出层1个神经元(预测的 u)
        """
        super().__init__()
        self.layers = torch.nn.ModuleList() # ModuleList是专门用来存放神经网络的层的列表

        # 设置前 n-1 层之间的连接，并激活。最后一层输出不需要激活
        # 前 n-1 层
        for i in range(len(layers)-2):
            self.layers.append(
                torch.nn.Sequential(torch.nn.Linear(layers[i], layers[i+1]),
                                    torch.nn.Tanh())
            ) # 构建前 n-1 层的每层之间的链接

        # 最后一层无需激活
        self.layers.append(
            torch.nn.Sequential(torch.nn.Linear(layers[-2], layers[-1]))
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def physics_residual(model,x,t): #计算物理残差：u_t - u_xx
    x.requires_grad=True
    t.requires_grad=True
    u = model(torch.cat([x,t],dim=1)) #传入(x,t)，输出u(x,t)
    u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True)[0] #计算u对t偏导
    u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True)[0] #u对x偏导
    u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True)[0] #u对x二阶偏导
    return u_t-u_xx

def loss_fn(model,X_ic,T_ic,U_ic,X_bc,T_bc,U_bc,X_f,T_f): #定义损失函数：物理残差+边值条件+初值条件
    """
    :param model:定义的神经网络
    :param X_ic: 初值条件下的x的值
    :param T_ic: 初值条件下的t值
    :param U_ic: 初值条件下的真实函数值
    :param X_bc: 边值条件下的x值
    :param T_bc: 边值条件下的t值
    :param U_bc: 边值条件下的真实函数值
    :param X_f: 计算物理残差的x值采样点
    :param T_f: 计算物理残差的t值采样点
    :return: 损失函数
    """

    #初值条件下的预测函数值
    u_ic_pred = model(torch.cat([X_ic,T_ic],dim=1))

    #边值条件下的预测函数值
    u_bc_pred = model(torch.cat([X_bc,T_bc],dim=1))

    # 残差项(预测的 ut-uxx 的值)
    f_pred = physics_residual(model,X_f,T_f)

    # 边值和初值条件损失的计算准则
    criterion = torch.nn.MSELoss()
    return criterion(u_ic_pred,U_ic) + criterion(u_bc_pred,U_bc) + torch.mean(f_pred**2)

# 生成训练数据，求解区域为 [0,1] × [0,1]

# 初值条件采样100个点，初值条件u(x,0) = sin(pi*x)
x_ic = torch.linspace(0,1,100).view(-1,1) # 更改形状为 100 * 1
t_ic = torch.zeros_like(x_ic) # t=0
u_ic = torch.sin(np.pi*x_ic)

# 边值条件采样100个点，边值条件u(0,t) = u(1,t) = 0
t_bc = torch.linspace(0,1,100).view(-1,1) # 更改形状为 100 * 1
x0 = torch.zeros_like(t_bc) # 左端点 x = 0
x1 = torch.ones_like(t_bc) # 右端点 x = 1
x_bc = torch.cat([x0,x1],dim=0)
t_bc = torch.cat([t_bc, t_bc],dim=0)
u_bc = torch.zeros_like(t_bc)

# 残差计算采样5000个点
torch.manual_seed(42)
eps = 1e-5
# x_f = torch.rand(size=(5000,1)) #会取到0，而边界条件已给定，所以尽量不要取到0
# t_f = torch.rand(size=(5000,1)) # 同理
x_f = (1 - 2 * eps) * torch.rand((5000, 1)) + eps  # x ∈ (eps, 1 - eps)
t_f = (1 - 2 * eps) * torch.rand((5000, 1)) + eps  # t 同理

# 创建模型实例并设置优化器
model = FCN([2,40,40,1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model,x_ic,t_ic,u_ic,x_bc,t_bc,u_bc,x_f,t_f)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch:{epoch},loss:{loss.item()}")

# 评估
with torch.no_grad():
    model.eval()
    x_test = torch.linspace(0,1,51).view(-1,1)
    t_test = 0.5*torch.ones_like(x_test)
    u_pred = model(torch.cat([x_test,t_test],dim=1))
# 绘图
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("solution at t = 0.5")
ax1.scatter(x_test,u_pred,color='blue',label='pred')
ax1.plot(np.linspace(0,1,51),
         np.sin(np.pi*np.linspace(0,1,51))*np.exp(-np.pi**2*0.5),
         color='red',linestyle='--',alpha=0.5,label='exact'
        )
ax1.set_xlabel("x")
ax1.set_ylabel("u",rotation=0)
ax1.set_xlim(0,1)
ax1.set_ylim(-0.1,0.1)
ax1.legend()
plt.show()