import torch
import numpy as np
import matplotlib.pyplot as plt
# 设置设备和随机种子
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

class PINN(torch.nn.Module):
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

def physics_residual(Model,x,t): #计算物理残差：u_t - u_xx
    x.requires_grad=True
    t.requires_grad=True
    u = Model(torch.cat([x,t],dim=1)) #传入(x,t)，输出u(x,t)
    u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    u_xxt = torch.autograd.grad(u_xx,t,grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx,x,grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    return u_t - u_xxt + 3*u*u_x - 2*u_x*u_xx - u*u_xxx

def loss_fn(Model,X_ic,T_ic,U_ic,X_bc,T_bc,U_bc,X_f,T_f): #定义损失函数：物理残差+边值条件+初值条件
    """
    :param Model:定义的神经网络
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
    u_ic_pred = Model(torch.cat([X_ic,T_ic],dim=1))

    #边值条件下的预测函数值
    u_bc_pred = Model(torch.cat([X_bc,T_bc],dim=1))

    # 残差项(预测的 ut-uxx 的值)
    f_pred = physics_residual(Model,X_f,T_f)

    # 边值和初值条件损失的计算准则
    criterion = torch.nn.MSELoss()
    return criterion(U_ic,u_ic_pred) + criterion(U_bc,u_bc_pred) + torch.mean(f_pred**2)

# 生成训练数据，求解区域为 [-25,25] × [0,1.5]

# 初值条件采样1000个点，初值条件u(x,0) = 0.2 * exp(-|x|)
x_ic = torch.linspace(-25,25,1000).view(-1,1).to(device)
t_ic = torch.zeros_like(x_ic).to(device) # t=0
u_ic = 0.2 * torch.exp(-torch.abs(x_ic)).to(device)

# 边值条件采样1000个点，边值条件u(-25,t) = 0.2 * exp(-25-0.2t)
t_bc = torch.linspace(0,1.5,1000).view(-1,1).to(device)
# 左端点 x = -25
x_bc = -25 * torch.ones_like(t_bc).to(device)
u_bc = 0.2 * torch.exp(-25-0.2 * t_bc).to(device)

# 残差计算
x_f = torch.linspace(-25,25,5000).view(-1,1).to(device)
t_f = 1.5 * torch.rand((5000, 1)).to(device)

# 创建模型实例并设置优化器
model = PINN([2,20,20,20,1])
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
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
    x_test = np.linspace(-25,25,501).reshape(-1,1)
    t_test = np.ones_like(x_test)
    TestData = torch.cat([torch.tensor(x_test),torch.tensor(t_test)],dim=1).to(device)
    u_pred = model(TestData.float())
    u_pred = u_pred.to("cpu").numpy()

# 绘图
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x_test,u_pred,color='green',label='prediction',alpha=0.8)
ax1.plot(x_test,0.2*np.exp(-np.abs(x_test-0.2*t_test)),
         color='red',linestyle='--',label='exact'
        )
ax1.set_xlabel("x")
ax1.set_ylabel("u",rotation=0)
ax1.set_xlim(-25.5,25.5)
ax1.legend()
plt.savefig("PINN Solution1.pdf",format="pdf")
plt.show()
