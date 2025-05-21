import torch
import matplotlib.pyplot as plt
torch.cuda.init()
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
                                    torch.nn.SiLU())
            ) # 构建前 n-1 层的每层之间的链接

        # 最后一层无需激活
        self.layers.append(
            torch.nn.Sequential(torch.nn.Linear(layers[-2], layers[-1]))
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def physics_residual(Model,x,t): #计算物理残差
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

    # 残差项
    f_pred = physics_residual(Model,X_f,T_f)

    # 边值和初值条件损失的计算准则
    criterion = torch.nn.MSELoss()
    return criterion(U_ic,u_ic_pred) + criterion(U_bc,u_bc_pred) + torch.mean(f_pred**2)

# 生成训练数据，求解区域为 [0,a] × [4,5]
a=30
c1 = 2
c2 = 1.9
x1 = -5
x2 = 5

def phi_i(X, T, ci, xi,A):
    """
    计算 Φ_i(x, t) i=1,2
    """
    x = X.clone()
    t = T.clone()

    arg = x - ci * t - xi
    abs_arg = torch.abs(arg)

    base = ci / torch.cosh(torch.full_like(arg, A/2))

    phi = torch.where(
        abs_arg <= A/2,
        base * torch.cosh(arg),
        base * torch.cosh(A - arg)
    )
    return phi

def exact_solution(x, t):
    """
    构造初始解 U(x, t) = Φ_1(x, t) + Φ_2(x, t)
    保证输入输出一一对应，不改变顺序。
    """
    phi1 = phi_i(x, t, ci=c1, xi=x1, A=a)
    phi2 = phi_i(x, t, ci=c2, xi=x2, A=a)
    U = phi1 + phi2
    return U

# 初值条件采样
x_ic = torch.linspace(0, a,3000).view(-1,1).to(device)
t_ic = 4 * torch.ones_like(x_ic).to(device)
u_ic = exact_solution(x_ic, t_ic)

# 边值条件采样，左端点 x=0
t_bc = torch.linspace(0, 5, 3000).view(-1, 1).to(device)
x_bc = torch.zeros_like(t_bc).to(device)
u_bc = exact_solution(x_bc, t_bc)

# 物理残差采样
x_f = torch.linspace(0.1,a-0.1,8000).view(-1,1).to(device)
t_f = 2 * torch.rand((8000, 1)).to(device)
t_f = t_f + 4

# 创建模型实例并设置优化器
model = PINN([2,20,40,80,40,20,1])
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
for epoch in range(3000):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model,x_ic,t_ic,u_ic,x_bc,t_bc,u_bc,x_f,t_f)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch:{epoch},loss:{loss.item()}")

# 评估
with torch.no_grad():
    model.eval()
    x_test = torch.linspace(0,a,501).reshape(-1,1)
    t_test = 5 * torch.ones_like(x_test)
    u_test = exact_solution(x_test, t_test)
    u_test = u_test.to("cpu").numpy()
    TestData = torch.cat([x_test,t_test],dim=1).to(device)
    u_pred = model(TestData)
    u_pred = u_pred.to("cpu").numpy()

# 绘图
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x_test,u_pred,color='green',label='prediction',alpha=0.8)
ax1.plot(x_test,u_test,
         color='red',linestyle='--',label='exact'
        )
ax1.set_xlabel("x")
ax1.set_ylabel("u",rotation=0)
ax1.set_xlim(-0.5,a+0.5)
ax1.legend()
plt.savefig("PINN Solution2.pdf",format="pdf")
plt.show()