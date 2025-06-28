import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

def u_true(X, Y, T):
    result = (X ** 2 * Y ** 2 + torch.exp(-Y)) * torch.cos(2 * torch.pi * T)
    result = result.reshape(-1, 1)
    return result

def v_true(X, Y, T):
    result = (-2 / 3 * X * Y ** 3 + 2 - torch.pi * torch.sin(torch.pi * X)) * torch.cos(2 * torch.pi * T)
    result = result.reshape(-1, 1)
    return result

def p_true(X, Y, T):
    result = -(2 - torch.pi * torch.sin(torch.pi * X)) * torch.cos(2 * torch.pi * Y) * torch.cos(2 * torch.pi * T)
    result = result.reshape(-1, 1)
    return result

def f1_exact(X, Y, T):
    # 预计算常用的项
    pi = torch.pi
    sin_2pi_t = torch.sin(2 * pi * T)
    cos_2pi_t = torch.cos(2 * pi * T)
    cos2_2pi_t = cos_2pi_t ** 2
    cos_pi_x = torch.cos(pi * X)
    sin_pi_x = torch.sin(pi * X)
    cos_2pi_y = torch.cos(2 * pi * Y)

    term1 = -2 * pi * (X ** 2 * Y ** 2 + torch.exp(-Y)) * sin_2pi_t

    term2 = (
                    2 * X * Y ** 2 * (X ** 2 * Y ** 2 + torch.exp(-Y)) + (-2 / 3 * X * Y ** 3 + 2 - pi * sin_pi_x) * (2 * X ** 2 * Y - torch.exp(-Y))
            ) * cos2_2pi_t

    term3 = pi ** 2 * cos_pi_x * cos_2pi_y * cos_2pi_t

    term4 = -(2 * Y ** 2 + 2 * X ** 2 + torch.exp(-Y)) * cos_2pi_t

    f1 = term1 + term2 + term3 + term4
    return f1.reshape(-1, 1)

def f2_exact(X, Y, T):
    pi = torch.pi

    # 预计算常用项
    sin_pi_x = torch.sin(pi * X)
    cos_pi_x = torch.cos(pi * X)
    sin_2pi_t = torch.sin(2 * pi * T)
    cos_2pi_t = torch.cos(2 * pi * T)
    cos2_2pi_t = cos_2pi_t ** 2
    sin_2pi_y = torch.sin(2 * pi * Y)

    # f2 各部分
    term1 = -2 * pi * (-2 / 3 * X * Y ** 3 + 2 - pi * sin_pi_x) * sin_2pi_t

    term2 = (
                    (X ** 2 * Y ** 2 + torch.exp(-Y)) * (-2 / 3 * Y ** 3 - pi ** 2 * cos_pi_x)
                    + (-2 / 3 * X * Y ** 3 + 2 - pi * sin_pi_x) * (-2 * X * Y ** 2)
            ) * cos2_2pi_t

    term3 = 2 * pi * (2 - pi * sin_pi_x) * sin_2pi_y * cos_2pi_t

    term4 = (2 * X * Y - pi ** 3 * torch.sin(pi * X)) * cos_2pi_t
    f2 = term1 + term2 + term3 + term4
    return f2.reshape(-1, 1)

def f_true(X, Y, T):
    f1 = f1_exact(X, Y, T)
    f2 = f2_exact(X, Y, T)
    return f1, f2

def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

def f_pred(Model, X, Y, T):
    X.requires_grad = True
    Y.requires_grad = True
    T.requires_grad = True
    u, v, p = Model(X, Y, T)
    u_x = grad(u, X)
    u_y = grad(u, Y)
    u_t = grad(u, T)
    u_xx = grad(u_x, X)
    u_yy = grad(u_y, Y)
    v_x = grad(v, X)
    v_y = grad(v, Y)
    v_t = grad(v, T)
    v_xx = grad(v_x, X)
    v_yy = grad(v_y, Y)
    p_x = grad(p, X)
    p_y = grad(p, Y)
    f1 = u_t + u * u_x + v * u_y + p_x - u_xx - u_yy
    f2 = v_t + u * v_x + v * v_y + p_y - v_xx - v_yy
    f_div = u_x + v_y
    return f1,f2,f_div

class Solution(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.out = 0
        # 创建中间层
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        # 创建自适应权重
        self.loss_weights = {
            'u_ic': nn.Parameter(torch.tensor(1.0)),
            'v_ic': nn.Parameter(torch.tensor(1.0)),
            'p_ic': nn.Parameter(torch.tensor(1.0)),
            'u_bc': nn.Parameter(torch.tensor(1.0)),
            'v_bc': nn.Parameter(torch.tensor(1.0)),
            'p_bc': nn.Parameter(torch.tensor(1.0)),
            'f_1': nn.Parameter(torch.tensor(1.0)),
            'f_2': nn.Parameter(torch.tensor(1.0)),
            'div': nn.Parameter(torch.tensor(1.0))
        }
        self.init_weights()

    # 模型参数初始化
    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Xavier初始化
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, X, Y, T):
        self.out = torch.cat([X, Y, T], dim=1)
        for layer in self.layers[:-1]:
            self.out = torch.tanh(layer(self.out))
        self.out = self.layers[-1](self.out)
        # 分离输出 u1, u2, p
        return self.out[:, 0:1], self.out[:, 1:2], self.out[:, 2:3]

    def loss_fn(self, X_f, Y_f, T_f, X_bc, Y_bc, T_bc, X_ic, Y_ic, T_ic):
        """
            :param X_f:残差X
            :param Y_f:残差X
            :param T_f:残差T
            :param X_bc:边值X
            :param Y_bc:边值Y
            :param T_bc:边值T
            :param X_ic:初值X
            :param Y_ic:初值Y
            :param T_ic:初值T
        """
        error = nn.MSELoss(reduction='mean')
        with torch.no_grad():
            u_ic_true = u_true(X_ic, Y_ic, T_ic).reshape(-1, 1)
            v_ic_true = v_true(X_ic, Y_ic, T_ic).reshape(-1, 1)
            p_ic_true = p_true(X_ic, Y_ic, T_ic).reshape(-1, 1)
            u_bc_true = u_true(X_bc, Y_bc, T_bc).reshape(-1, 1)
            v_bc_true = v_true(X_bc, Y_bc, T_bc).reshape(-1, 1)
            p_bc_true = p_true(X_bc, Y_bc, T_bc).reshape(-1, 1)
            f1_true, f2_true = f_true(X_f, Y_f, T_f)
            f1_true = f1_true.reshape(-1, 1)
            f2_true = f2_true.reshape(-1, 1)

        # 初值估计
        u_ic_pred, v_ic_pred, p_ic_pred = self(X_ic, Y_ic, T_ic)
        u_ic_pred = u_ic_pred.reshape(-1, 1)
        v_ic_pred = v_ic_pred.reshape(-1, 1)
        p_ic_pred = p_ic_pred.reshape(-1, 1)

        # 边值估计
        u_bc_pred, v_bc_pred, p_bc_pred = self(X_bc, Y_bc, T_bc)
        u_bc_pred = u_bc_pred.reshape(-1, 1)
        v_bc_pred = v_bc_pred.reshape(-1, 1)
        p_bc_pred = p_bc_pred.reshape(-1, 1)

        # 残差估计
        f1_pred, f2_pred, div = f_pred(self, X_f, Y_f, T_f)

        # 损失计算
        loss_ic = (
                self.loss_weights['u_ic']*error(u_ic_true, u_ic_pred) +
                self.loss_weights['v_ic']*error(v_ic_true, v_ic_pred) +
                self.loss_weights['p_ic']*error(p_ic_true, p_ic_pred)
        )
        loss_bc = (
                self.loss_weights['u_bc']*error(u_bc_true, u_bc_pred) +
                self.loss_weights['v_bc']*error(v_bc_true, v_bc_pred) +
                self.loss_weights['p_bc']*error(p_bc_true, p_bc_pred)
        )
        loss_f = (
                self.loss_weights['f_1']*error(f1_true, f1_pred) +
                self.loss_weights['f_2']*error(f2_true, f2_pred)
        )
        loss_div = self.loss_weights['div']*error(div, torch.zeros_like(div))

        # 加权总损失
        total_loss = loss_ic + loss_bc + loss_f + loss_div

        return total_loss

# 生成训练数据
# ----------物理残差----------------
# 使用网格采样生成物理残差点
nx, ny, nt = 50, 50, 10  # 根据需要设置网格的密度
x_vals = torch.linspace(0, 1, nx).reshape(-1, 1).to(device)
y_vals = torch.linspace(-0.25, 0, ny).reshape(-1, 1).to(device)
t_vals = torch.linspace(0, 1, nt).reshape(-1, 1).to(device)

# 先生成二维网格, 再扩展到时
xg, yg = torch.meshgrid(x_vals.squeeze(), y_vals.squeeze(), indexing='ij')
xg = xg.reshape(-1, 1)
yg = yg.reshape(-1, 1)

# 为每个时刻取整个空间网格
x_f = []
y_f = []
t_f = []
for t in t_vals:
    t_tmp = t * torch.ones_like(xg).to(device)
    x_f.append(xg)
    y_f.append(yg)
    t_f.append(t_tmp)
x_f = torch.cat(x_f, dim=0)
y_f = torch.cat(y_f, dim=0)
t_f = torch.cat(t_f, dim=0)

# -----------初值条件---------------
x_ic_vals = torch.linspace(0, 1, 80).to(device)
y_ic_vals = torch.linspace(-0.25, 0, 50).to(device)
x_ic, y_ic = torch.meshgrid(x_ic_vals, y_ic_vals, indexing='ij')
x_ic = x_ic.reshape(-1, 1)
y_ic = y_ic.reshape(-1, 1)
t_ic = torch.zeros_like(x_ic).to(device)  # 初始时刻为0

# ----------边界条件(每个时间点采样整个边界线)------------
n_bc_spatial = 50  # 每个边界线的空间点数
n_bc_time = 20     # 时间点数
# 左边界 x=0
x_left = torch.zeros((n_bc_spatial * n_bc_time, 1))
y_left = torch.linspace(-0.25, 0, n_bc_spatial).repeat(n_bc_time).reshape(-1, 1)
t_left = torch.linspace(0, 1, n_bc_time).repeat_interleave(n_bc_spatial).reshape(-1, 1)

# 右边界 x=1
x_right = torch.ones((n_bc_spatial * n_bc_time, 1))
y_right = torch.linspace(-0.25, 0, n_bc_spatial).repeat(n_bc_time).reshape(-1, 1)
t_right = torch.linspace(0, 1, n_bc_time).repeat_interleave(n_bc_spatial).reshape(-1, 1)

# 下边界 y=-0.25
x_bottom = torch.linspace(0, 1, n_bc_spatial).repeat(n_bc_time).reshape(-1, 1)
y_bottom = -0.25 * torch.ones((n_bc_spatial * n_bc_time, 1))
t_bottom = torch.linspace(0, 1, n_bc_time).repeat_interleave(n_bc_spatial).reshape(-1, 1)

# 上边界 y=0
x_top = torch.linspace(0, 1, n_bc_spatial).repeat(n_bc_time).reshape(-1, 1)
y_top = torch.zeros((n_bc_spatial * n_bc_time, 1))
t_top = torch.linspace(0, 1, n_bc_time).repeat_interleave(n_bc_spatial).reshape(-1, 1)

x_bc = torch.cat([x_left, x_right, x_bottom, x_top]).to(device)
y_bc = torch.cat([y_left, y_right, y_bottom, y_top]).to(device)
t_bc = torch.cat([t_left, t_right, t_bottom, t_top]).to(device)

# 模型实例，优化器，学习率调度器
model = Solution([3, 64, 64, 64, 64, 64, 3]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)

# 训练
for epoch in range(1, 6501):
    model.train()
    optimizer.zero_grad()
    loss = model.loss_fn(x_f,y_f,t_f,x_bc,y_bc,t_bc,x_ic,y_ic,t_ic)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % 500 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"epoch:{epoch},loss:{loss:.4f},lr:{current_lr:.6f}")

# -------------------- 第二阶段精细优化：使用 LBFGS ----------------------
# 当 AdamW 达到初步收敛后，切换至 LBFGS 进行精细调优
optimizer_lbfgs = optim.LBFGS(model.parameters(),
                              lr=1.0, max_iter=1000, max_eval=1000,
                              tolerance_grad=1e-6, tolerance_change=1e-10, history_size=100)

def closure():
    optimizer_lbfgs.zero_grad()
    loss_lbfgs = model.loss_fn(x_f,y_f,t_f,x_bc,y_bc,t_bc,x_ic,y_ic,t_ic)
    loss_lbfgs.backward()
    return loss_lbfgs

print("Starting LBFGS optimization ...")
optimizer_lbfgs.step(closure)
print("LBFGS optimization finished.")

# 评估代码：在 x ∈ [0,1] 和 y ∈ [-0.25, 0] 上选取一个网格，并固定 t = 1
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    # 构建新的网格点，用于测试
    nx, ny = 100, 100  # 网格划分数
    x = np.linspace(0, 1, nx)
    y = np.linspace(-0.25, 0, ny)
    # 使用 "ij" 排序构造网格
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")

    # 转换为 torch 张量，并调整为 (N, 1)
    x_test = torch.tensor(x_mesh, dtype=torch.float32).reshape(-1, 1).to(device)
    y_test = torch.tensor(y_mesh, dtype=torch.float32).reshape(-1, 1).to(device)
    t_val = 1.0  # 固定时刻 t = 1
    t_test = torch.full((x_test.size(0), 1), t_val, dtype=torch.float32, device=device)

    # 计算模型预测值
    u_pred, v_pred, p_pred = model(x_test, y_test, t_test)
    # 转换为 numpy，并 reshape 成网格形状
    u_pred_np = u_pred.cpu().detach().numpy().reshape(nx, ny)
    v_pred_np = v_pred.cpu().detach().numpy().reshape(nx, ny)
    p_pred_np = p_pred.cpu().detach().numpy().reshape(nx, ny)

    # 计算真实解
    u_true_np = u_true(x_test, y_test, t_test).cpu().detach().numpy().reshape(nx, ny)
    v_true_np = v_true(x_test, y_test, t_test).cpu().detach().numpy().reshape(nx, ny)
    p_true_np = p_true(x_test, y_test, t_test).cpu().detach().numpy().reshape(nx, ny)

    # 计算误差
    error_u = np.abs(u_true_np - u_pred_np)
    error_v = np.abs(v_true_np - v_pred_np)
    error_p = np.abs(p_true_np - p_pred_np)

    # 绘图对比：采用 3 行 3 列布局，每个物理量设置为:
    fig, axs = plt.subplots(3, 3, figsize=(16, 16))
    titles = ['Exact u', 'Predicted u', 'Error u',
              'Exact v', 'Predicted v', 'Error v',
              'Exact p', 'Predicted p', 'Error p']
    data = [u_true_np, u_pred_np, error_u,
            v_true_np, v_pred_np, error_v,
            p_true_np, p_pred_np, error_p]

    for idx, ax in enumerate(axs.flat):
        # 如果在第三列（误差图），选择 cmap='coolwarm'，否则使用 'rainbow'
        col = idx % 3
        cmap_choice = 'coolwarm' if col == 2 else 'rainbow'
        cf = ax.contourf(x_mesh, y_mesh, data[idx], levels=100, cmap=cmap_choice)
        fig.colorbar(cf, ax=ax)
        ax.set_title(titles[idx])

    # 调整图片布局
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig("二阶非稳态Navier-Stokes方程.pdf", format="pdf")
