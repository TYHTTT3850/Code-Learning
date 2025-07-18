import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using device:",device)

# ======================================================================
# 1. 定义 MLP 网络
# ======================================================================
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=4, neurons=64):
        super(MLP, self).__init__()
        layers = []
        # 第一层：输入层到隐藏层
        layers.append(nn.Linear(input_dim, neurons))
        layers.append(nn.Tanh())
        # 构建多个隐藏层
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.Tanh())
        # 输出层
        layers.append(nn.Linear(neurons, output_dim))
        self.network = nn.Sequential(*layers)

        # 创建自适应权重（例如用于多任务损失加权）
        self.loss_weights = {
            'initial': nn.Parameter(torch.tensor(1.0)),
            'boundary': nn.Parameter(torch.tensor(1.0)),
            'residual': nn.Parameter(torch.tensor(1.0)),
            'interface': nn.Parameter(torch.tensor(1.0)),
        }

        # 调用参数初始化
        self.init_weights()

    def init_weights(self):
        """
        对模型中所有 Linear 层进行 Xavier 初始化，
        并将对应的 bias 初始化为 0。
        """
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)


# 网络1：达西区域（输出标量 φ）
model_darcy = MLP(input_dim=3, output_dim=1, hidden_layers=5, neurons=64).to(device)
# 网络2：斯托克斯区域（输出 u1, u2, p）
model_stokes = MLP(input_dim=3, output_dim=3, hidden_layers=5, neurons=64).to(device)

# ======================================================================
# 2. 定义解析解
# ======================================================================
def phi_D_exact(x, y, t):
    return (torch.exp(y) - torch.exp(-y)) * torch.sin(x) * torch.exp(t)

def u_stokes_exact(x, y, t):
    u1 = 1/torch.pi*torch.sin(2*torch.pi * y) * torch.cos(x) * torch.exp(t)
    u2 = (-2.0 + 1/torch.pi**2*torch.sin(y) ** 2) * torch.sin(x) * torch.exp(t)
    return u1, u2

def p_stokes_exact(x, y, t):
    return torch.zeros_like(x)

# ======================================================================
# 3. 采样方案
# ======================================================================

def adaptive_sampler(n, domain, adapt_fraction=1,adapt_x_range=(0.5, 2.5), adapt_y_range=(-0.7, -0.2)):
    """
    生成内部采样点，其中：
    - domain = [xmin, xmax, ymin, ymax, tmin, tmax] 定义整体采样区域；
    - adapt_fraction 表示在额外采样中占总点数的比例，用于误差较大的区域（u₂高误差区域）；
    - adapt_y_range 为针对 u₂ 误差高的区域在 y 方向的子区间。
    返回采样点 x, y, t 均为 numpy 数组。
    """
    n_adapt = int(n * adapt_fraction)   # 自适应区域内采样点数
    n_uniform = n # 整体区域内均匀采样的点数

    # 生成均匀采样点（覆盖整个区域）
    x_uniform = np.random.uniform(domain[0], domain[1], (n_uniform, 1))
    y_uniform = np.random.uniform(domain[2], domain[3], (n_uniform, 1))
    t_uniform = np.random.uniform(domain[4], domain[5], (n_uniform, 1))

    # 自适应采样：在全 x、全 t 范围内，但 y 仅在 adapt_y_range 内采样
    x_adapt = np.random.uniform(adapt_x_range[0], adapt_x_range[1], (n_adapt, 1))
    y_adapt = np.random.uniform(adapt_y_range[0], adapt_y_range[1], (n_adapt, 1))
    t_adapt = np.random.uniform(domain[4], domain[5], (n_adapt, 1))

    # 合并均匀采样和自适应采样得到总点集
    x_total = np.vstack([x_uniform, x_adapt])
    y_total = np.vstack([y_uniform, y_adapt])
    t_total = np.vstack([t_uniform, t_adapt])
    return x_total, y_total, t_total

def sampler_initial(n, domain):
    xmin, xmax, ymin, ymax, tmin, tmax = domain
    Xs, Ys, Ts = [], [], []

    # 初始条件：t=tmin
    x0 = np.random.uniform(xmin, xmax, (n, 1))
    y0 = np.random.uniform(ymin, ymax, (n, 1))
    t0 = tmin * np.ones((n, 1))
    return x0,y0,t0

def sampler_boundary(n, domain):
    xmin, xmax, ymin, ymax, tmin, tmax = domain
    Xs, Ys, Ts = [], [], []
    # x = xmin 与 x = xmax
    x_left = xmin * np.ones((n, 1))
    y_left = np.random.uniform(ymin, ymax, (n, 1))
    t_left = np.random.uniform(tmin, tmax, (n, 1))
    Xs.append(x_left);
    Ys.append(y_left);
    Ts.append(t_left)

    x_right = xmax * np.ones((n, 1))
    y_right = np.random.uniform(ymin, ymax, (n, 1))
    t_right = np.random.uniform(tmin, tmax, (n, 1))
    Xs.append(x_right);
    Ys.append(y_right);
    Ts.append(t_right)

    # y = ymin 与 y = ymax
    x_bottom = np.random.uniform(xmin, xmax, (n, 1))
    y_bottom = ymin * np.ones((n, 1))
    t_bottom = np.random.uniform(tmin, tmax, (n, 1))
    Xs.append(x_bottom);
    Ys.append(y_bottom);
    Ts.append(t_bottom)

    x_top = np.random.uniform(xmin, xmax, (n, 1))
    y_top = ymax * np.ones((n, 1))
    t_top = np.random.uniform(tmin, tmax, (n, 1))
    Xs.append(x_top);
    Ys.append(y_top);
    Ts.append(t_top)

    Xs = np.vstack(Xs)
    Ys = np.vstack(Ys)
    Ts = np.vstack(Ts)
    return Xs, Ys, Ts

def sampler_interface(n):
    # 接口：y=0, x in [0, π] , t in [0, T_final]
    x = np.random.uniform(0.0, np.pi, (n, 1))
    y = np.zeros((n, 1))
    t = np.random.uniform(0.0, T_final, (n, 1))
    return x, y, t

# ======================================================================
# 4. 构造残差
# ======================================================================

# 达西区域 PDE
def darcy_residual(model, x, y, t):
    # 拼接输入，注意 x, y, t 均应具有 requires_grad=True
    X = torch.cat([x, y, t], dim=1)
    phi = model(X)

    # 计算时间导数 phi_t
    phi_t = torch.autograd.grad(
        phi, t,
        grad_outputs=torch.ones_like(phi),
        create_graph=True
    )[0]

    # 计算空间一阶导数
    grad_phi = torch.autograd.grad(
        phi, [x, y],
        grad_outputs=torch.ones_like(phi),
        create_graph=True
    )
    phi_x, phi_y = grad_phi[0], grad_phi[1]

    # 计算二阶空间导数
    phi_xx = torch.autograd.grad(
        phi_x, x,
        grad_outputs=torch.ones_like(phi_x),
        create_graph=True
    )[0]
    phi_yy = torch.autograd.grad(
        phi_y, y,
        grad_outputs=torch.ones_like(phi_y),
        create_graph=True
    )[0]

    # 右端项 f_D
    f_D = (torch.exp(y) - torch.exp(-y)) * torch.sin(x) * torch.exp(t)

    # 构造残差：phi_t - (phi_xx+phi_yy) - f_D = 0
    res = phi_t - phi_xx - phi_yy - f_D
    return res

# 斯托克斯区域 PDE
def stokes_residual(model, x, y, t):
    # 拼接输入
    X = torch.cat([x, y, t], dim=1)
    pred = model(X)
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]

    # 计算时间导数
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    # 计算一阶和二阶空间导数 (对于 u)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    # 计算一阶和二阶空间导数 (对于 v)
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

    # 计算一阶压力梯度
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    # 根据公式给出右端项 f1 与 f2
    f1 = (2 / np.pi + 4 * np.pi ) * torch.sin(2 * np.pi * y) * torch.cos(x) * torch.exp(t)
    f2 = 2 * (-2 + (1 / (np.pi ** 2)) * (torch.sin(np.pi * y)) ** 2 - torch.cos(2 * np.pi * y)) * torch.sin(x) * torch.exp(t)

    r1 = u_t + p_x - (u_xx + u_yy) - f1
    r2 = v_t + p_y - (v_xx + v_yy) - f2

    # 无散条件残差
    incompress = u_x + v_y

    return r1, r2, incompress

# 接口残差
def interface_residual(model_stokes, model_darcy, x, t):
    y = torch.zeros_like(x, requires_grad=True)
    X_stokes = torch.cat([x, y, t], dim=1)
    X_darcy = torch.cat([x, y, t], dim=1)

    pred_stokes = model_stokes(X_stokes)
    u1 = pred_stokes[:, 0:1]
    u2 = pred_stokes[:, 1:2]
    p = pred_stokes[:, 2:3]

    phi = model_darcy(X_darcy)
    phi_y = torch.autograd.grad(phi, y, grad_outputs=torch.ones_like(phi), create_graph=True)[0]

    r_normal = u2 + phi_y
    u2_y = torch.autograd.grad(u2, y, grad_outputs=torch.ones_like(u2), create_graph=True)[0]
    r_force = 2 * u2_y - p + phi
    r_tangent = u1
    return r_normal, r_force, r_tangent

# ======================================================================
# 5. 训练设置与采样（区域调整如下）
# 达西区域： x in [0, π], y in [0,1]
# Stokes区域： x in [0, π], y in [-1, 0]
# ======================================================================
T_final = 1.0
domain_darcy = [0.0, np.pi, 0.0, 1.0, 0.0, T_final]
domain_stokes = [0.0, np.pi, -1.0, 0.0, 0.0, T_final]

# 采样个数（可根据实际情况调整）
N_interior_darcy = 1000
N_interior_stokes = 1000
N_ic_darcy = 200
N_bd_darcy = 200
N_ic_stokes = 200
N_bd_stokes = 200
N_interface = 200

"""内部采样点"""
# 达西区域
x_d_np, y_d_np, t_d_np = adaptive_sampler(N_interior_darcy, domain_darcy)
x_d = torch.tensor(x_d_np, dtype=torch.float32, device=device, requires_grad=True)
y_d = torch.tensor(y_d_np, dtype=torch.float32, device=device, requires_grad=True)
t_d = torch.tensor(t_d_np, dtype=torch.float32, device=device, requires_grad=True)

# 斯托克斯区域
x_s_np, y_s_np, t_s_np = adaptive_sampler(N_interior_stokes, domain_stokes)
x_s = torch.tensor(x_s_np, dtype=torch.float32, device=device, requires_grad=True)
y_s = torch.tensor(y_s_np, dtype=torch.float32, device=device, requires_grad=True)
t_s = torch.tensor(t_s_np, dtype=torch.float32, device=device, requires_grad=True)

"""初值条件采样"""
# 达西区域
x_ic_d_np, y_ic_d_np, t_ic_d_np = sampler_initial(N_ic_darcy, domain_darcy)
x_ic_d = torch.tensor(x_ic_d_np, dtype=torch.float32, device=device)
y_ic_d = torch.tensor(y_ic_d_np, dtype=torch.float32, device=device)
t_ic_d = torch.tensor(t_ic_d_np, dtype=torch.float32, device=device)

# 斯托克斯区域
x_ic_s_np, y_ic_s_np, t_ic_s_np = sampler_initial(N_ic_stokes, domain_stokes)
x_ic_s = torch.tensor(x_ic_s_np, dtype=torch.float32, device=device)
y_ic_s = torch.tensor(y_ic_s_np, dtype=torch.float32, device=device)
t_ic_s = torch.tensor(t_ic_s_np, dtype=torch.float32, device=device)

"""边界采样点"""
# 达西区域
x_bd_d_np, y_bd_d_np, t_bd_d_np = sampler_boundary(N_bd_darcy, domain_darcy)
x_bd_d = torch.tensor(x_bd_d_np, dtype=torch.float32, device=device)
y_bd_d = torch.tensor(y_bd_d_np, dtype=torch.float32, device=device)
t_bd_d = torch.tensor(t_bd_d_np, dtype=torch.float32, device=device)

# 斯托克斯区域
x_bd_s_np, y_bd_s_np, t_bd_s_np = sampler_boundary(N_bd_stokes, domain_stokes)
x_bd_s = torch.tensor(x_bd_s_np, dtype=torch.float32, device=device)
y_bd_s = torch.tensor(y_bd_s_np, dtype=torch.float32, device=device)
t_bd_s = torch.tensor(t_bd_s_np, dtype=torch.float32, device=device)

"""接口采样点（注意这里仅采样 x 与 t，接口处 y 固定为 0）"""
x_itf_np, y_itf_np, t_itf_np = sampler_interface(N_interface)
x_itf = torch.tensor(x_itf_np, dtype=torch.float32, device=device, requires_grad=True)
t_itf = torch.tensor(t_itf_np, dtype=torch.float32, device=device, requires_grad=True)

"""定义优化器（同时更新两个模型参数）"""
optimizer = optim.Adam(list(model_darcy.parameters()) + list(model_stokes.parameters()), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=750, gamma=0.85)


def compute_total_loss():
    # 达西区域残差及损失
    res_d = darcy_residual(model_darcy, x_d, y_d, t_d)
    loss_darcy_PDE = torch.mean(res_d ** 2)

    # 斯托克斯区域残差及损失
    r1, r2, incompress = stokes_residual(model_stokes, x_s, y_s, t_s)
    loss_stokes_PDE = torch.mean(r1 ** 2) + torch.mean(r2 ** 2) + torch.mean(incompress ** 2)

    # 初值条件损失（达西区域）
    X_ic_d = torch.cat([x_ic_d, y_ic_d, t_ic_d], dim=1)
    phi_ic_pred = model_darcy(X_ic_d)
    phi_ic_true = phi_D_exact(x_ic_d, y_ic_d, t_ic_d)
    loss_ic_darcy = torch.mean((phi_ic_pred - phi_ic_true) ** 2)

    # 初值条件损失（斯托克斯区域）
    X_ic_s = torch.cat([x_ic_s, y_ic_s, t_ic_s], dim=1)
    pred_ic_s = model_stokes(X_ic_s)
    u1_ic_pred = pred_ic_s[:, 0:1]
    u2_ic_pred = pred_ic_s[:, 1:2]
    p_ic_pred = pred_ic_s[:, 2:3]
    u1_ic_true, u2_ic_true = u_stokes_exact(x_ic_s, y_ic_s, t_ic_s)
    p_ic_true = p_stokes_exact(x_ic_s, y_ic_s, t_ic_s)
    loss_ic_stokes = torch.mean((u1_ic_pred - u1_ic_true) ** 2) + \
                     torch.mean((u2_ic_pred - u2_ic_true) ** 2) + \
                     torch.mean((p_ic_pred - p_ic_true) ** 2)

    # 边界条件损失（达西区域）
    X_bd_d = torch.cat([x_bd_d, y_bd_d, t_bd_d], dim=1)
    phi_bd_pred = model_darcy(X_bd_d)
    phi_bd_true = phi_D_exact(x_bd_d, y_bd_d, t_bd_d)
    loss_bd_darcy = torch.mean((phi_bd_pred - phi_bd_true) ** 2)

    # 边界条件损失（斯托克斯区域）
    X_bd_s = torch.cat([x_bd_s, y_bd_s, t_bd_s], dim=1)
    pred_bd_s = model_stokes(X_bd_s)
    u1_bd_pred = pred_bd_s[:, 0:1]
    u2_bd_pred = pred_bd_s[:, 1:2]
    p_bd_pred = pred_bd_s[:, 2:3]
    u1_bd_true, u2_bd_true = u_stokes_exact(x_bd_s, y_bd_s, t_bd_s)
    p_bd_true = p_stokes_exact(x_bd_s, y_bd_s, t_bd_s)
    loss_bd_stokes = torch.mean((u1_bd_pred - u1_bd_true) ** 2) + \
                     torch.mean((u2_bd_pred - u2_bd_true) ** 2) + \
                     torch.mean((p_bd_pred - p_bd_true) ** 2)

    # 接口条件损失
    r_normal, r_force, r_tangent = interface_residual(model_stokes, model_darcy, x_itf, t_itf)
    loss_interface = torch.mean(r_normal ** 2) + torch.mean(r_force ** 2) + torch.mean(r_tangent ** 2)

    total_loss = (model_darcy.loss_weights['residual'] * loss_darcy_PDE +
                  model_stokes.loss_weights['residual'] * loss_stokes_PDE +
                  model_darcy.loss_weights['initial'] * loss_ic_darcy +
                  model_stokes.loss_weights['initial'] * loss_ic_stokes +
                  model_darcy.loss_weights['boundary'] * loss_bd_darcy +
                  model_stokes.loss_weights['boundary'] * loss_bd_stokes +
                  model_darcy.loss_weights['interface'] * model_stokes.loss_weights['interface'] * loss_interface)
    return total_loss, loss_darcy_PDE, loss_stokes_PDE, loss_ic_darcy, loss_ic_stokes, loss_bd_darcy, loss_bd_stokes, loss_interface


# -------------------- 第一阶段训练：使用 Adam ----------------------
nIter = 6001
print_every = 500

for it in range(1, nIter):
    optimizer.zero_grad()

    loss, loss_darcy_PDE, loss_stokes_PDE, loss_ic_darcy, loss_ic_stokes, loss_bd_darcy, loss_bd_stokes, loss_interface = compute_total_loss()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if it % print_every == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Iter {it}, Total Loss: {loss.item():.3e}, "
              f"Darcy PDE: {loss_darcy_PDE.item():.3e}, "
              f"Stokes PDE: {loss_stokes_PDE.item():.3e}, "
              f"IC Darcy: {loss_ic_darcy.item():.3e}, "
              f"IC Stokes: {loss_ic_stokes.item():.3e}, "
              f"BC Darcy: {loss_bd_darcy.item():.3e}, "
              f"BC Stokes: {loss_bd_stokes.item():.3e}, "
              f"Interface: {loss_interface.item():.3e}, "
              f"LR: {current_lr:.1e}")
        print("\n")

# -------------------- 第二阶段精细优化：使用 LBFGS ----------------------
# 当 Adam 达到初步收敛后，切换至 LBFGS 进行精细调优
optimizer_lbfgs = optim.LBFGS(list(model_darcy.parameters()) + list(model_stokes.parameters()),
                              lr=1.0, max_iter=1000, max_eval=1000,
                              tolerance_grad=1e-6, tolerance_change=1e-10, history_size=100)

def closure():
    optimizer_lbfgs.zero_grad()
    loss_lbfgs, _, _, _, _, _, _, _ = compute_total_loss()
    loss_lbfgs.backward()
    return loss_lbfgs

print("Starting LBFGS optimization ...")
optimizer_lbfgs.step(closure)
print("LBFGS optimization finished.")

# ======================================================================
# 7. 可视化：所有子图放在同一张图中显示
# ======================================================================
# 求t=1时的解
# ---------- 达西区域 φ ----------
nx_d_plot, ny_d_plot = 100, 50
x_space_d = np.linspace(0, np.pi, nx_d_plot)
y_space_d = np.linspace(0, 1, ny_d_plot)
X_d_plot, Y_d_plot = np.meshgrid(x_space_d, y_space_d)
T_d_plot = np.ones_like(X_d_plot) * T_final

X_d_flat = torch.tensor(X_d_plot.flatten()[:, None], dtype=torch.float32, device=device)
Y_d_flat = torch.tensor(Y_d_plot.flatten()[:, None], dtype=torch.float32, device=device)
T_d_flat = torch.tensor(T_d_plot.flatten()[:, None], dtype=torch.float32, device=device)
inp_d_plot = torch.cat([X_d_flat, Y_d_flat, T_d_flat], dim=1)

with torch.no_grad():
    phi_pred_val = model_darcy(inp_d_plot).cpu().numpy().reshape(X_d_plot.shape)
    phi_true_val = phi_D_exact(X_d_flat, Y_d_flat, T_d_flat).cpu().numpy().reshape(X_d_plot.shape)
    phi_err = np.abs(phi_pred_val - phi_true_val)

# ---------- 斯托克斯区域（u1,u2,p）----------
nx_s_plot, ny_s_plot = 100, 50
x_space_s = np.linspace(0, np.pi, nx_s_plot)
y_space_s = np.linspace(-1, 0, ny_s_plot)
X_s_plot, Y_s_plot = np.meshgrid(x_space_s, y_space_s)
T_s_plot = np.ones_like(X_s_plot) * T_final

X_s_flat = torch.tensor(X_s_plot.flatten()[:, None], dtype=torch.float32, device=device)
Y_s_flat = torch.tensor(Y_s_plot.flatten()[:, None], dtype=torch.float32, device=device)
T_s_flat = torch.tensor(T_s_plot.flatten()[:, None], dtype=torch.float32, device=device)
inp_s_plot = torch.cat([X_s_flat, Y_s_flat, T_s_flat], dim=1)

with torch.no_grad():
    stokes_pred = model_stokes(inp_s_plot)
    u1_pred_val = stokes_pred[:, 0:1].cpu().numpy().reshape(X_s_plot.shape)
    u2_pred_val = stokes_pred[:, 1:2].cpu().numpy().reshape(X_s_plot.shape)
    p_pred_val = stokes_pred[:, 2:3].cpu().numpy().reshape(X_s_plot.shape)

    u1_true_val, u2_true_val = u_stokes_exact(X_s_flat, Y_s_flat, T_s_flat)
    u1_true_val = u1_true_val.cpu().numpy().reshape(X_s_plot.shape)
    u2_true_val = u2_true_val.cpu().numpy().reshape(X_s_plot.shape)
    p_true_val = p_stokes_exact(X_s_flat, Y_s_flat, T_s_flat).cpu().numpy().reshape(X_s_plot.shape)

    u1_err = np.abs(u1_pred_val - u1_true_val)
    u2_err = np.abs(u2_pred_val - u2_true_val)
    p_err = np.abs(p_pred_val - p_true_val)

# ---------- 绘图设置 ----------
# 设置4行3列的子图: 每行依次对应达西 φ, 斯托克斯 u1, 斯托克斯 u2, 斯托克斯 p
fig, axs = plt.subplots(4, 3, figsize=(15, 20))

# Row 1: 达西区域 φ
im = axs[0, 0].contourf(X_d_plot, Y_d_plot, phi_pred_val, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[0, 0])
axs[0, 0].set_title("PINNs predicted φ (Darcy)")

im = axs[0, 1].contourf(X_d_plot, Y_d_plot, phi_true_val, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[0, 1])
axs[0, 1].set_title("Exact φ (Darcy)")

im = axs[0, 2].contourf(X_d_plot, Y_d_plot, phi_err, 50, cmap='coolwarm')
fig.colorbar(im, ax=axs[0, 2])
axs[0, 2].set_title("Absolute Error φ (Darcy)")

# Row 2: Stokes区域 u1
im = axs[1, 0].contourf(X_s_plot, Y_s_plot, u1_pred_val, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[1, 0])
axs[1, 0].set_title("PINNs predicted u1 (Stokes)")

im = axs[1, 1].contourf(X_s_plot, Y_s_plot, u1_true_val, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[1, 1])
axs[1, 1].set_title("Exact u1 (Stokes)")

im = axs[1, 2].contourf(X_s_plot, Y_s_plot, u1_err, 50, cmap='coolwarm')
fig.colorbar(im, ax=axs[1, 2])
axs[1, 2].set_title("Absolute Error u1 (Stokes)")

# Row 3: Stokes区域 u2
im = axs[2, 0].contourf(X_s_plot, Y_s_plot, u2_pred_val, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[2, 0])
axs[2, 0].set_title("PINNs predicted u2 (Stokes)")

im = axs[2, 1].contourf(X_s_plot, Y_s_plot, u2_true_val, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[2, 1])
axs[2, 1].set_title("Exact u2 (Stokes)")

im = axs[2, 2].contourf(X_s_plot, Y_s_plot, u2_err, 50, cmap='coolwarm')
fig.colorbar(im, ax=axs[2, 2])
axs[2, 2].set_title("Absolute Error u2 (Stokes)")

# Row 4: Stokes区域 p
im = axs[3, 0].contourf(X_s_plot, Y_s_plot, p_pred_val, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[3, 0])
axs[3, 0].set_title("PINNs predicted p (Stokes)")

im = axs[3, 1].contourf(X_s_plot, Y_s_plot, p_true_val, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[3, 1])
axs[3, 1].set_title("Exact p (Stokes)")

im = axs[3, 2].contourf(X_s_plot, Y_s_plot, p_err, 50, cmap='coolwarm')
fig.colorbar(im, ax=axs[3, 2])
axs[3, 2].set_title("Absolute Error p (Stokes)")

plt.tight_layout()
plt.savefig("Stokes-Darcy耦合求解结果.pdf", format="pdf")
