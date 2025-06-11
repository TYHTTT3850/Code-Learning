import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using device:",device)

# ======================================================================
# 1. 定义 MLP 网络
# ======================================================================
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=4, neurons=50):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, neurons))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(neurons, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 网络1：达西区域（输出标量 φ）
model_darcy = MLP(input_dim=3, output_dim=1, hidden_layers=4, neurons=50).to(device)
# 网络2：斯托克斯区域（输出 u1, u2, p）
model_stokes = MLP(input_dim=3, output_dim=3, hidden_layers=4, neurons=50).to(device)


# ======================================================================
# 2. 定义解析解（仅用于边界/初始条件监督，作为示例，此处不再用于 residual 计算）
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
# 3. 采样函数
# ======================================================================
def sampler(n, domain):
    # domain = [xmin, xmax, ymin, ymax, tmin, tmax]
    x = np.random.uniform(domain[0], domain[1], (n, 1))
    y = np.random.uniform(domain[2], domain[3], (n, 1))
    t = np.random.uniform(domain[4], domain[5], (n, 1))
    return x, y, t

def sampler_boundary(n, domain):
    xmin, xmax, ymin, ymax, tmin, tmax = domain
    Xs, Ys, Ts = [], [], []

    # 初始条件：t=tmin
    x0 = np.random.uniform(xmin, xmax, (n, 1))
    y0 = np.random.uniform(ymin, ymax, (n, 1))
    t0 = tmin * np.ones((n, 1))
    Xs.append(x0);
    Ys.append(y0);
    Ts.append(t0)

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
    X = torch.cat([x, y, t], dim=1)
    phi = model(X)

    grad_phi = torch.autograd.grad(phi, [x, y],
                                   grad_outputs=torch.ones_like(phi),
                                   create_graph=True)[0:2]
    phi_x = grad_phi[0]
    phi_y = grad_phi[1]

    phi_xx = torch.autograd.grad(phi_x, x, grad_outputs=torch.ones_like(phi_x), create_graph=True)[0]
    phi_yy = torch.autograd.grad(phi_y, y, grad_outputs=torch.ones_like(phi_y), create_graph=True)[0]
    laplacian = phi_xx + phi_yy

    # 右端项 f_D
    f_D = (torch.exp(y) - torch.exp(-y)) * torch.sin(x) * torch.exp(t)
    res = -laplacian - f_D
    return res

# 斯托克斯区域 PDE
def stokes_residual(model, x, y, t):
    X = torch.cat([x, y, t], dim=1)
    pred = model(X)
    u1 = pred[:, 0:1]
    u2 = pred[:, 1:2]
    p = pred[:, 2:3]

    u1_t = torch.autograd.grad(u1, t, grad_outputs=torch.ones_like(u1), create_graph=True)[0]
    u2_t = torch.autograd.grad(u2, t, grad_outputs=torch.ones_like(u2), create_graph=True)[0]

    u1_x = torch.autograd.grad(u1, x, grad_outputs=torch.ones_like(u1), create_graph=True)[0]
    u1_y = torch.autograd.grad(u1, y, grad_outputs=torch.ones_like(u1), create_graph=True)[0]
    u2_x = torch.autograd.grad(u2, x, grad_outputs=torch.ones_like(u2), create_graph=True)[0]
    u2_y = torch.autograd.grad(u2, y, grad_outputs=torch.ones_like(u2), create_graph=True)[0]

    # 计算应变率张量
    D11 = u1_x
    D22 = u2_y
    D12 = 0.5 * (u1_y + u2_x)

    # 应力张量，设 v=1： T = 2D(u) - pI
    T11 = 2.0 * D11 - p
    T22 = 2.0 * D22 - p
    T12 = 2.0 * D12

    T11_x = torch.autograd.grad(T11, x, grad_outputs=torch.ones_like(T11), create_graph=True)[0]
    T12_y = torch.autograd.grad(T12, y, grad_outputs=torch.ones_like(T12), create_graph=True)[0]
    T12_x = torch.autograd.grad(T12, x, grad_outputs=torch.ones_like(T12), create_graph=True)[0]
    T22_y = torch.autograd.grad(T22, y, grad_outputs=torch.ones_like(T22), create_graph=True)[0]

    divT1 = T11_x + T12_y
    divT2 = T12_x + T22_y

    # 右端项 f1 和 f2 (注意此处使用 np.pi 转为浮点数)
    f1 = (2 / np.pi) * torch.sin(2 * np.pi * y) * torch.cos(x) * torch.exp(t) + 4 * np.pi * torch.sin(
        2 * np.pi * y) * torch.cos(x) * torch.exp(t)
    f2 = 2 * (-2 + (1 / (np.pi ** 2)) * (torch.sin(np.pi * y)) ** 2) * torch.sin(x) * torch.exp(t) - 2 * torch.cos(
        2 * np.pi * y) * torch.sin(x) * torch.exp(t)

    r1 = u1_t - divT1 - f1
    r2 = u2_t - divT2 - f2

    incompress = u1_x + u2_y
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
    T_yy = 2 * u2_y - p
    r_force = -T_yy - phi
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
N_bd_darcy = 200
N_bd_stokes = 200
N_interface = 200

# 内部采样点
x_d_np, y_d_np, t_d_np = sampler(N_interior_darcy, domain_darcy)
x_s_np, y_s_np, t_s_np = sampler(N_interior_stokes, domain_stokes)

x_d = torch.tensor(x_d_np, dtype=torch.float32, device=device, requires_grad=True)
y_d = torch.tensor(y_d_np, dtype=torch.float32, device=device, requires_grad=True)
t_d = torch.tensor(t_d_np, dtype=torch.float32, device=device, requires_grad=True)

x_s = torch.tensor(x_s_np, dtype=torch.float32, device=device, requires_grad=True)
y_s = torch.tensor(y_s_np, dtype=torch.float32, device=device, requires_grad=True)
t_s = torch.tensor(t_s_np, dtype=torch.float32, device=device, requires_grad=True)

# 边界/初始采样点
x_bd_d_np, y_bd_d_np, t_bd_d_np = sampler_boundary(N_bd_darcy, domain_darcy)
x_bd_d = torch.tensor(x_bd_d_np, dtype=torch.float32, device=device)
y_bd_d = torch.tensor(y_bd_d_np, dtype=torch.float32, device=device)
t_bd_d = torch.tensor(t_bd_d_np, dtype=torch.float32, device=device)

x_bd_s_np, y_bd_s_np, t_bd_s_np = sampler_boundary(N_bd_stokes, domain_stokes)
x_bd_s = torch.tensor(x_bd_s_np, dtype=torch.float32, device=device)
y_bd_s = torch.tensor(y_bd_s_np, dtype=torch.float32, device=device)
t_bd_s = torch.tensor(t_bd_s_np, dtype=torch.float32, device=device)

# 接口采样点（注意这里仅采样 x 与 t，接口处 y 固定为 0）
x_itf_np, y_itf_np, t_itf_np = sampler_interface(N_interface)
x_itf = torch.tensor(x_itf_np, dtype=torch.float32, device=device, requires_grad=True)
t_itf = torch.tensor(t_itf_np, dtype=torch.float32, device=device, requires_grad=True)

# 定义优化器（同时更新两个模型参数）
optimizer = optim.Adam(list(model_darcy.parameters()) + list(model_stokes.parameters()), lr=1e-3)

# ======================================================================
# 6. 训练循环
# ======================================================================
nIter = 7501
print_every = 500

for it in range(1,nIter):
    optimizer.zero_grad()

    # 达西区域残差及损失
    res_d = darcy_residual(model_darcy, x_d, y_d, t_d)
    loss_darcy_PDE = torch.mean(res_d ** 2)

    # 斯托克斯区域残差及损失
    r1, r2, incompress = stokes_residual(model_stokes, x_s, y_s, t_s)
    loss_stokes_PDE = torch.mean(r1 ** 2) + torch.mean(r2 ** 2) + torch.mean(incompress ** 2)

    # 边界/初始条件损失（达西区域）
    X_bd_d = torch.cat([x_bd_d, y_bd_d, t_bd_d], dim=1)
    phi_bd_pred = model_darcy(X_bd_d)
    phi_bd_true = phi_D_exact(x_bd_d, y_bd_d, t_bd_d)
    loss_bd_darcy = torch.mean((phi_bd_pred - phi_bd_true) ** 2)

    # 边界/初始条件损失（斯托克斯区域）
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

    loss = loss_darcy_PDE + loss_stokes_PDE + loss_bd_darcy + loss_bd_stokes + loss_interface
    loss.backward()
    optimizer.step()

    if it % print_every == 0:
        print(f"Iter {it:05d}, Total Loss: {loss.item():.3e}, "
              f"Darcy PDE: {loss_darcy_PDE.item():.3e}, "
              f"Stokes PDE: {loss_stokes_PDE.item():.3e}, "
              f"BC Darcy: {loss_bd_darcy.item():.3e}, "
              f"BC Stokes: {loss_bd_stokes.item():.3e}, "
              f"Interface: {loss_interface.item():.3e}")

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

im = axs[0, 2].contourf(X_d_plot, Y_d_plot, phi_err, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[0, 2])
axs[0, 2].set_title("Absolute Error φ (Darcy)")

# Row 2: Stokes区域 u1
im = axs[1, 0].contourf(X_s_plot, Y_s_plot, u1_pred_val, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[1, 0])
axs[1, 0].set_title("PINNs predicted u1 (Stokes)")

im = axs[1, 1].contourf(X_s_plot, Y_s_plot, u1_true_val, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[1, 1])
axs[1, 1].set_title("Exact u1 (Stokes)")

im = axs[1, 2].contourf(X_s_plot, Y_s_plot, u1_err, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[1, 2])
axs[1, 2].set_title("Absolute Error u1 (Stokes)")

# Row 3: Stokes区域 u2
im = axs[2, 0].contourf(X_s_plot, Y_s_plot, u2_pred_val, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[2, 0])
axs[2, 0].set_title("PINNs predicted u2 (Stokes)")

im = axs[2, 1].contourf(X_s_plot, Y_s_plot, u2_true_val, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[2, 1])
axs[2, 1].set_title("Exact u2 (Stokes)")

im = axs[2, 2].contourf(X_s_plot, Y_s_plot, u2_err, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[2, 2])
axs[2, 2].set_title("Absolute Error u2 (Stokes)")

# Row 4: Stokes区域 p
im = axs[3, 0].contourf(X_s_plot, Y_s_plot, p_pred_val, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[3, 0])
axs[3, 0].set_title("PINNs predicted p (Stokes)")

im = axs[3, 1].contourf(X_s_plot, Y_s_plot, p_true_val, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[3, 1])
axs[3, 1].set_title("Exact p (Stokes)")

im = axs[3, 2].contourf(X_s_plot, Y_s_plot, p_err, 50, cmap='rainbow')
fig.colorbar(im, ax=axs[3, 2])
axs[3, 2].set_title("Absolute Error p (Stokes)")

plt.tight_layout()
plt.savefig("Stokes-Darcy耦合求解结果.pdf", format="pdf")
