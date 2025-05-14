import torch
import torch.nn as nn
import torch.fft
import matplotlib.pyplot as plt

# ===========================================
# Step 1: 构造模拟数据集
# ===========================================
def generate_dataset(n_samples=1000, Resolution=128):
    """
    生成一个模拟数据集，用来训练神经算子。
    输入函数 a(x) 是不同频率的正弦波；
    输出函数 u(x) 是 a(x)^2 加上一点噪声，用来模拟 PDE 的解。
    """
    x = torch.linspace(0, 2 * torch.pi, Resolution)  # 定义空间网格
    data_A = []  # 存放输入函数
    data_U = []  # 存放输出函数
    for _ in range(n_samples):
        freq = torch.rand(1) * 3 + 1     # 生成随机频率（控制输入变化幅度）
        phase = torch.rand(1) * 2 * torch.pi  # 生成随机相位
        A = torch.sin(freq * x + phase)       # 输入函数 A(x)
        U = A**2 + 0.1 * torch.randn_like(A)  # 输出函数 U(x)，模拟非线性方程解
        data_A.append(A)
        data_U.append(U)
    return torch.stack(data_A), torch.stack(data_U)  # 返回形状为 (N, resolution) 的张量

# ===========================================
# Step 2: 定义傅立叶卷积层(频域操作)
# ===========================================
class SpectralConv1d(nn.Module):
    """
        这是 FNO 的核心模块之一。它不是在时域上卷积，而是：
        1. 把输入变换到频域（通过 FFT）；
        2. 在频域对每个频率分量做线性变换；
        3. 再用 IFFT 回到时域。
        这样做可以捕捉全局信息（频率分量是全局的）。
    """
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes # 保留的频率分量的数量(保留前modes个频率分量进行处理)
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(torch.randn([in_channels, out_channels, modes], dtype=torch.cfloat)*scale)# 取复数类型的浮点数。初始化一个三维张量 weight，表示在频域里，用于每个频率分量的线性变换参数。[in_channels, out_channels, modes]
    def forward(self, x):
        batch_size, in_channels, resolution = x.shape # 样本个数(几个函数组)，输入通道数(函数组里有几个函数)，分辨率(每个函数采样几个点)
        x_ft = torch.fft.rfft(x)  # 快速傅里叶变换，但只保留实输入信号的正频率部分。逐个样本、逐个通道地，在最后一个空间维度上做傅里叶变换。

        # 创建输出频谱
        out_ft = torch.zeros(batch_size, self.out_channels, x_ft.shape[-1], device=x.device,dtype=torch.cfloat)

        # 手动进行逐频率点乘积
        for i in range(self.modes):
            out_ft[:, :, i] = torch.einsum("bi,io->bo", x_ft[:, :, i], self.weight[:, :, i])
            """
            bi：代表第一个输入张量 x_ft[:, :, i] 的维度（b = batch size, i = 输入通道数），即 [batch_size, in_channels]。

            io：代表第二个输入张量 self.weight[:, :, i] 的维度（i = 输入通道数, o = 输出通道数），即 [in_channels, out_channels]。

            bo：代表输出张量 out_ft[:, :, i] 的维度(b = batch size, o = 输出通道数)，即 [batch_size, out_channels]。
            
            x_ft[:, :, i]表示所有样本，所有通道的第i个频率分量。
            self.weight[:, :, i]表示所有样本，所有通道的第i个频率分量。
            这一步相当于原来所有样本，所有通道的第i个频率分量间是相互独立的，通过权重线性组合起来。
            """
        # IFFT 变换回时域
        x = torch.fft.irfft(out_ft, n=resolution)
        return x


# ===========================================
# Step 3: 构建完整的 FNO 网络
# ===========================================
class FNO1D(nn.Module):
    """
    整个结构包含：
    - 输入升维（线性层）；
    - 多层频域卷积（SpectralConv）+ 残差连接；
    - 输出投影层（线性降维）；
    """
    def __init__(self, modes, width):
        super().__init__()
        self.modes = modes
        self.width = width

        # 第一个线性层：将输入从 1 通道升维到 width 通道
        self.fc0 = nn.Linear(1, width)

        # 线性全连接后增加批归一化(一维)
        self.bn0 = nn.BatchNorm1d(width)

        # 三层频域卷积（类似于 ResNet 结构）
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)

        #每个卷积层后添加批归一化
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)
        self.bn3 = nn.BatchNorm1d(width)

        # 对应的残差连接（pointwise 1x1 卷积）
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)

        # 输出映射层：先变到中间维度，再输出到 1D 函数
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # 输入 x 是形状 (batch_size, resolution)，表示每个样本是一个一维函数
        # unsqueeze(-1) 将其形状变为 (batch_size, resolution, in_channels=1)
        # 即每个样本现在被视为包含一个通道的输入函数，方便送入后续线性层
        x = x.unsqueeze(-1)

        # 输入升维：从 1 → width
        x = self.fc0(x) #(batch_size,resolution, in_channels = width)，相当于每个函数组从一个函数变成了多个含函数
        x = x.permute(0, 2, 1) #转换成(batch_size, in_channels = width, resolution)形状方便卷积操作
        x = self.bn0(x)
        # 三层频域卷积 + 残差连接
        x1 = self.conv1(x) + self.w1(x)
        x1 = self.bn1(x1)
        x2 = self.conv2(x1) + self.w2(x1)
        x2 = self.bn2(x2)
        x3 = self.conv3(x2) + self.w3(x2)
        x3 = self.bn3(x3)

        # 转回(batch_size,resolution,in_channels = width)形状
        x = x3.permute(0, 2, 1)
        x = torch.relu(self.fc1(x)) #激活
        x = self.fc2(x) #输出到 1 维函数,(batch_size,resolution,1)形状
        return x.squeeze(-1) #压缩维度，变为(batch_size,resolution)形状

# ------------------------------
# 超参数
# ------------------------------
train_size = 800
test_size = 200
resolution = 128
batch_size = 20
learning_rate = 0.0005
epochs = 100

# 设置设备和随机种子
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# ------------------------------
# 数据准备
# ------------------------------
a, u = generate_dataset(n_samples=train_size + test_size, Resolution=resolution)
a_train, u_train = a[:train_size], u[:train_size]
a_test, u_test = a[train_size:], u[train_size:]
a_train = a_train.to(device)
u_train = u_train.to(device)
a_test = a_test.to(device)

train_dataset = torch.utils.data.TensorDataset(a_train, u_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# ------------------------------
# 模型、优化器、损失
# ------------------------------
model = FNO1D(modes=16, width=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# ------------------------------
# 开始训练
# ------------------------------
for epoch in range(epochs):
    model.train()
    for batch_idx,(a_batch, u_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        prediction_train = model(a_batch)
        loss = criterion(prediction_train, u_batch)
        loss.backward()
        # 添加梯度裁剪，限制梯度范数
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f"训练轮数：{epoch}", end=' ')
            print(f"已处理数据：{batch_idx * batch_size}/{train_size}", end=' ')
            print(f"({100. * batch_idx / len(train_loader):.0f}%) \tloss:{loss.item():.6f}")


# ------------------------------
# 可视化某个样本结果
# ------------------------------
with torch.no_grad():
    model.eval()
    prediction_test = model(a_test)
idx = 0  # 可以改成其他样本编号
a_test = a_test.cpu()
prediction_test = prediction_test.cpu()
plt.figure(figsize=(12, 4))
plt.plot(a_test[idx].numpy(), label="input a(x)")
plt.plot(u_test[idx].numpy(), label="exact u(x)")
plt.plot(prediction_test[idx].numpy(), '--', label="prediction u(x)")
plt.legend()
plt.title("FNO Prediction Result")
plt.grid(True)
plt.show()
