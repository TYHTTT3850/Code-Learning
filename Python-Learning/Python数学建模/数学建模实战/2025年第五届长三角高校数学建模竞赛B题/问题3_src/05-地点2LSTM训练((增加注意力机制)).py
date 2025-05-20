import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.cuda.empty_cache()

# 超参数
torch.manual_seed(42)
SEQ_LEN = 8  # 使用过去8小时
PRED_HORIZON = 4  # 预测4小时后
BATCH_SIZE = 64
EPOCHS = 700
LR = 0.001

# 1. 加载数据
df = pd.read_excel("../地点2训练数据.xlsx", parse_dates=["采集时刻"])
df.sort_values("采集时刻", inplace=True)

# 2. 选取特征列
features = ['热泵功率(kw)', '供温(℃)', '回温(℃)', '补水流速(m3h)', '设定温度(℃)','hour','weekday','供回温差','热泵是否开启','室内温度变化', '环境温度(℃)', '室内温度(℃)']
data = df[features].values
input_size = len(features)

# 3. 归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 4. 构造样本数据
X, y = [], []
for i in range(len(data_scaled) - SEQ_LEN - PRED_HORIZON):
    x_seq = data_scaled[i:i + SEQ_LEN]
    y_target = data_scaled[i + SEQ_LEN + PRED_HORIZON - 1][-1]  # 室内温度在最后一列
    X.append(x_seq)
    y.append(y_target)

X = np.array(X)  # shape: [N, SEQ_LEN, 7]
y = np.array(y)  # shape: [N,]

# 5. 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1).to(device)


# 6. 模型定义
class LSTMWithAttention(nn.Module):
    def __init__(self, input_size=input_size, hidden_size=64, num_layers=2, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=0.2,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.attn = nn.Linear(lstm_output_size, 1)  # Attention打分层
        self.fc = nn.Linear(lstm_output_size, 1)    # 输出层

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [B, T, H]
        weights = torch.softmax(self.attn(lstm_out), dim=1)  # [B, T, 1]
        context = torch.sum(weights * lstm_out, dim=1)       # [B, H]
        out = self.fc(context)                               # [B, 1]
        return out


model = LSTMWithAttention().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 7. 模型训练
for epoch in range(EPOCHS):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.6f}")

# 8. 模型评估
model.eval()
with torch.no_grad():
    y_pred = model(X_test).cpu().numpy()
    y_true = y_test.cpu().numpy()

# 9. 反归一化（只对室内温度一列）
indoor_temp_index = features.index("室内温度(℃)")
min_temp = scaler.data_min_[indoor_temp_index]
max_temp = scaler.data_max_[indoor_temp_index]

y_pred_rescaled = y_pred * (max_temp - min_temp) + min_temp
y_true_rescaled = y_true * (max_temp - min_temp) + min_temp

# 10. 可视化

# 获取对应的预测目标时间（采集时刻 + 4 小时）
timestamps = df['采集时刻'].values
# y 是从第 SEQ_LEN + PRED_HORIZON 开始生成的
start_idx_in_timestamps = SEQ_LEN + PRED_HORIZON
# y_test 是 y 的后 20%，找到它在 y 中的起始位置
test_start_idx_in_y = int(len(y) * 0.8)
# 对应到原始 timestamps 的起始点
test_start_time_idx = start_idx_in_timestamps + test_start_idx_in_y
# 截取对应的时间戳作为横坐标
target_times = timestamps[test_start_time_idx : test_start_time_idx + len(y_test)]

# 训练结果可视化
plt.figure(figsize=(12, 5))
plt.plot(target_times, y_true_rescaled, label="exact")
plt.plot(target_times, y_pred_rescaled, label="prediction")
plt.xlabel("date")
plt.ylabel("indoor temperature")
plt.legend()
plt.tight_layout()
plt.savefig("../问题3/地点2LSTM拟合结果(增加注意力机制).pdf",format="pdf")