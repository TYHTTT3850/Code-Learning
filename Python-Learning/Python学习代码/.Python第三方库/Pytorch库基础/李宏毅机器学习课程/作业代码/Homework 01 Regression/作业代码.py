# 作业目标：预测 COVID-19 每日病例数
"""
数据说明：
    test：893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
    train：2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
"""
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 库配置
torch.manual_seed(42069)
np.random.seed(42069)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42069)


class COVID19Dataset(Dataset):
    # 1、构造函数
    def __init__(self, csv_file, mode="train", target_only=False):
        """
        csv_file: 文件路径
        mode: 训练或测试
        target_only: 是否只返回标签数据
        """
        self.mode = mode
        with open(csv_file, 'r') as f:
            data = list(csv.reader(f)) #使用python内置的csv读取器读取文件并转化为二维列表
            data = np.array(data[1:]) #去掉表头
            data = data[:, 1:].astype(float) #去掉第一列并转换为浮点数

        if not target_only:
            #使用93个特征(即除了第三天检测阳性人数以外的全部特征)
            feats = list(range(93)) #创建列表[0, ... ,92]
        else:
            #仅使用40个州和两天的阳性数目作为特征
            feats = list(range(40)) + [57,75]

        if self.mode == "test": # 使用测试数据
            data = data[:, feats] #按照特征列提取数据
            self.data = torch.tensor(data, dtype=torch.float)#将数据转换为张量并转移到设备上
        else: # 使用训练数据
            target = data[:, -1] #提取最后一列数据(第三天检测阳性人数)
            data = data[:, feats]#按照特征列提取数据
            indexes = []
            # 按照10:1的数据划分训练集和验证集
            if self.mode == "train":
                indexes = [i for i in range(len(data)) if i % 10 != 0]
            elif self.mode == "validation":
                indexes = [i for i in range(len(data)) if i % 10 == 0]

            self.data = torch.tensor(data[indexes], dtype=torch.float) #提取训练或验证数据并转换为张量
            self.target = torch.tensor(target[indexes], dtype=torch.float) #提取训练或验证标签并转换为张量

        #对除了40个州以外的特征做标准化
        self.data[:, 40:] = (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1] #获取特征维度

        print(f"{self.mode}模式下的数据读取完毕")

    def __getitem__(self, index):
        if self.mode in ["train","validation"]:
            # 训练用数据
            return self.data[index], self.target[index]
        else:
            # 测试用数据
            return self.data[index]

    def __len__(self):
        return len(self.data)

def prepare_dataloader(csv_file, mode, batch_size, n_jobs=0, target_only=False):
    dataset = COVID19Dataset(csv_file, mode=mode, target_only=target_only)  #构造数据集
    dataloader = DataLoader(dataset, batch_size,shuffle=(mode == 'train'), drop_last=False,num_workers=n_jobs, pin_memory=True) #构造数据加载器
    return dataloader

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.criterion = nn.MSELoss()# 采用均方差作为损失函数

    def forward(self, x):
        return self.net(x).squeeze(1)

    def calculate_loss(self, predict, target):
        loss = self.criterion(predict, target)
        return loss

def validate(validate_set, model, device):
    model.eval()
    total_loss = 0
    for x, y in validate_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            predict = model(x)
            mse_loss = model.calculate_loss(predict, y)
        total_loss += mse_loss.detach().cpu().item() * len(x) #还原为当前batch的总损失
    total_loss = total_loss / len(validate_set.dataset) #数据集平均损失
    return total_loss

def train(train_set,validate_set, model, device, n_epochs, min_loss, early_stop):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    early_stop_count = 0
    epoch = 1
    finished_epochs = 0
    while epoch <= n_epochs:
        model.train()
        for x, y in train_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            predict = model(x)
            mse_loss = model.calculate_loss(predict, y)
            mse_loss.backward()
            optimizer.step()
        # 每一轮训练完之后都验证一下
        validate_loss = validate(validate_set, model, device)
        if validate_loss < min_loss:
            min_loss = validate_loss
            # 保存模型
            torch.save(model.state_dict(),"model.pth")
            print(f"Saving model to model.pth(epoch={epoch}, min_loss={min_loss})")
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count >= early_stop:
            finished_epochs = epoch
            break
        finished_epochs = epoch
        epoch += 1
    print(f"经过{finished_epochs}轮后，训练结束，early_stop_count={early_stop_count}")

def test(test_set, model, device):
    model.eval()
    predicts = []
    for x in test_set:
        x = x.to(device)
        with torch.no_grad():
            predict = model(x)
            predicts.append(predict.detach().cpu())
    predicts = torch.cat(predicts, dim=0).numpy() #将不同批次的预测数据首尾拼接
    return predicts

train_set = prepare_dataloader("covid.train.csv", 'train', batch_size=270, target_only=False)
validate_set = prepare_dataloader("covid.train.csv", 'validation', batch_size=270, target_only=False)
test_set = prepare_dataloader("covid.test.csv", 'test', batch_size=270, target_only=False)

model = NeuralNetwork(train_set.dataset.dim).to(device)
train(train_set, validate_set, model, device, n_epochs=3000, min_loss=1000,early_stop=200)

# 载入最好的模型
model.load_state_dict(torch.load("model.pth"))

def plot_validation(validate_set, model, device):
    model.eval()
    predicts = []
    targets = []
    for x, y in validate_set:
        x = x.to(device)
        with torch.no_grad():
            predict = model(x)
            predicts.append(predict.detach().cpu())
            targets.append(y.detach().cpu())

    predicts = torch.cat(predicts, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()

    # 绘制真实值与预测值的对比图
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, predicts, c='blue', alpha=0.5, label='Predictions')

    # 获取x和y的全局最小最大值，确保横纵轴范围一致
    min_val = min(targets.min(), predicts.min())
    max_val = max(targets.max(), predicts.max())

    # 理想情况下预测值等于真实值，画出 y=x 的红色曲线
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Validation Set: True vs Predicted Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig("在验证集上的预测与真实结果对比图.pdf",format='pdf')
    plt.show()

plot_validation(validate_set, model, device)

# 对测试集进行预测并保存结果
def predict_and_save(test_set, model, device, filename="predict.csv"):
    predicts = test(test_set, model, device)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(predicts):
            writer.writerow([i, p])
    print(f"测试集预测完成，结果已保存到 {filename}")

predict_and_save(test_set, model, device)
