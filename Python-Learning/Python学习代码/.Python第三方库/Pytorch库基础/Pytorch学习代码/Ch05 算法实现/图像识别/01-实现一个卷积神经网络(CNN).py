import torch
import torch.nn.functional as F
from torchvision import transforms,datasets
from torch.utils.data import DataLoader


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

# 使用 torchvision 提供的 MNIST 数据集，加载和预处理数据。
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True) #root：保存的目录
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 定义CNN模型
class CNN(torch.nn.Module):
    def __init__(self):
        """
        使用torch.nn.Conv2d创建卷积层，参数解释：
            1、in_channels：输入图像的通道数。对于灰度图像，只有一个通道(黑白图像)，所以是 1。对于彩色图像(RGB)，则是 3。
            2、out_channels：即卷积核的个数。每个卷积核提取输入的一种特征，有几个卷积核就有几个不同的特征图(feature maps)。这个值通常越大，网络越能提取复杂的特征。
            3、kernel_size：卷积核的大小。
        """
        super().__init__()
        # 第一个卷积层
        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=10,kernel_size=5)
        # 第二个卷积层
        self.conv2 = torch.nn.Conv2d(in_channels=10,out_channels=20,kernel_size=5)
        # 定义池化层
        self.pool = torch.nn.MaxPool2d(2,2)
        # Dropout，用于防止过拟合
        self.dropout = torch.nn.Dropout2d()
        # 定义全连接层
        """
        输入尺寸：1×28×28
        第一个 5×5 卷积 + 2×2池化后尺寸：10×(28-5+1)/2×(28-5+1)/2=10×12×12
        第二个第一个 5×5 卷积 + 2×2池化后尺寸：20×(12-5+1)/2×(12-5+1)/2=20×4×4
        """
        self.fc1 = torch.nn.Linear(in_features=320,out_features=50)
        self.fc2 = torch.nn.Linear(in_features=50,out_features=10) # 有10个类别所以输出特征为10
    def forward(self, x):
        # 第一层卷积 + ReLU 激活函数 + 最大池化
        x = self.pool(F.relu(self.conv1(x)))
        # 第二层卷积 → Dropout → ReLU → 池化
        x = self.pool(F.relu(self.dropout(self.conv2(x))))
        # 展平
        x = x.view(-1, 320)
        # 全连接层 + ReLU
        x = F.relu(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x

# 实例化模型并转移到指定设备
model = CNN().to(device)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 定义如何训练
def train(Model,Device,Train_Loader,Optimizer,Epoch):
    Model.train()
    for epoch in range(Epoch):
        for batch_idx, (data, target) in enumerate(Train_Loader):
            data, target = data.to(Device), target.to(Device)
            Optimizer.zero_grad()
            output = Model(data) # 预测值
            loss = criterion(output, target)
            loss.backward()
            Optimizer.step()
            if batch_idx % 100 == 0:
                print(f"训练轮数：{epoch}",end=' ')
                print(f"已处理数据：{batch_idx*len(data)}/{len(Train_Loader.dataset)}",end=' ')
                print(f"({100.*batch_idx/len(Train_Loader):.0f}%) \tloss:{loss.item():.6f}")

# 定义如何测试
def test(Model,Device,Test_Loader):
    Model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in Test_Loader:
            data, target = data.to(Device), target.to(Device)
            output = Model(data) # 10个类别的概率
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True) #求概率最大的那个类别
            correct += pred.eq(target.view_as(pred)).sum().item() #统计这一批次图片中预测正确的数量
    test_loss /= len(Test_Loader.dataset) # 计算整个数据集的平均损失
    accuracy = correct / len(Test_Loader.dataset) # 计算对整个数据集预测精确度
    print(f"Test set Average loss:{test_loss:.6f}, Accuracy:{accuracy:.6f}")

# 训练并测试
train(model, device, train_loader, optimizer, Epoch=6)

test(model, device, test_loader)
