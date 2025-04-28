import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# 1、图像预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 2、加载整个数据集
full_dataset = datasets.ImageFolder(root='./Lung_Xray_Image', transform=transform)

# 划分训练集和测试集
train_size = int(0.8 * len(full_dataset))  # 80%训练
val_size = len(full_dataset) - train_size  # 20%验证

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 检查
print(f"Total samples: {len(full_dataset)}")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(full_dataset.classes)

# 3、配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 4、定义模型
class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 输入通道是1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #默认 stride = kernel_size

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 36 * 36, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes) #输出4类
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  #展平
        x = self.fc_layers(x)
        return x

model = CNN(num_classes=4).to(device)

# 5、损失函数和优化器
criterion = nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6、训练过程
num_epochs = 20 # 训练轮数
best_val_acc = 0.0 # 最佳精度，初始化为0

for epoch in range(num_epochs):
    model.train()

    for batch_idx,(inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"训练轮数：{epoch}",end=' ')
            print(f"已处理数据：{batch_idx*len(inputs)}/{train_size}",end=' ')
            print(f"({100.*batch_idx/len(train_loader):.0f}%) \tloss:{loss.item():.6f}")

    # 每一轮训练完都评估
    with torch.no_grad():
        model.eval()
        correct = 0
        accuracy = 0

        for batch_idx,(inputs,labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

        accuracy = correct / val_size
        print(f"第{epoch}轮精度：{accuracy}",end=' ')

        if accuracy > best_val_acc:
            best_val_acc = accuracy
            torch.save(model.state_dict(), './best_model.pth')
            print(f"==> Best model saved at epoch {epoch}, accuracy: {accuracy:.6f}")

print(f"最高精度：{best_val_acc:.6f}")