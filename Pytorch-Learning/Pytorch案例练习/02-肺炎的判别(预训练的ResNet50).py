import os
import torch
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms,models
from torch.utils.data import DataLoader, random_split
from torchvision.models import ResNet50_Weights
os.environ["TORCH_HOME"] = "./pretrained_models" #改变预训练模型下载路径

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
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 检查
print(f"Total samples: {len(full_dataset)}")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(full_dataset.classes)

# 3、配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 4、引入预训练的ResNet
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5、训练
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
        accuracy_previous = 0
        accuracy_current = 0

        for batch_idx,(inputs,labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

        accuracy_current = correct / val_size
        print(f"第{epoch}轮精度：{accuracy_current:.6f}")

        if accuracy_current > best_val_acc:
            best_val_acc = accuracy_current
            torch.save(model.state_dict(), './best_model.pth')
            print(f"==> Best model saved at epoch {epoch}, accuracy: {accuracy_current:.6f}")

        accuracy_previous = accuracy_current
        accuracy_current = 0
        torch.cuda.empty_cache() # 清理GPU缓存
    if np.abs(accuracy_current - accuracy_previous) < 0.001:
        break

print(f"最高精度：{best_val_acc:.6f}")
