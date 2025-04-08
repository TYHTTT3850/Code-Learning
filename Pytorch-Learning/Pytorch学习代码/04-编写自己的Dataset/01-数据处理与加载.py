"""
在 PyTorch 中，处理和加载数据是深度学习训练过程中的关键步骤。

为了高效地处理数据，PyTorch 提供了强大的工具，包括 torch.utils.data.Dataset 和 torch.utils.data.DataLoader，帮助我们管理数据集、批量加载和数据增强等任务。

PyTorch 数据处理与加载的介绍：
1、自定义 Dataset：通过继承 torch.utils.data.Dataset 来加载自己的数据集。
2、DataLoader：DataLoader 按批次加载数据，支持多线程加载并进行数据打乱。
3、数据预处理与增强：使用 torchvision.transforms 进行常见的图像预处理和增强操作，提高模型的泛化能力。
4、加载标准数据集：torchvision.datasets 提供了许多常见的数据集，简化了数据加载过程。
5、多个数据源：通过组合多个 Dataset 实例来处理来自不同来源的数据。
"""

import torch
from torch.utils.data import Dataset

# 自定义 Dataset
class MyDataset(Dataset):
    """
    torch.utils.data.Dataset 是一个抽象类，允许你从自己的数据源中创建数据集。
    我们需要继承该类并实现以下两个方法：
        1、__len__(self)：返回数据集中的样本数量。
        2、__getitem__(self, idx)：通过索引返回一个样本。
    """
    def __init__(self,X_data,Y_data):
        """
        初始化数据集，X_data 和 Y_data 是两个列表或数组
        X_data: 输入特征
        Y_data: 目标标签
        """
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        return len(self.X_data) #返回数据的组数

    def __getitem__(self, idx):
        x = torch.tensor(self.X_data[idx]) # 转换为 tensor
        y = torch.tensor(self.Y_data[idx])
        return x, y

# 示例数据
x_train = [[1, 2], [3, 4], [5, 6], [7, 8]]  # 输入特征
y_train = [1, 0, 1, 0]  # 目标标签

# 创建数据集实例
dataset = MyDataset(x_train, y_train)

from torch.utils.data import DataLoader
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)# batch_size 设置每次加载的样本数量
"""
DataLoader工作原理：
1、DataLoader 本身不保存数据，它只是一个包装器，围绕你提供的 Dataset ，按你的设定来访问、打乱、分批、加载数据。
2、加载数据的流程图：
    CSVDataset(N个样本)
        ↓
    shuffle=True → 打乱索引 → 比如 [3, 7, 0, 1, 6, ...]
        ↓
    batch_size=5 → 将索引分成几个 batch → [[3,7,0,1,6], [2,4,5,8,9], ...]
        ↓
    调用 __getitem__ 方法取出第 i 个 batch 中每个索引的数据
        ↓
    将第 i 个 batch 中每个索引的数据合成为张量
        ↓
    每个 batch 的结构为：(features, labels)
        ↓
    每个 batch 中的数据堆叠成为：(batch1的数据,batch2的数据, ... )
3、总的来说，DataLoader 加载数据后产生的结果的结构：(batch1的数据,batch2的数据, ... )，每个batch的结构又是(features,labels)
"""

# 打印加载的数据
for batch_idx, (inputs, labels) in enumerate(train_loader):
    print(f'Batch {batch_idx + 1}:')
    print(f'Inputs: {inputs}')
    print(f'Labels: {labels}')
