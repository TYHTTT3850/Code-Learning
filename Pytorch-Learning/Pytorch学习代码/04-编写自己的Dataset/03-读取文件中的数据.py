import torch
import pandas as pd
from torch.utils.data import Dataset,DataLoader

class CSVDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx,:] #基于位置索引,取出一行，数据类型为series
        # 分离特征和标签
        feature = torch.tensor(row.iloc[:-1].values,dtype=torch.float32)
        label = torch.tensor(row.iloc[-1],dtype=torch.float32)
        return feature,label

# 实例化数据集和数据加载器
dataset = CSVDataset("example_data.csv")
dataloader = DataLoader(dataset, batch_size=5,shuffle=True)

# 打印数据
for features,labels in dataloader:
    print(features)
    print(labels)